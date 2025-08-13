#include "Albany_OmegahMeshFieldAccessor.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_OmegahUtils.hpp"

namespace Albany {

OmegahMeshFieldAccessor::
OmegahMeshFieldAccessor (const Teuchos::RCP<Omega_h::Mesh>& mesh)
 : m_mesh (mesh)
{
  // Nothing to do here
}

void OmegahMeshFieldAccessor::
addFieldOnMesh (const std::string& name,
                const int entityDim,
                const int numComps)
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_mesh->has_tag(entityDim,name), std::logic_error,
      "Error! Tag '" + name + "' is already defined on the mesh.\n");
  Omega_h::Write<ST> f(m_mesh->nents(entityDim)*numComps,name);
  m_mesh->add_tag<ST>(entityDim,name,numComps,f,false);
  m_tags[name] = f;
}

void OmegahMeshFieldAccessor::
setFieldOnMesh (const std::string& name,
                const int entityDim,
                const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  auto tag = m_mesh->get_tag<ST>(entityDim,name);
  TEUCHOS_TEST_FOR_EXCEPTION (tag->ncomps()!=mv->domain()->dim(), std::logic_error,
      "Error! Cannot copy MV on mesh tag, since the number of vecs does not match the tag ncomps.\n"
      "  - tag name: " + name + "\n"
      "  - tag ncomps: " << tag->ncomps() << "\n"
      "  - MV num vecs: " << mv->domain()->dim() << "\n");

  // Create 1d view of input MV
  auto dev_mv = getDeviceData(mv);

  // Copy into tag. WARNING: tags have entity id striding slower, while the input mv makes
  // entity id stride faster (it's a 2d view with layout left)
  int ncmps = dev_mv.extent(1);
  int nents = dev_mv.extent(0);
  Kokkos::RangePolicy<> policy(0,nents*ncmps);
  auto tag_view = m_tags.at(name).view();
  auto lambda = KOKKOS_LAMBDA(int idx) {
    int ient = idx % nents;
    int icmp = idx / nents;
    tag_view (ient*ncmps + icmp) = dev_mv(ient,icmp);
  };
  Kokkos::parallel_for(policy,lambda);
}

void OmegahMeshFieldAccessor::
addStateStructs(const StateInfoStruct& sis)
{
  auto product = [](const auto& vec, int start) {
    return std::accumulate(vec.begin()+start, vec.end(), 1, std::multiplies<int>());
  };

  auto get_ent_dim_and_ncomp = [&] (const StateStruct& st) {
    std::pair<int,int> dim_ncomp;
    switch (st.stateType()) {
      case StateStruct::NodeState:
        dim_ncomp.first  =  0;
        dim_ncomp.second = product(st.dim,st.entity==StateStruct::NodalData ? 1 : 2);
        break;
      case StateStruct::ElemState:
        dim_ncomp.first  = m_mesh->dim();
        dim_ncomp.second = product(st.dim,0);
        break;
      default:
        throw std::runtime_error(
            "Error! Invalid/unsupported state type.\n"
            "  - state name: " + st.name + "\n");
    }
    return dim_ncomp;
  };

  for (const auto& st : sis) {
    // These will be warranted a dof mgr later
    if (st->entity==StateStruct::NodalDistParameter) {
      nodal_parameter_sis.push_back(st);
      nodal_sis.push_back(st);
    } else if (st->entity==StateStruct::NodalDataToElemNode) {
      nodal_sis.push_back(st);
    }

    auto dim_ncomp = get_ent_dim_and_ncomp(*st);
    int ent_dim = dim_ncomp.first;
    int ncomp = dim_ncomp.second;
    if (ent_dim==-1) {
      if (ncomp==1) {
        mesh_scalar_states.emplace(st->name,st->initValue);
      } else {
        mesh_vector_states[st->name].resize(ncomp,st->initValue);
      }
    } else {
      addFieldOnMesh(st->name,ent_dim,ncomp);
    }

    if (st->layered) {
      // Need to also add the global vector state for the normalized layers coords
      auto nlayers = st->dim.back();
      mesh_vector_states[st->name+"_NLC"].resize(nlayers);
    }
  }
}

void OmegahMeshFieldAccessor::
fillVector (Thyra_Vector&        field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }

  // For now, assume only 1 entity has dofs (verts or elems)
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()!=1, std::runtime_error,
      "[OmegahMeshFieldAccessor::fillVector] Only P0 or P1 fields supported for now.\n");
  auto dim = entity_dims_with_dofs[0];

  // TODO: you may want to do this on device, but you need an overload of
  //       the getNonconstDeviceData util that accepts a ref not an RCP.
  const auto& elem_dof_lids = field_dof_mgr->elem_dof_lids().host();
  const auto& elems = field_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int nelems = elems.size();
  const int ncomps = field_dof_mgr->getNumFields();

  auto owned_h = hostRead(m_mesh->owned(dim));
  auto mesh_data_h  = hostRead(m_mesh->get_array<ST>(dim,field_name));
  auto thyra_data_h = getNonconstLocalData(field_vector);
  auto elem_ents_h = hostRead(m_mesh->ask_down(m_mesh->dim(),dim).ab2b);
  auto nents_per_elem = elem_ents_h.size() / nelems;
  for (int ielem=0; ielem<nelems; ++ielem) {
    for (int icmp=0; icmp<field_dof_mgr->getNumFields(); ++icmp) {
      const auto& offsets = field_dof_mgr->getGIDFieldOffsets(icmp);
      for (int ient=0; ient<nents_per_elem; ++ient) {
        auto ent_lid = elem_ents_h[ielem*nents_per_elem+ient];
        if (overlapped or owned_h[ent_lid]) {
          auto lid = elem_dof_lids(ielem,offsets[ient]);
          // We may have lid<0 if the dof mgr is restricted to a mesh part
          // This happens for dirichlet BCs fields.
          if (lid>=0) {
            thyra_data_h[lid] = mesh_data_h[ent_lid*ncomps + icmp];
          }
        }
      }
    }
  }
}

void OmegahMeshFieldAccessor::
saveVector (const Thyra_Vector&  field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }

  // For now, assume only 1 entity has dofs (verts or elems)
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()!=1, std::runtime_error,
      "[OmegahMeshFieldAccessor::fillVector] Only P0 or P1 fields supported for now.\n");
  auto dim = entity_dims_with_dofs[0];

  TEUCHOS_TEST_FOR_EXCEPTION (not m_mesh->has_tag(dim,field_name), std::runtime_error,
      "Error! Field '" + field_name + "' was not found as a tag in the mesh.\n");

  // TODO: you may want to do this on device, but you need an overload of
  //       the getNonconstDeviceData util that accepts a ref not an RCP.
  const auto& elem_dof_lids = field_dof_mgr->elem_dof_lids().host();
  const auto& elems = field_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int nelems = elems.size();
  const int ncomps = field_dof_mgr->getNumFields();

  auto mesh_data_h = hostWrite<ST>(m_mesh->nents(dim)*ncomps,field_name);
  auto owned_h = hostRead(m_mesh->owned(dim));
  auto thyra_data_h = getLocalData(field_vector);
  auto elem_ents_h = hostRead(m_mesh->ask_down(m_mesh->dim(),dim).ab2b);
  auto nents_per_elem = elem_ents_h.size() / nelems;
  for (int ielem=0; ielem<nelems; ++ielem) {
    for (int icmp=0; icmp<field_dof_mgr->getNumFields(); ++icmp) {
      const auto& offsets = field_dof_mgr->getGIDFieldOffsets(icmp);
      for (int ient=0; ient<nents_per_elem; ++ient) {
        auto ent_lid = elem_ents_h[ielem*nents_per_elem+ient];
        if (overlapped or owned_h[ent_lid]) {
          auto lid = elem_dof_lids(ielem,offsets[ient]);
          // We may have lid<0 if the dof mgr is restricted to a mesh part
          // This happens for dirichlet BCs fields.
          if (lid>=0) {
            mesh_data_h[ent_lid*ncomps + icmp] = thyra_data_h[lid];
          }
        }
      }
    }
  }

  m_mesh->set_tag(dim,field_name,read(mesh_data_h.write()),false);
}

} // namespace Albany
