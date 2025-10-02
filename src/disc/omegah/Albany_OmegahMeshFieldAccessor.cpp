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
addStateStruct(const Teuchos::RCP<StateStruct>& st)
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

  // nodal/nodal_parameter states  will be warranted a dof mgr later,
  // while elem_sis are states that can be processed by LoadStateField
  // and LoadSideSetStateField evaluators
  switch(st->entity) {
    case StateStruct::NodalDistParameter:
      nodal_parameter_sis.push_back(st);
      nodal_sis.push_back(st);
      break;
    case StateStruct::NodalDataToElemNode:
      nodal_sis.push_back(st);
      elem_sis.push_back(st);
      break;
    case StateStruct::ElemData:   [[fallthrough]];
    case StateStruct::ElemNode:   [[fallthrough]];
    case StateStruct::QuadPoint:
      elem_sis.push_back(st);
      break;
    default:
      throw std::runtime_error("Error! Unrecognized/unsupported state entity type.\n");
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

void OmegahMeshFieldAccessor::createStateArrays (const WorksetArray<int>& worksets_sizes)
{
  int num_elems = m_mesh->nelems();
  int num_nodes = m_mesh->nverts();

  // We don't have workset sizes in here, so use single workset
  elemStateArrays.resize(1);
  nodeStateArrays.resize(1);

  // Elem states
  int num_ws = worksets_sizes.size();
  for (const auto& st : elem_sis) {
    auto data = m_tags.at(st->name).data();
    auto dim = st->dim;
    int stride = 1;
    for (auto d : dim) stride *= d;
    stride /= dim[0];

    for (int ws=0; ws<num_ws; ++ws) {
      switch (dim.size()) {
        case 1:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems); break;
        case 2:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems,dim[1]); break;
        case 3:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems,dim[1],dim[2]); break;
        case 4:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems,dim[1],dim[2],dim[3]); break;
        default:
          throw std::runtime_error("Error! Unsupported rank for elem state '" + st->name + "'.\n");
      }
      data += worksets_sizes[ws]*stride;
    }
  }

  // Nodal states
  // NOTE: nodal states have just 1 workset
  for (const auto& st : nodal_sis) {
    auto data = m_tags.at(st->name).data();
    auto dim = st->dim;
    if (st->entity != StateStruct::NodalData) {
      // Add an elem state array, which the SaveStateField/SaveSideSetStateField evaluators will use
      for (int ws=0; ws<worksets_sizes.size(); ++ws) {
        auto& state = elemStateArrays[ws][st->name];
        switch (dim.size()) {
          case 2:
            state.resize(st->name,worksets_sizes[ws],dim[1]); break;
          case 3:
            state.resize(st->name,worksets_sizes[ws],dim[1],dim[2]); break;
          case 4:
            state.resize(st->name,worksets_sizes[ws],dim[1],dim[2],dim[3]); break;
          default:
            throw std::runtime_error("Error! Unsupported rank for elem state '" + st->name + "'.\n");
        }
      }
      // Remove <Cell> extent from dim, so we can use for sizing the nodeStateArray
      dim.erase(dim.begin());
    }
    switch (dim.size()) {
      case 1:
        nodeStateArrays[0][st->name].reset_from_dev_ptr(data,num_nodes); break;
      case 2:
        nodeStateArrays[0][st->name].reset_from_dev_ptr(data,num_nodes,dim[1]); break;
      case 3:
        nodeStateArrays[0][st->name].reset_from_dev_ptr(data,num_nodes,dim[1],dim[2]); break;
      default:
        throw std::runtime_error("Error! Unsupported rank for node state '" + st->name + "'.\n");
    }
  }

  // Global states
  for (const auto& st : global_sis) {
    auto& state = globalStates[st->name];
    if (st->dim.size()==1) {
      state.reset_from_host_ptr(&mesh_scalar_states[st->name],1);
    } else if (st->dim.size()==1) {
      state.reset_from_host_ptr(mesh_vector_states[st->name].data(),st->dim[0]);
    } else {
      throw std::runtime_error("Error! Unsupported rank for global state '" + st->name + "'.\n");
    }
  }
}

void OmegahMeshFieldAccessor::transferNodeStatesToElemStates ()
{
  int num_elems = m_mesh->nelems();
  auto elem_nodes = m_mesh->ask_elem_verts();
  auto elem_nodes_h = hostRead(elem_nodes);
  int num_elem_nodes = elem_nodes.size() / num_elems;

  for (const auto& st : nodal_sis) {
    if (st->entity!=StateStruct::NodalDataToElemNode)
      continue;
    const auto& dim = st->dim;
    const auto rank = st->dim.size();

    const auto& node_state = m_mesh->get_tag<ST>(0,st->name)->array();
          auto& elem_state = elemStateArrays[0][st->name];
    switch (rank) {
      case 2:
        elem_state.resize(st->name,num_elems,dim[1]); break;
      case 3:
        elem_state.resize(st->name,num_elems,dim[1],dim[2]); break;
      case 4:
        elem_state.resize(st->name,num_elems,dim[1],dim[2],dim[3]); break;
      default:
        throw std::runtime_error("Error! Unsupported rank for node state '" + st->name + "'.\n");
    }

    auto& elem_state_h = elem_state.host();
    auto  node_state_h = hostRead(node_state);
    for (int i=0; i<num_elems; ++i) {
      for (int j=0; j<num_elem_nodes; ++j) {
        switch(rank) {
          case 2:
            elem_state_h(i, j) = node_state_h[elem_nodes_h[i*num_elem_nodes+j]];
            break;
          case 3:
            for (size_t k=0; k<dim[2]; ++k) {
              auto offset = i*num_elem_nodes*dim[2]+j*dim[2]+k;
              elem_state_h(i, j, k) = node_state_h[elem_nodes_h[offset]];
            } break;
          case 4:
            for (size_t k=0; k<dim[2]; ++k) {
              for (size_t l=0; l<dim[3]; ++l) {
                auto offset = i*num_elem_nodes*dim[2]*dim[3]+j*dim[2]*dim[3]+k*dim[3]+l;
                elem_state_h(i, j, k, l) = node_state_h[elem_nodes_h[offset]];
              }
            } break;
        }
      }
    }
    elem_state.sync_to_dev();
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
