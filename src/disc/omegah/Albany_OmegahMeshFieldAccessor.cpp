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
                const FE_Type fe_type,
                const int numComps)
{

  switch (fe_type) {
    case FE_Type::HGRAD:
    {
      Omega_h::Write<ST> f(m_mesh->nverts(),name);
      m_mesh->add_tag<ST>(OMEGA_H_VERT,name,numComps,f,false);
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (fe_type!=FE_Type::HGRAD, std::runtime_error,
          "[OmegahMeshFieldAccessor::addFieldOnMesh] Error! Unsupported value for fe_type.\n"
          "  - input value: " << e2str(fe_type) << "\n"
          "  - supported values: " << e2str(FE_Type::HGRAD) << "\n");
  }
}

void OmegahMeshFieldAccessor::
addStateStructs(const Teuchos::RCP<StateInfoStruct>& sis)
{
  // Extract underlying integer value from an enum
  auto e2i = [] (const StateStruct::MeshFieldEntity e) {
    return static_cast<typename std::underlying_type<StateStruct::MeshFieldEntity>::type>(e);
  };

  auto get_nodal_state_ncomps = [](const StateStruct& st) {
    const auto& dims = st.dim;
    int ncomps = -1;
    switch (dims.size()) {
      case 2:
        ncomps = 1;
        break;
      case 3:
        ncomps = dims[2];
        break;
      case 4:
        ncomps = dims[2]*dims[3];
        break;
      default:
        throw std::runtime_error(
            "Error! Unsupported rank for state field.\n"
            "  - state name: " + st.name + "\n"
            "  - state rank: " + std::to_string(dims.size()) + "\n");
    }
    return ncomps;
  };
  for (const auto& st : *sis) {
    if (st->entity==StateStruct::NodalDataToElemNode) {
      nodal_sis.push_back(st);
    } else if (st->entity==StateStruct::NodalDistParameter) {
      nodal_sis.push_back(st);
      nodal_parameter_sis.push_back(st);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
          "Error! Unsupported state mesh entity type for Omega_h.\n"
          "  - input value: " << e2i(st->entity) << "\n"
          "  - supported values:\n"
          "       NodalDataToElemNode: " << e2i(StateStruct::NodalDataToElemNode) << "\n"
          "       NodalDistParameter : " << e2i(StateStruct::NodalDistParameter)  << "\n");
    }

    // TODO: this needs to change for non-nodal states
    addFieldOnMesh(st->name,FE_Type::HGRAD,get_nodal_state_ncomps(*st));
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

  if(! m_mesh->has_tag(dim,field_name)) {
    m_mesh->add_tag(dim,field_name, ncomps, read(mesh_data_h.write()));
  } else {
    m_mesh->set_tag(dim,field_name,read(mesh_data_h.write()),false);
  }
}

} // namespace Albany
