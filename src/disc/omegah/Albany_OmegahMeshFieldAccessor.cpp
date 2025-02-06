#include "Albany_OmegahMeshFieldAccessor.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_OmegahUtils.hpp"

#include "Omega_h_adapt.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_file.hpp"
namespace debug {
template <typename T>
void printTagInfo(Omega_h::Mesh mesh, std::ostringstream& oss, int dim, int tag, std::string type) {
    auto tagbase = mesh.get_tag(dim, tag);
    auto array = Omega_h::as<T>(tagbase)->array();

    Omega_h::Real min = get_min(array);
    Omega_h::Real max = get_max(array);

    oss << std::setw(18) << std::left << tagbase->name().c_str()
        << std::setw(5) << std::left << dim 
        << std::setw(7) << std::left << type 
        << std::setw(5) << std::left << tagbase->ncomps() 
        << std::setw(10) << std::left << min 
        << std::setw(10) << std::left << max 
        << "\n";
}

void printAllTags(Omega_h::Mesh& mesh) {
  std::ostringstream oss;
  // always print two places to the right of the decimal
  // for floating point types (i.e., imbalance)
  oss.precision(2);
  oss << std::fixed;

  oss << "\nTag Properties by Dimension: (Name, Dim, Type, Number of Components, Min. Value, Max. Value)\n";
  for (int dim=0; dim <= mesh.dim(); dim++) {
    for (int tag=0; tag < mesh.ntags(dim); tag++) {
      auto tagbase = mesh.get_tag(dim, tag);
      if (tagbase->type() == OMEGA_H_I8)
        printTagInfo<Omega_h::I8>(mesh, oss, dim, tag, "I8");
      if (tagbase->type() == OMEGA_H_I32)
        printTagInfo<Omega_h::I32>(mesh, oss, dim, tag, "I32");
      if (tagbase->type() == OMEGA_H_I64)
        printTagInfo<Omega_h::I64>(mesh, oss, dim, tag, "I64");
      if (tagbase->type() == OMEGA_H_F64)
        printTagInfo<Omega_h::Real>(mesh, oss, dim, tag, "F64");
    }
  }

  std::cout << oss.str();
}
}



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
      if(! m_mesh->has_tag(OMEGA_H_VERT,name)) {
        fprintf(stderr, "%s add_tag %s\n", __func__, name.c_str());
        m_mesh->add_tag<ST>(OMEGA_H_VERT,name,numComps,f,false);
      } else {
        fprintf(stderr, "%s set_tag %s\n", __func__, name.c_str());
        m_mesh->set_tag<ST>(OMEGA_H_VERT,name,f,false);
      }
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
  fprintf(stderr, "field_name %s\n", field_name.c_str());
  for(int i=0; i<mesh_data_h.size(); i++) {
    fprintf(stderr, "mesh_data_h[%d] %f\n", i, mesh_data_h[i]);
  }
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
  fprintf(stderr, "OmegahMeshFieldAccessor::saveVector %s\n", field_name.c_str());
  fprintf(stderr, "OmegahMeshFieldAccessor::saveVector meshPtr %p\n", m_mesh.get());
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

  //debug::printAllTags(*m_mesh);

  if(! m_mesh->has_tag(dim,field_name)) {
    m_mesh->add_tag(dim,field_name, ncomps, read(mesh_data_h.write()));
  } else {
    m_mesh->set_tag(dim,field_name,read(mesh_data_h.write()),false);
  }

  {
    auto foo  = hostRead(m_mesh->get_array<Omega_h::Real>(0,field_name));
    fprintf(stderr, "saveVector field_name %s\n", field_name.c_str());
    for(int i=0; i<foo.size(); i++) {
      fprintf(stderr, "tag[%d] %.13f\n", i, foo[i]);
    }
  }

}

} // namespace Albany
