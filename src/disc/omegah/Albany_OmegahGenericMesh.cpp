#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_OmegahUtils.hpp"

#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Gather.hpp"

#include <regex>

namespace Albany
{

OmegahGenericMesh::
OmegahGenericMesh(const Teuchos::RCP<Teuchos::ParameterList>& params)
 : m_params(params)
{
  // Nothing to do here
}

void OmegahGenericMesh::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& /* comm */,
              const Teuchos::RCP<StateInfoStruct>& sis)
{
  m_field_accessor = Teuchos::rcp(new OmegahMeshFieldAccessor(m_mesh));
  if (not sis.is_null()) {
    m_field_accessor->addStateStructs (sis);
  }
}

LO OmegahGenericMesh::get_num_local_nodes () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::runtime_error,
      "Error! Cannot query number of local nodes until bulk data is set.\n");

  return m_mesh->nverts();
}

LO OmegahGenericMesh::get_num_local_elements () const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::runtime_error,
      "Error! Cannot query number of local elements until bulk data is set.\n");

  return m_mesh->nelems();
}

GO OmegahGenericMesh::get_max_node_gid () const
{
  if (m_max_node_gid==-1) {
    auto globals_d = m_mesh->globals(0);
    Omega_h::HostRead<Omega_h::GO> global_h(globals_d);
    for (int i=0; i<global_h.size(); ++i) {
      m_max_node_gid = std::max(m_max_node_gid,GO(global_h[i]));
    }

    auto comm = m_mesh->comm();
    m_max_node_gid = comm->allreduce(static_cast<std::int64_t>(m_max_node_gid),OMEGA_H_MAX);
  }
  return m_max_node_gid;
}

GO OmegahGenericMesh::get_max_elem_gid () const
{
  if (m_max_elem_gid==-1) {
    auto globals_d = m_mesh->globals(m_mesh->dim());
    Omega_h::HostRead<Omega_h::GO> global_h(globals_d);
    for (int i=0; i<global_h.size(); ++i) {
      m_max_elem_gid = std::max(m_max_elem_gid,GO(global_h[i]));
    }

    auto comm = m_mesh->comm();
    m_max_elem_gid = comm->allreduce(static_cast<std::int64_t>(m_max_elem_gid),OMEGA_H_MAX);
  }
  return m_max_elem_gid;
}

int OmegahGenericMesh::
part_dim (const std::string& part_name) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_part_topo.find(part_name)==m_part_topo.end(), std::runtime_error,
      "[OmegahGenericMesh::part_dim] Error! Cannot find input part: " << part_name << "\n");
  return topo_dim(m_part_topo.at(part_name));
}

void OmegahGenericMesh::
declare_part (const std::string& name, const Topo_type topo)
{
  TEUCHOS_TEST_FOR_EXCEPTION (
      m_part_topo.find(name)!=m_part_topo.end() and m_part_topo[name]!=topo, std::logic_error,
      "[OmegahGenericMesh::declare_part] Error! Redefining part topology to a different value.\n"
      "  - part name: " << name << "\n"
      "  - curr topo: " << e2str(m_part_topo[name]) << "\n"
      "  - new  topo: " << e2str(topo) << "\n");

  // Check that this topo matches the topo of one of the element sub-entities topos
  TEUCHOS_TEST_FOR_EXCEPTION (not m_mesh->has_ents(topo_dim(topo)), std::logic_error,
      "[OmegahGenericMesh::declare_part] Mesh does not store any entity of the input topology.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << topo_dim(topo) << "\n"
      "  - part topo: " << e2str(topo) << "\n"
      "  - mesh dim : " << m_mesh->dim() << "\n"
      "  - mesh type: " << e2str(m_mesh->family()) << "\n");

  // All good, store it
  m_part_topo[name] = topo;
}

void OmegahGenericMesh::
declare_part (const std::string& name, const Topo_type topo,
              Omega_h::Read<Omega_h::I8> is_entity_in_part,
              const bool markDownward)
{
  declare_part (name,topo);
  mark_part_entities (name,is_entity_in_part,markDownward);
}

void OmegahGenericMesh::
mark_part_entities (const std::string& name,
                   Omega_h::Read<Omega_h::I8> is_entity_in_part,
                   const bool markDownward)
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_part_topo.find(name)==m_part_topo.end(), std::runtime_error,
      "[OmegahGenericMesh::set_part_entities] Error! Part not found.\n"
      "  - part name: " << name << "\n");

  auto dim = topo_dim(m_part_topo[name]);

  TEUCHOS_TEST_FOR_EXCEPTION (is_entity_in_part.size()!=m_mesh->nents(dim), std::logic_error,
      "[OmegahGenericMesh::set_part_entities] Error! Input array has the wrong dimensions.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << dim << "\n"
      "  - num ents : " << m_mesh->nents(dim) << "\n"
      "  - array dim: " << is_entity_in_part.size() << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (m_mesh->has_tag(dim,name), std::runtime_error,
      "[OmegahGenericMesh::set_part_entities] Error! A tag with this name was already set.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << dim << "\n");

  m_mesh->add_tag(dim,name,1,is_entity_in_part);

  if (markDownward) {
    TEUCHOS_TEST_FOR_EXCEPTION (dim==0, std::logic_error,
      "[OmegahGenericMesh::set_part_entities] Error! Cannot mark downward if the part dimension is 0.\n"
      "  - part name: " << name << "\n"
      "  - part dim : " << dim << "\n")
    Omega_h::Write<Omega_h::I8> downMarked;

    // NOTE: In the following, we keep converting Topo_type to its integer dim. That's because
    //       Omega_h::Mesh seems to store a valid nent(dim) count, but uninited nent(topo) count.
    //       If topo is 2d (or less), we can safely use nent(dim), since there is only one possible
    //       down_topo (edge or vertex) for each topo. But if this method is called to mark a 3d
    //       region, then down ents may be of multiple topos (e.g., tria and quad for a wedge).
    //       Until then, turn topo into dim
    auto upMarked = is_entity_in_part;
    auto topo = m_part_topo[name];
    while (topo!=Topo_type::vertex) {
      const auto down_topo = get_side_topo(topo);

      Omega_h::Write<Omega_h::I8> downMarked (m_mesh->nents(topo_dim(down_topo)),0);

      const int deg = Omega_h::element_degree(topo,down_topo);

      auto adj = m_mesh->ask_down(topo_dim(topo),topo_dim(down_topo));
      auto f = OMEGA_H_LAMBDA (LO i) {
        if (upMarked[i]) {
          for (int j=0; j<deg; ++j) {
            // Note: this has a race condition, but it doesn't matter,
            //       since all threads would write 1
            downMarked[adj.ab2b[i*deg + j]] = 1;
          }
        }
      };
      Kokkos::parallel_for ("OmegahGenericMesh::set_part_entities::markDownward",m_mesh->nents(topo_dim(topo)),f);
      m_mesh->add_tag(topo_dim(down_topo),name,1,read(downMarked));
      upMarked = downMarked;

      topo = down_topo;
    }
  }
}

void OmegahGenericMesh::
loadRequiredInputFields (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  auto out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  out->setProcRankAndSize(comm->getRank(), comm->getSize());
  out->setOutputToRootOnly(0);

  *out << "[OmegahGenericMesh] Processing field requirements...\n";

  // Get nodes/elems global ids
  auto node_gids = m_mesh->globals(0);
  auto elem_gids = m_mesh->globals(m_mesh->dim());
  auto node_gids_h = hostRead(node_gids);
  auto elem_gids_h = hostRead(elem_gids);

  // NOTE: the reinterpret_cast is safe, since both Albany and Omegah use 64bit int for Global ids
  Teuchos::ArrayView<const GO> node_gids_av(reinterpret_cast<const GO*>(node_gids_h.data()),node_gids_h.size());
  Teuchos::ArrayView<const GO> elem_gids_av(reinterpret_cast<const GO*>(elem_gids_h.data()),elem_gids_h.size());

  auto nodes_vs = createVectorSpace(comm,node_gids_av);
  auto elems_vs = createVectorSpace(comm,elem_gids_av);

  // Check whether we need the serial map or not. The only scenario where we DO need it is if we are
  // loading a field from an ASCII file. So let's check the fields info to see if that's the case.
  auto& req_fields_info = m_params->sublist("Required Fields Info");
  int num_fields = req_fields_info.get<int>("Number Of Fields",0);
  bool node_field_ascii_loads = false;
  bool elem_field_ascii_loads = false;
  std::string fname, fusage, ftype, forigin;
  for (int ifield=0; ifield<num_fields; ++ifield) {
    std::stringstream ss;
    ss << "Field " << ifield;
    auto& fparams = req_fields_info.sublist(ss.str());

    fusage = fparams.get<std::string>("Field Usage", "Input");
    ftype  = fparams.get<std::string>("Field Type","INVALID");
    if (fusage == "Input" || fusage == "Input-Output") {
      forigin = fparams.get<std::string>("Field Origin","INVALID");
      if (forigin=="File" && fparams.isParameter("File Name")) {
        if (ftype.find("Node")!=std::string::npos) {
          node_field_ascii_loads = true;
        } else if (ftype.find("Elem")!=std::string::npos) {
          elem_field_ascii_loads = true;
        }
      }
    }
  }

  // NOTE: the serial vs cannot be created linearly, with GIDs from 0 to numGlobalNodes/Elems, since
  //       this may be a boundary mesh, and the GIDs may not start from 0, nor be contiguous.
  //       Therefore, we must create a root vs. Moreover, we need the GIDs sorted (so that, regardless
  //       of the GID, we read the serial input files in the correct order), and we can't sort them
  //       once the vs is created.

  auto serial_nodes_vs = nodes_vs;
  auto serial_elems_vs = elems_vs;
  if (node_field_ascii_loads) {
    Teuchos::Array<GO> all_node_gids;
    gatherV(comm,node_gids_av,all_node_gids,0);
    std::sort(all_node_gids.begin(),all_node_gids.end());
    auto it = std::unique(all_node_gids.begin(),all_node_gids.end());
    all_node_gids.erase(it,all_node_gids.end());
    serial_nodes_vs = createVectorSpace(comm,all_node_gids);
  }
  if (elem_field_ascii_loads) {
    Teuchos::Array<GO> all_elem_gids;
    gatherV(comm,elem_gids_av,all_elem_gids,0);
    std::sort(all_elem_gids.begin(),all_elem_gids.end());
    serial_elems_vs = createVectorSpace(comm,all_elem_gids);
  }

  // Creating the combine and scatter manager object (to transfer from serial to parallel vectors)
  auto cas_manager_node = createCombineAndScatterManager(serial_nodes_vs,nodes_vs);
  auto cas_manager_elem = createCombineAndScatterManager(serial_elems_vs,elems_vs);

  std::string valid_ftype_r = "(Node|Elem) (Layered) (Scalar|Vector)|(Node|Elem) (Scalar|Vector)";
  std::string valid_fusage_r = R"(Input(-Output)?|Output|Unused)";

  std::regex nodal_r("Node");
  std::regex scalar_r("Scalar");
  std::regex layered_r("Layered");
  std::regex input_r("Input");
  for (int ifield=0; ifield<num_fields; ++ifield) {
    std::stringstream ss;
    ss << "Field " << ifield;
    Teuchos::ParameterList& fparams = req_fields_info.sublist(ss.str());

    // First, get the name and usage of the field, and check if it's used
    if (fparams.isParameter("State Name")) {
      fname = fparams.get<std::string>("State Name");
    } else {
      fname = fparams.get<std::string>("Field Name");
    }

    fusage = fparams.get<std::string>("Field Usage", "Input");
    TEUCHOS_TEST_FOR_EXCEPTION (not std::regex_search(fusage,std::regex(valid_fusage_r)),
        Teuchos::Exceptions::InvalidParameterValue,
        "Error! Invalid field usage format.\n"
        " - field name: " << fname << "\n"
        " - field usage: " << fusage << "\n"
        " - valid usage regex: " << valid_fusage_r << "\n");

    if (fusage == "Unused") {
      *out << "  - Skipping field '" << fname << "' since it's listed as unused.\n";
      continue;
    }

    // The field is used somehow. Check that it is present in the mesh
    ftype = fparams.get<std::string>("Field Type","INVALID");

    TEUCHOS_TEST_FOR_EXCEPTION (not std::regex_search(ftype,std::regex(valid_ftype_r)),
        Teuchos::Exceptions::InvalidParameterValue,
        "Error! Invalid field type format.\n"
        " - field name: " << fname << "\n"
        " - field type: " << ftype << "\n"
        " - valid type regex: " << valid_ftype_r << "\n");

    // Check if it's an output file (nothing to be done then). If not, check that the usage is a valid string
    if (fusage == "Output") {
      *out << "  - Skipping field '" << fname << "' since it's listed as output. Make sure there's an evaluator set to save it!\n";
      continue;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (fusage!="Input" && fusage!="Input-Output",
          Teuchos::Exceptions::InvalidParameter,
          "Error! 'Field Usage' for field '" << fname << "' must be one of 'Input', 'Output', 'Input-Output' or 'Unused'.\n");
    }

    // Ok, it's an input (or input-output) field. Find out where the field comes from
    forigin = fparams.get<std::string>("Field Origin","INVALID");
    if (forigin=="Mesh") {
      *out << "  - Skipping field '" << fname << "' since it's listed as present in the mesh.\n";
      continue;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (forigin!="File",
          Teuchos::Exceptions::InvalidParameter,
          "Error! 'Field Origin' for field '" << fname << "' must be one of 'File' or 'Mesh'.\n");
    }

    // The field is not already present (with updated values) in the mesh, and must be loaded/computed filled here.

    // Detect load type
    bool load_ascii = fparams.isParameter("File Name");
    bool load_math_expr = fparams.isParameter("Field Expression");
    bool load_value = fparams.isParameter("Field Value") || fparams.isParameter("Random Value");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_ascii && load_value, std::logic_error,
        "Error! You cannot specify both 'File Name' and 'Field Value' (or 'Random Value') for loading a field.\n");
    TEUCHOS_TEST_FOR_EXCEPTION ( load_math_expr, std::logic_error,
        "Error! 'Field Expression' not supported by Omegah meshes (yet).\n");

    // Depending on the input field type, we need to use different pointers/importers/vectors
    bool nodal = std::regex_search(ftype,nodal_r);
    bool scalar = std::regex_search(ftype,scalar_r);
    bool layered = std::regex_search(ftype,layered_r);

    auto cas_manager = nodal ? cas_manager_node : cas_manager_elem;
    auto serial_vs = cas_manager->getOwnedVectorSpace();
    auto vs = cas_manager->getOverlappedVectorSpace();  // It is not overlapped, it is just distributed.

    std::vector<double> norm_layers_coords;
    if (layered) {
      norm_layers_coords = m_field_accessor->getMeshVectorStates()[fname + "_NLC"];
    }
    Teuchos::RCP<Thyra_MultiVector> field_mv;
    if (load_ascii) {
      field_mv = loadField (fname, fparams, *cas_manager, comm, nodal, scalar, layered, out, norm_layers_coords);
    } else if (load_value) {
      field_mv = fillField (fname, fparams, vs, nodal, scalar, layered, out, norm_layers_coords);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
          "Error! No means were specified for loading field '" + fname + "'.\n");
    }

  //   auto field_mv_view = getLocalData(field_mv.getConst());

  //   //Now we have to stuff the vector in the mesh data
  //   using SFT = AbstractSTKFieldContainer::STKFieldType;
  //   stk::topology::rank_t entity_rank = nodal ? stk::topology::NODE_RANK : stk::topology::ELEM_RANK;
  //   SFT* stk_field = metaData->get_field<double> (entity_rank, fname);
  //   TEUCHOS_TEST_FOR_EXCEPTION (stk_field==nullptr, std::logic_error,
  //       "Error! Field " << fname << " not present (perhaps is not '" << ftype << "'?).\n");

  //   stk::mesh::EntityId gid;
  //   LO lid;
  //   auto indexer = createGlobalLocalIndexer(vs);
  //   for (unsigned int i(0); i<entities->size(); ++i) {
  //     double* values = stk::mesh::field_data(*stk_field, (*entities)[i]);

  //     gid = bulkData->identifier((*entities)[i]) - 1;
  //     lid = indexer->getLocalElement(GO(gid));
  //     for (int iDim(0); iDim<field_mv_view.size(); ++iDim) {
  //       values[iDim] = field_mv_view[iDim][lid];
  //     }
  //   }
  }
}

} // namespace Albany
