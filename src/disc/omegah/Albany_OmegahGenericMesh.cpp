#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_OmegahUtils.hpp"
#include "Albany_Omegah.hpp"

#include <Omega_h_build.hpp>  // for Omega_h::build_box
#include <Omega_h_file.hpp>   // for Omega_h::binary::read
#include <Omega_h_mark.hpp>   // for Omega_h::mark_by_class

namespace Albany
{

OmegahGenericMesh::
OmegahGenericMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
                   const Teuchos::RCP<const Teuchos_Comm>& /* comm */,
                   const int /* num_params */)
{
  const auto& method = params->get<std::string>("Method");
  if (method=="OshFile") {
    loadOmegahMesh(params);
  } else if (method=="Box1D" or method=="Box2D" or method=="Box3D") {
    // Digits have CONSECUTIVE char values. So '1' = '0'+1, etc.
    int dim = method[3] - '0';
    buildBox(params,dim);
  }
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

void OmegahGenericMesh::
setBulkData (const Teuchos::RCP<const Teuchos_Comm>& /* comm */)
{
  m_bulk_data_set = true;
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
loadOmegahMesh (const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  auto& lib = get_omegah_lib();
  m_mesh = Teuchos::RCP<Omega_h::Mesh>(new Omega_h::Mesh(&lib));

  const auto& filename = params->get<std::string>("Filename");
  Omega_h::binary::read(filename, lib.world(), m_mesh.get());
  if (params->get("Rebalance",true)) {
    m_mesh->balance(); // re-partition to the number of ranks in world communicator
  }

  m_coords_d = m_mesh->coords().view();
  m_coords_h = Kokkos::create_mirror_view(m_coords_d);
  Kokkos::deep_copy(m_coords_h,m_coords_d);

  // Node/Side sets names
  std::vector<std::string> nsNames, ssNames;

  // The omegah 'exo2osh' converter creates geometric model entities from node
  // and side sets that exist within the exodus file.
  // The mesh entities in the sets are then 'classified' (sets the association)
  // on those model entities.
  // 'Classification' of mesh entities to the geometric model is an alternative
  // to the generic creation of 'parts' (sets of mesh entities with a label) and
  // provides a subset of the functionality.
  // Note, 'classification' is the approach taken when having (at a minimum) a
  // topological definition of the domain is a common part of the mesh
  // generation/adaptation workflow.
  // A dimension and id uniquely defines a geometric model entity.
  const auto& parts_names = params->get<Teuchos::Array<std::string>>("Mark Parts",{});
  for (const auto& pn : parts_names) {
          auto& pl = params->sublist(pn);
    const auto topo_str = pl.get<std::string>("Topo");
    const auto topo = str2topo(pl.get<std::string>("Topo"));
    const int dim = topo_dim(topo);
    const int id  = pl.get<int>("Id");
    const bool markDownward = pl.get<int>("Mark Downward",true); // Is default=true ok?
    auto is_in_part = Omega_h::mark_by_class(m_mesh.get(), dim, dim, id);
    this->declare_part(pn,topo,is_in_part,markDownward);

    if (dim==0) {
      nsNames.push_back(pn);
    } else if (dim==m_mesh->dim()-1) {
      ssNames.push_back(pn);
    }
  }

  const CellTopologyData* ctd;

  TEUCHOS_TEST_FOR_EXCEPTION (
      m_mesh->family()!=Omega_h_Family::OMEGA_H_SIMPLEX and m_mesh->family()!=Omega_h_Family::OMEGA_H_HYPERCUBE,
      std::runtime_error,
      "Error! OmegahOshMesh only available for simplex/hypercube meshes.\n");

  Topo_type elem_topo;
  switch (m_mesh->dim()) {
    case 1:
      ctd = shards::getCellTopologyData<shards::Line<2>>();
      elem_topo = Topo_type::edge;
      break;
    case 2:
      if (m_mesh->family()==Omega_h_Family::OMEGA_H_SIMPLEX) {
        ctd = shards::getCellTopologyData<shards::Triangle<3>>();
        elem_topo = Topo_type::triangle;
      } else {
        ctd = shards::getCellTopologyData<shards::Quadrilateral<4>>();
        elem_topo = Topo_type::quadrilateral;
      }
      break;
    case 3:
      if (m_mesh->family()==Omega_h_Family::OMEGA_H_SIMPLEX) {
        ctd = shards::getCellTopologyData<shards::Tetrahedron<4>>();
        elem_topo = Topo_type::tetrahedron;
      } else {
        ctd = shards::getCellTopologyData<shards::Hexahedron<8>>();
        elem_topo = Topo_type::hexahedron;
      }
      break;
  }
  std::string ebName = "element_block_0";
  std::map<std::string,int> ebNameToIndex =
  {
    { ebName, 0}
  };
  this->declare_part(ebName,elem_topo);

  // Omega_h does not know what worksets are, so all elements are in one workset
  this->meshSpecs.resize(1);
  int ws_size_max = params->get<int>("Workset Size", -1);
  int ws_size = computeWorksetSize(ws_size_max,m_mesh->nelems());
  this->meshSpecs[0] = Teuchos::rcp(
      new MeshSpecsStruct(MeshType::Unstructured, *ctd, m_mesh->dim(),
                          nsNames, ssNames, ws_size, ebName,
                          ebNameToIndex));
}


OmegahGenericMesh::GeomMdlToSets OmegahGenericMesh::setGeomModelToNodeSets(int dim) const {
  GeomMdlToSets gm2ns;
  if( dim==1 ) {
    const int vtxDim = 0;
    gm2ns.insert({"NodeSet0", {vtxDim,0}});
    gm2ns.insert({"NodeSet1", {vtxDim,2}});
    //vertices that are not at the endpoints of the line have id=1
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Construction of the map from geometric model ids to node/side sets is only supported for 1d domains.\n");
  }
  return gm2ns;
}

OmegahGenericMesh::GeomMdlToSets OmegahGenericMesh::setGeomModelToSideSets(int dim) const {
  GeomMdlToSets gm2ss;
  if( dim==1 ) {
    const int vtxDim = 0;
    gm2ss.insert({"SideSet0", {vtxDim,0}});
    gm2ss.insert({"SideSet1", {vtxDim,2}});
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Construction of the map from geometric model ids to node/side sets is only supported for 1d domains.\n");
  }
  return gm2ss;
}

std::vector<std::string>
OmegahGenericMesh::createNodeSets() {
  std::vector<std::string> nsNames;
  for( auto& [name, ent] : geomMdlToNodeSets ) {
    const auto geomMdlEntDim = std::get<0>(ent);
    const auto geomMdlEntId = std::get<1>(ent);
    fprintf(stderr, "name: %s dim: %d id: %d\n",
        name.c_str(), geomMdlEntDim, geomMdlEntId);
    nsNames.push_back(name);
    auto tag = Omega_h::mark_by_class(m_mesh.get(),0,geomMdlEntDim,geomMdlEntId);
    this->declare_part(name,Topo_type::vertex,tag,false);
  }
  return nsNames;
}

std::vector<std::string>
OmegahGenericMesh::createSideSets() {
  std::vector<std::string> ssNames;
  for( auto& [name, ent] : geomMdlToSideSets ) {
    const auto geomMdlEntDim = std::get<0>(ent);
    const auto geomMdlEntId = std::get<1>(ent);
    fprintf(stderr, "name: %s dim: %d id: %d\n",
        name.c_str(), geomMdlEntDim, geomMdlEntId);
    ssNames.push_back(name);
    if(getOmegahMesh()->dim()==1) {
      auto tag = Omega_h::mark_by_class(m_mesh.get(),0,geomMdlEntDim,geomMdlEntId);
      this->declare_part(name,Topo_type::vertex,tag,false);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
          "Omega_h BuildBox: Only 1d meshes supported.\n");
    }
  }
  return ssNames;
}

void OmegahGenericMesh::
buildBox (const Teuchos::RCP<Teuchos::ParameterList>& params, const int dim)
{
  using I8 = Omega_h::I8;

  auto nelems = params->get<Teuchos::Array<int>>("Number of Elements");
  auto scale  = params->get<Teuchos::Array<double>>("Box Scaling",Teuchos::Array<double>(dim,1.0));
  TEUCHOS_TEST_FOR_EXCEPTION (nelems.size()!=dim, std::logic_error,
      "Input array for 'Number of Elements' has the wrong dimension.\n"
      "  - Expected length: " << dim << "\n"
      "  - Input length   : " << nelems.size() << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (scale.size()!=dim, std::logic_error,
      "Input array for 'Box Scaling' has the wrong dimension.\n"
      "  - Expected length: " << dim << "\n"
      "  - Input length   : " << scale.size() << "\n");

  std::string topo_str = "Simplex";
  if (params->isParameter("Topology Type")) {
    topo_str = params->get<std::string>("Topology Type");
  }

  TEUCHOS_TEST_FOR_EXCEPTION (topo_str!="Simplex" and topo_str!="Hypercube", std::runtime_error,
      "Error! Invalid topology type '" << topo_str << "'\n"
      "   Valid choices: Simplex, Hypercube\n");

  TEUCHOS_TEST_FOR_EXCEPTION (topo_str=="Hypercube", std::runtime_error,
      "Error! Hypercube box meshes not yet supported.\n");

  int nelemx = 0;
  int nelemy = 0;
  int nelemz = 0;

  double scalex = 1.0;
  double scaley = 1.0;
  double scalez = 1.0;

  nelemx = nelems[0];
  scalex = scale[0];
  if (dim>1) {
    nelemy = nelems[1];
    scaley = scale[1];
    if (dim>2) {
      nelemz = nelems[2];
      scalez = scale[2];
    }
  }

  // Create the omegah mesh obj
  Topo_type elem_topo;
  if (topo_str=="Simplex") {
    m_mesh = Teuchos::rcp(new Omega_h::Mesh(Omega_h::build_box(get_omegah_lib().world(),OMEGA_H_SIMPLEX,
                                            scalex,scaley,scalez,nelemx,nelemy,nelemz)));
    elem_topo = dim==3 ? Topo_type::tetrahedron
                       : dim==2 ? Topo_type::triangle : Topo_type::edge;
  } else {
    m_mesh = Teuchos::rcp(new Omega_h::Mesh(Omega_h::build_box(get_omegah_lib().world(),OMEGA_H_HYPERCUBE,
                                            scalex,scaley,scalez,nelemx,nelemy,nelemz)));

    elem_topo = dim==3 ? Topo_type::hexahedron
                       : dim==2 ? Topo_type::quadrilateral : Topo_type::edge;
  }

  m_mesh->set_parting(OMEGA_H_ELEM_BASED);
  m_coords_d = m_mesh->coords().view();
  m_coords_h = Kokkos::create_mirror_view(m_coords_d);
  Kokkos::deep_copy(m_coords_h,m_coords_d);

  // Create the mesh specs
  std::string ebName = "element_block_0";
  std::map<std::string,int> ebNameToIndex =
  {
    { ebName, 0}
  };
  this->declare_part(ebName,elem_topo);

  geomMdlToNodeSets = setGeomModelToNodeSets(dim);
  geomMdlToSideSets = setGeomModelToSideSets(dim);
  auto nsNames = createNodeSets();
  auto ssNames = createSideSets();

  // Omega_h does not know what worksets are, so all elements are in one workset
  const CellTopologyData* ctd;

  switch (dim) {
    case 1: ctd = shards::getCellTopologyData<shards::Line<2>>();         break;
    case 2: ctd = shards::getCellTopologyData<shards::Triangle<3>>();     break;
    case 3: ctd = shards::getCellTopologyData<shards::Tetrahedron<4>>();  break;
  }

  this->meshSpecs.resize(1);
  int ws_size_max = params->get<int>("Workset Size", -1);
  int ws_size = computeWorksetSize(ws_size_max,m_mesh->nelems());
  this->meshSpecs[0] = Teuchos::rcp(
      new MeshSpecsStruct(MeshType::Structured, *ctd, dim,
                          nsNames, ssNames, ws_size, ebName,
                          ebNameToIndex));
}

} // namespace Albany
