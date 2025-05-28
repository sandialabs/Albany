#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_OmegahUtils.hpp"
#include "Albany_Omegah.hpp"

#include <Omega_h_for.hpp>        // for Omega_h::parallel_for
#include <Omega_h_build.hpp>      // for Omega_h::build_box
#include <Omega_h_file.hpp>       // for Omega_h::binary::read
#include <Omega_h_mark.hpp>       // for Omega_h::mark_by_class

#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Gather.hpp"

#include <regex>

// Uncomment this to print some debug output
// #define DEBUG_OUTPUT

namespace Albany
{

OmegahGenericMesh::
OmegahGenericMesh (const Teuchos::RCP<Teuchos::ParameterList>& params)
 : m_params(params)
{
  const auto& method = m_params->get<std::string>("Mesh Creation Method");
  if (method=="OshFile") {
    loadOmegahMesh();
  } else if (method=="Box1D" or method=="Box2D" or method=="Box3D") {
    // Digits have CONSECUTIVE char values. So '1' = '0'+1, etc.
    int dim = method[3] - '0';
    buildBox(dim);
  }
}

void OmegahGenericMesh::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
              const Teuchos::RCP<StateInfoStruct>& sis)
{
  m_field_accessor = Teuchos::rcp(new OmegahMeshFieldAccessor(m_mesh));
  if (not sis.is_null()) {
    m_field_accessor->addStateStructs (sis);
  }

  loadRequiredInputFields(comm);
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
loadOmegahMesh ()
{
  auto& lib = get_omegah_lib();
  m_mesh = Teuchos::RCP<Omega_h::Mesh>(new Omega_h::Mesh(&lib));

  const auto& filename = m_params->get<std::string>("Input Filename");
  Omega_h::binary::read(filename, lib.world(), m_mesh.get());
  if (m_params->get("Rebalance",true)) {
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
  const auto& parts_names = m_params->get<Teuchos::Array<std::string>>("Mark Parts",{});
  for (const auto& pn : parts_names) {
          auto& pl = m_params->sublist(pn);
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
  int ws_size_max = m_params->get<int>("Workset Size", -1);
  int ws_size = computeWorksetSize(ws_size_max,m_mesh->nelems());
  this->meshSpecs[0] = Teuchos::rcp(
      new MeshSpecsStruct(MeshType::Unstructured, *ctd, m_mesh->dim(),
                          nsNames, ssNames, ws_size, ebName,
                          ebNameToIndex));
}

OmegahGenericMesh::PartToGeoModelEntities
OmegahGenericMesh::setNodeSetsGeoModelEntities() const
{
  // NOTE: this routine is explicitly tailored for BOX meshes only.

  int dim = m_mesh->dim();
  PartToGeoModelEntities ns_gm_entitites;
  const int nodeDim = 0;
  const int edgeDim = 1;
  const int faceDim = 2;
  if (dim==1) {
    auto& ns0 = ns_gm_entitites["NodeSet0"];
    auto& ns1 = ns_gm_entitites["NodeSet1"];

    // Class ids are
    //  x=0 -> 0
    //  x=1 -> 2
    //vertices that are not at the endpoints of the line have class id=1
    ns0.push_back({nodeDim,0});
    ns1.push_back({nodeDim,2});
  } else if (dim==2) {
    auto& ns0 = ns_gm_entitites["NodeSet0"];
    auto& ns1 = ns_gm_entitites["NodeSet1"];
    auto& ns2 = ns_gm_entitites["NodeSet2"];
    auto& ns3 = ns_gm_entitites["NodeSet3"];
    // Class ids are
    //  x=0 -> 0, 3, and 6 (0,6 are corners, 3 is the geo model edge)
    //  x=1 -> 2, 5, and 8 (2,8 are corners, 5 is the geo model edge)
    //  y=0 -> 0, 1, and 2 (0,2 are corners, 1 is the geo model edge)
    //  y=1 -> 6, 7, and 8 (6,8 are corners, 7 is the geo model edge)
    ns0.push_back({nodeDim,0});
    ns0.push_back({edgeDim,3});
    ns0.push_back({nodeDim,6});

    ns1.push_back({nodeDim,2});
    ns1.push_back({edgeDim,5});
    ns1.push_back({nodeDim,8});

    ns2.push_back({nodeDim,0});
    ns2.push_back({edgeDim,1});
    ns2.push_back({nodeDim,2});

    ns3.push_back({nodeDim,6});
    ns3.push_back({edgeDim,7});
    ns3.push_back({nodeDim,8});
  } else {
    // TODO: implement dim=3
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Construction of the map from geometric model ids to node/side sets is only supported for 1d domains.\n");
  }
  return ns_gm_entitites;
}

OmegahGenericMesh::PartToGeoModelEntities
OmegahGenericMesh::setSideSetsGeoModelEntities() const
{
  // NOTE: this routine is explicitly tailored for BOX meshes only.

  int dim = m_mesh->dim();
  PartToGeoModelEntities ss_gm_entitites;
  // Class ids follow the same numbering as explained in the ns equivalent fcn,
  // with the difference that we now only care about geom model sides (vertices in 1d,
  // edges in 2d, faces in 3d)
  const int nodeDim = 0;
  const int edgeDim = 1;
  const int faceDim = 2;
  if (dim==1) {
    auto& ss0 = ss_gm_entitites["SideSet0"];
    auto& ss1 = ss_gm_entitites["SideSet1"];

    ss0.push_back({nodeDim,0});
    ss1.push_back({nodeDim,2});
  } else if (dim==2) {
    auto& ss0 = ss_gm_entitites["SideSet0"];
    auto& ss1 = ss_gm_entitites["SideSet1"];
    auto& ss2 = ss_gm_entitites["SideSet2"];
    auto& ss3 = ss_gm_entitites["SideSet3"];

    ss0.push_back({edgeDim,3});
    ss1.push_back({edgeDim,5});
    ss2.push_back({edgeDim,1});
    ss3.push_back({edgeDim,7});
  } else {
    // TODO: implement dim=3
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Construction of the map from geometric model ids to node/side sets is only supported for 1d domains.\n");
  }
  return ss_gm_entitites;
}

std::vector<std::string>
OmegahGenericMesh::createNodeSets()
{
  std::vector<std::string> nsNames;
  for( auto& [name, gm_ents] : nodeSetsGeoModelEntities ) {
    nsNames.push_back(name);

    Omega_h::Write<Omega_h::I8> tag (m_mesh->nents(0),0);

    for (auto ent : gm_ents) {
      auto id_tag = Omega_h::mark_by_class(m_mesh.get(),0,ent.dim,ent.id);

      auto f = OMEGA_H_LAMBDA(LO i) { tag[i] = (tag[i] or id_tag[i]); };
      Omega_h::parallel_for(tag.size(),f);
    }
#ifdef DEBUG_OUTPUT
    std::cout << "ns " << name << " tag:";
    for (int i=0; i<tag.size(); ++i) { std::cout << " " << static_cast<int>(tag[i]); } std::cout << "\n";
#endif
    this->declare_part(name,Topo_type::vertex,tag,false);
  }
  return nsNames;
}

std::vector<std::string>
OmegahGenericMesh::createSideSets()
{
  std::vector<std::string> ssNames;
  int sideDim = m_mesh->dim()-1;
  Topo_type elem_topo = m_part_topo.at("element_block_0");
  Topo_type side_topo;
  switch (sideDim) {
    case 0: side_topo = Topo_type::vertex; break;
    case 1: side_topo = Topo_type::edge; break;
    case 2:
      switch (elem_topo) {
        case Topo_type::hexahedron: side_topo = Topo_type::quadrilateral;
        case Topo_type::tetrahedron: side_topo = Topo_type::triangle;
        default:
          // TODO: should we handle wedges? Maybe not, since so fare this method is only called for box mesh?
          throw std::runtime_error("Error! Side topology not yet implemented for box mesh.\n");
      } break;
    default:
      throw std::runtime_error("Error! Unsupported side dimension (" + std::to_string(sideDim) + ").\n");
  }

  for( auto& [name, gm_ents] : sideSetsGeoModelEntities ) {
    ssNames.push_back(name);

    Omega_h::Write<Omega_h::I8> tag (m_mesh->nents(sideDim),0);

    for (auto ent : gm_ents) {
      auto id_tag = Omega_h::mark_by_class(m_mesh.get(),sideDim,ent.dim,ent.id);

      auto f = OMEGA_H_LAMBDA(LO i) { tag[i] = (tag[i] or id_tag[i]); };
      Omega_h::parallel_for(tag.size(),f);
    }
#ifdef DEBUG_OUTPUT
    std::cout << "ss " << name << " tag:";
    for (int i=0; i<tag.size(); ++i) { std::cout << " " << static_cast<int>(tag[i]); } std::cout << "\n";
#endif
    this->declare_part(name,side_topo,tag,false);
  }
  return ssNames;
}

void OmegahGenericMesh::setCoordinates() {
  m_coords_d = m_mesh->coords().view();
  m_coords_h = Kokkos::create_mirror_view(m_coords_d);
  Kokkos::deep_copy(m_coords_h,m_coords_d);
}

void OmegahGenericMesh::
buildBox (const int dim)
{
  using I8 = Omega_h::I8;

  auto nelems = m_params->get<Teuchos::Array<int>>("Number of Elements");
  auto scale  = m_params->get<Teuchos::Array<double>>("Box Scaling",Teuchos::Array<double>(dim,1.0));
  TEUCHOS_TEST_FOR_EXCEPTION (nelems.size()!=dim, std::logic_error,
      "Input array for 'Number of Elements' has the wrong dimension.\n"
      "  - Expected length: " << dim << "\n"
      "  - Input length   : " << nelems.size() << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (scale.size()!=dim, std::logic_error,
      "Input array for 'Box Scaling' has the wrong dimension.\n"
      "  - Expected length: " << dim << "\n"
      "  - Input length   : " << scale.size() << "\n");

  std::string topo_str = "Simplex";
  if (m_params->isParameter("Topology Type")) {
    topo_str = m_params->get<std::string>("Topology Type");
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

#ifdef DEBUG_OUTPUT
  // This chunk prints ALL mesh entities, along with their geometric classification and centroids
  int mdim = m_mesh->dim();
  auto coords = m_mesh->coords();
  Omega_h::Reals centroids;
  Omega_h::TagSet tags;
  Omega_h::get_all_dim_tags(m_mesh.get(),0,&tags);
  for (int idim=0; idim<=mdim; ++idim) {
    std::cout << "DIM=" << idim << "\n";
    for (auto n : tags[idim]) {
      auto tag = m_mesh->get_tagbase(idim,n);
      auto ids = tag->class_ids();
      std::cout << "tag=" << n << ", ids:";
      for (int i=0;i<ids.size(); ++i) {
        std::cout << " " << ids[i];
      } std::cout << "\n";
    }
    auto class_id = m_mesh->get_tag<Omega_h::ClassId>(idim,"class_id");
    auto class_dim = m_mesh->get_tag<Omega_h::Byte>(idim,"class_dim");
    auto gids = m_mesh->globals(idim);
    std::cout << "  gids size: " << gids.size() << "\n";
    std::cout << "  class_id size: " << class_id->array().size() << "\n";
    std::cout << "  class_dim size: " << class_dim->array().size() << "\n";
    for (int i=0; i<gids.size(); ++i) {
      std::cout << "i=" << i
                << ", gid=" << gids[i]
                << ", class_id=" << class_id->array()[i]
                << ", class_dim=" << static_cast<int>(class_dim->array()[i]);
      if (idim==0) {
        centroids = coords;
      } else {
        centroids = Omega_h::average_field(m_mesh.get(), idim, Omega_h::LOs(m_mesh->nents(idim), 0, 1),mdim,coords);
      }
      std::cout << ", centroids=[" << centroids[mdim*i];
      for (int k=1; k<mdim; ++k) {
        std::cout << " " << centroids[mdim*i+k];
      }
      std::cout << "]\n";
    }
  }
#endif
  m_mesh->set_parting(OMEGA_H_ELEM_BASED);
  setCoordinates();

  // Create the mesh specs
  std::string ebName = "element_block_0";
  std::map<std::string,int> ebNameToIndex =
  {
    { ebName, 0}
  };
  this->declare_part(ebName,elem_topo);

  nodeSetsGeoModelEntities = setNodeSetsGeoModelEntities();
  sideSetsGeoModelEntities = setSideSetsGeoModelEntities();
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
  int ws_size_max = m_params->get<int>("Workset Size", -1);
  int ws_size = computeWorksetSize(ws_size_max,m_mesh->nelems());
  this->meshSpecs[0] = Teuchos::rcp(
      new MeshSpecsStruct(MeshType::Structured, *ctd, dim,
                          nsNames, ssNames, ws_size, ebName,
                          ebNameToIndex));
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

    m_field_accessor->setFieldOnMesh (fname,nodal ? 0 : m_mesh->dim(), field_mv.getConst());
  }
}

} // namespace Albany
