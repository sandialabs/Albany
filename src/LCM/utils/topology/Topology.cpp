//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <boost/foreach.hpp>

#include "Subgraph.h"
#include "Topology.h"
#include "Topology_FractureCriterion.h"
#include "Topology_Utils.h"
#include "stk_mesh/base/FEMHelpers.hpp"

#include <Albany_STKNodeSharing.hpp>

namespace LCM {

//
// Default constructor
//
Topology::Topology()
    : discretization_(Teuchos::null),
      stk_mesh_struct_(Teuchos::null),
      fracture_criterion_(Teuchos::null),
      output_type_(UNIDIRECTIONAL_UNILEVEL)
{
  return;
}

//
// Constructor with input and output files.
//
Topology::Topology(
    std::string const& input_file,
    std::string const& output_file)
    : discretization_(Teuchos::null),
      stk_mesh_struct_(Teuchos::null),
      fracture_criterion_(Teuchos::null),
      output_type_(UNIDIRECTIONAL_UNILEVEL)
{
  Teuchos::RCP<Teuchos::ParameterList> params =
      Teuchos::rcp(new Teuchos::ParameterList("params"));

  // Create discretization object
  Teuchos::RCP<Teuchos::ParameterList> disc_params =
      Teuchos::sublist(params, "Discretization");

  // Set Method to Exodus and set input file name
  disc_params->set<std::string>("Method", "Exodus");
  disc_params->set<std::string>("Exodus Input File Name", input_file);
  disc_params->set<std::string>("Exodus Output File Name", output_file);
  disc_params->set<int>("Number Of Time Derivatives", 0);

  Teuchos::RCP<Teuchos::ParameterList> problem_params =
      Teuchos::sublist(params, "Problem");

  // Make adaptation list to force Albany::DiscretizationFactory
  // to create interface block.
  Teuchos::RCP<Teuchos::ParameterList> adapt_params =
      Teuchos::sublist(problem_params, "Adaptation");

  Teuchos::RCP<Teuchos_Comm> communicatorT =
      Albany::createTeuchosCommFromMpiComm(Albany_MPI_COMM_WORLD);
  adapt_params->set<std::string>("Method", "Topmod");

  std::string const bulk_block_name = "Bulk Element";

  adapt_params->set<std::string>("Bulk Block Name", bulk_block_name);

  std::string const interface_block_name = "Surface Element";

  adapt_params->set<std::string>("Interface Block Name", interface_block_name);

  set_bulk_block_name(bulk_block_name);

  set_interface_block_name(interface_block_name);

  Albany::DiscretizationFactory disc_factory(params, communicatorT);

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs =
      disc_factory.createMeshSpecs();

  Teuchos::RCP<Albany::StateInfoStruct> state_info =
      Teuchos::rcp(new Albany::StateInfoStruct());

  // The default fields
  Albany::AbstractFieldContainer::FieldContainerRequirements req;

  Teuchos::RCP<Albany::AbstractDiscretization> const& abstract_disc =
      disc_factory.createDiscretization(3, state_info, req);

  set_discretization(abstract_disc);

  Albany::STKDiscretization& stk_disc =
      static_cast<Albany::STKDiscretization&>(*abstract_disc);

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct =
      stk_disc.getSTKMeshStruct();

  set_stk_mesh_struct(stk_mesh_struct);

  Topology::graphInitialization();

  return;
}

//
// Construct by using given discretization.
//
Topology::Topology(
    Teuchos::RCP<Albany::AbstractDiscretization>& abstract_disc,
    std::string const&                            bulk_block_name,
    std::string const&                            interface_block_name)
    : discretization_(Teuchos::null),
      stk_mesh_struct_(Teuchos::null),
      fracture_criterion_(Teuchos::null),
      output_type_(UNIDIRECTIONAL_UNILEVEL)
{
  set_discretization(abstract_disc);

  Albany::STKDiscretization& stk_disc =
      static_cast<Albany::STKDiscretization&>(*abstract_disc);

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct =
      stk_disc.getSTKMeshStruct();

  set_stk_mesh_struct(stk_mesh_struct);

  set_bulk_block_name(bulk_block_name);

  set_interface_block_name(interface_block_name);

  Topology::graphInitialization();

  return;
}

//
//
//
stk::mesh::EntityId
Topology::get_entity_id(stk::mesh::Entity const entity)
{
  size_t const space_dimension = get_space_dimension();

  int const parallel_rank = get_bulk_data().parallel_rank();

  stk::mesh::EntityRank const rank = get_bulk_data().entity_rank(entity);

  stk::mesh::EntityId const high_id = get_bulk_data().identifier(entity);

  stk::mesh::EntityId const low_id =
      low_id_from_high_id(space_dimension, parallel_rank, rank, high_id);

  return low_id;
}

//
// Check fracture criterion
//
bool
Topology::checkOpen(stk::mesh::Entity e)
{
  return fracture_criterion_->check(get_bulk_data(), e);
}

//
// Initialize fracture state field
// It exists for all entities except cells (elements)
//
void
Topology::initializeFractureState()
{
  stk::mesh::Selector local_part = get_local_part();

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK;
       rank < stk::topology::ELEMENT_RANK;
       ++rank) {
    std::vector<stk::mesh::Bucket*> const& buckets =
        get_bulk_data().buckets(rank);

    stk::mesh::EntityVector entities;

    stk::mesh::get_selected_entities(local_part, buckets, entities);

    for (EntityVectorIndex i = 0; i < entities.size(); ++i) {
      stk::mesh::Entity entity = entities[i];

      set_fracture_state(entity, CLOSED);
    }
  }

  return;
}

//
// Create the full mesh representation. This must be done prior to
// the adaptation query.
//
void
Topology::graphInitialization()
{
  stk::mesh::PartVector add_parts;

  stk::mesh::create_adjacent_entities(get_bulk_data(), add_parts);

  get_bulk_data().modification_begin();
  removeMultiLevelRelations();
  initializeFractureState();
  Albany::fix_node_sharing(get_bulk_data());
  get_bulk_data().modification_end();
  get_stk_discretization().updateMesh();

  initializeTopologies();
  initializeHighestIds();

  return;
}

//
// Creates temporary nodal connectivity for the elements and removes
// the relationships between the elements and nodes.
//
void
Topology::removeNodeRelations()
{
  // Create the temporary connectivity array
  stk::mesh::EntityVector elements;

  stk::mesh::get_entities(
      get_bulk_data(), stk::topology::ELEMENT_RANK, elements);

  get_bulk_data().modification_begin();

  for (size_t i = 0; i < elements.size(); ++i) {
    stk::mesh::Entity const* relations =
        get_bulk_data().begin_nodes(elements[i]);
    size_t const num_relations = get_bulk_data().num_nodes(elements[i]);

    stk::mesh::EntityVector nodes(relations, relations + num_relations);

    connectivity_.push_back(nodes);

    for (size_t j = 0; j < nodes.size(); ++j) {
      get_bulk_data().destroy_relation(elements[i], nodes[j], j);
    }
  }

  Albany::fix_node_sharing(get_bulk_data());
  get_bulk_data().modification_end();

  return;
}

//
// Removes multilevel relations.
//
void
Topology::removeMultiLevelRelations()
{
  typedef std::vector<EdgeId>   EdgeIdList;
  typedef EdgeIdList::size_type EdgeIdListIndex;

  // Go from points to cells
  for (stk::mesh::EntityRank source_rank = stk::topology::NODE_RANK;
       source_rank <= stk::topology::ELEMENT_RANK;
       ++source_rank) {
    stk::mesh::EntityVector source_entities;

    stk::mesh::get_entities(get_bulk_data(), source_rank, source_entities);

    bool const source_is_point = source_rank == stk::topology::NODE_RANK;

    for (RelationVectorIndex i = 0; i < source_entities.size(); ++i) {
      stk::mesh::Entity source_entity = source_entities[i];

      stk::mesh::EntityVector target_entities;

      EdgeIdList multilevel_relation_ids;

      for (stk::mesh::EntityRank target_rank = stk::topology::NODE_RANK;
           target_rank < get_meta_data().entity_rank_count();
           ++target_rank) {
        stk::mesh::Entity const* relations =
            get_bulk_data().begin(source_entity, target_rank);

        size_t const num_relations =
            get_bulk_data().num_connectivity(source_entity, target_rank);

        stk::mesh::ConnectivityOrdinal const* ordinals =
            get_bulk_data().begin_ordinals(source_entity, target_rank);

        bool const target_is_point = target_rank == stk::topology::NODE_RANK;

        // Collect relations to delete
        for (size_t r = 0; r < num_relations; ++r) {
          size_t const rank_distance =
              source_rank > target_rank ? source_rank - target_rank : 0;

          bool const end_is_point = source_is_point || target_is_point;

          bool const is_invalid_relation =
              end_is_point == false && rank_distance > 1;

          if (is_invalid_relation == true) {
            target_entities.push_back(relations[r]);
            multilevel_relation_ids.push_back(ordinals[r]);
          }
        }
      }

      // Delete them
      for (EdgeIdListIndex i = 0; i < multilevel_relation_ids.size(); ++i) {
        stk::mesh::Entity far_entity = target_entities[i];

        EdgeId const multilevel_relation_id = multilevel_relation_ids[i];

        get_bulk_data().destroy_relation(
            source_entity, far_entity, multilevel_relation_id);
      }
    }
  }

  return;
}

//
// After mesh manipulations are complete, need to recreate a stk
// mesh understood by Albany::STKDiscretization.
//
void
Topology::restoreElementToNodeConnectivity()
{
  stk::mesh::EntityVector elements;

  stk::mesh::get_entities(
      get_bulk_data(), stk::topology::ELEMENT_RANK, elements);

  get_bulk_data().modification_begin();

  // Add relations from element to nodes
  for (size_t i = 0; i < elements.size(); ++i) {
    stk::mesh::Entity element = elements[i];

    stk::mesh::EntityVector element_connectivity = connectivity_[i];

    for (size_t j = 0; j < element_connectivity.size(); ++j) {
      stk::mesh::Entity node = element_connectivity[j];

      get_bulk_data().declare_relation(element, node, j);
    }
  }

  // Recreate Albany STK Discretization
  Albany::STKDiscretization& stk_discretization =
      static_cast<Albany::STKDiscretization&>(*discretization_);

  Teuchos::RCP<Teuchos_Comm> communicatorT =
      Albany::createTeuchosCommFromMpiComm(Albany_MPI_COMM_WORLD);

  Albany::fix_node_sharing(get_bulk_data());
  get_bulk_data().modification_end();

  return;
}

//
// Determine nodes associated with a boundary entity
//
stk::mesh::EntityVector
Topology::getBoundaryEntityNodes(stk::mesh::Entity boundary_entity)
{
  stk::mesh::EntityRank const boundary_rank =
      get_bulk_data().entity_rank(boundary_entity);

  assert(boundary_rank == stk::topology::ELEMENT_RANK - 1);

  stk::mesh::Entity const* relations =
      get_bulk_data().begin_elements(boundary_entity);

  stk::mesh::Entity first_cell = relations[0];

  size_t const num_cell_nodes = get_bulk_data().num_nodes(first_cell);

  stk::mesh::Entity const* node_relations =
      get_bulk_data().begin_nodes(first_cell);

  stk::mesh::ConnectivityOrdinal const* node_ords =
      get_bulk_data().begin_node_ordinals(first_cell);

  stk::mesh::ConnectivityOrdinal const* ords =
      get_bulk_data().begin_element_ordinals(boundary_entity);

  EdgeId const face_order = ords[0];

  shards::CellTopology const cell_topology = get_cell_topology();

  RelationVectorIndex const number_face_nodes =
      cell_topology.getNodeCount(boundary_rank, face_order);

  stk::mesh::EntityVector nodes;

  for (RelationVectorIndex i = 0; i < number_face_nodes; ++i) {
    EdgeId const cell_order =
        cell_topology.getNodeMap(boundary_rank, face_order, i);

    // Brute force approach. Maybe there is a better way to do this?
    for (size_t j = 0; j < num_cell_nodes; ++j) {
      if (node_ords[j] == cell_order) { nodes.push_back(node_relations[j]); }
    }
  }

  return nodes;
}

//
// Get nodal coordinates
//
std::vector<minitensor::Vector<double>>
Topology::getNodalCoordinates()
{
  stk::mesh::Selector local_selector = get_meta_data().locally_owned_part();

  std::vector<stk::mesh::Bucket*> const& buckets =
      get_bulk_data().buckets(stk::topology::NODE_RANK);

  stk::mesh::EntityVector entities;

  stk::mesh::get_selected_entities(local_selector, buckets, entities);

  EntityVectorIndex const number_nodes = entities.size();

  std::vector<minitensor::Vector<double>> coordinates(number_nodes);

  size_t const dimension = get_space_dimension();

  minitensor::Vector<double> X(dimension);

  VectorFieldType& node_coordinates =
      *(get_stk_mesh_struct()->getCoordinatesField());

  for (EntityVectorIndex i = 0; i < number_nodes; ++i) {
    stk::mesh::Entity node = entities[i];

    double const* const pointer_coordinates =
        stk::mesh::field_data(node_coordinates, node);

    for (size_t j = 0; j < dimension; ++j) { X(j) = pointer_coordinates[j]; }

    coordinates[i] = X;
  }

  return coordinates;
}

//
// Output of boundary
//
void
Topology::outputBoundary(std::string const& output_filename)
{
  // Open output file
  std::ofstream ofs;

  ofs.open(output_filename.c_str(), std::ios::out);

  if (ofs.is_open() == false) {
    std::cout << "Unable to open boundary output file: ";
    std::cout << output_filename << '\n';
    return;
  }

  std::cout << "Write boundary file: ";
  std::cout << output_filename << '\n';

  // Header
  ofs << "# vtk DataFile Version 3.0\n";
  ofs << "Albany/LCM\n";
  ofs << "ASCII\n";
  ofs << "DATASET UNSTRUCTURED_GRID\n";

  // Coordinates
  Coordinates const coordinates = getNodalCoordinates();

  CoordinatesIndex const number_nodes = coordinates.size();

  ofs << "POINTS " << number_nodes << " double\n";

  for (CoordinatesIndex i = 0; i < number_nodes; ++i) {
    minitensor::Vector<double> const& X = coordinates[i];

    for (minitensor::Index j = 0; j < X.get_dimension(); ++j) {
      ofs << std::setw(24) << std::scientific << std::setprecision(16) << X(j);
    }
    ofs << '\n';
  }

  Connectivity const connectivity = getBoundary();

  ConnectivityIndex const number_cells = connectivity.size();

  size_t cell_list_size = 0;

  for (size_t i = 0; i < number_cells; ++i) {
    cell_list_size += connectivity[i].size() + 1;
  }

  // Boundary cell connectivity
  ofs << "CELLS " << number_cells << " " << cell_list_size << '\n';
  for (size_t i = 0; i < number_cells; ++i) {
    size_t const number_cell_nodes = connectivity[i].size();

    ofs << number_cell_nodes;

    for (size_t j = 0; j < number_cell_nodes; ++j) {
      ofs << ' ' << connectivity[i][j] - 1;
    }
    ofs << '\n';
  }

  ofs << "CELL_TYPES " << number_cells << '\n';
  for (size_t i = 0; i < number_cells; ++i) {
    size_t const number_cell_nodes = connectivity[i].size();

    VTKCellType cell_type = INVALID;

    switch (number_cell_nodes) {
      default:
        std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
        std::cerr << '\n';
        std::cerr << "Invalid number of nodes in boundary cell: ";
        std::cerr << number_cell_nodes;
        std::cerr << '\n';
        exit(1);
        break;

      case 1: cell_type = VERTEX; break;

      case 2: cell_type = LINE; break;

      case 3: cell_type = TRIANGLE; break;

      case 4: cell_type = QUAD; break;
    }
    ofs << cell_type << '\n';
  }

  ofs.close();
  return;
}

//
// Create boundary mesh
//
Connectivity
Topology::getBoundary()
{
  stk::mesh::EntityRank const boundary_rank = get_boundary_rank();

  stk::mesh::EntityRank const cell_rank =
      static_cast<stk::mesh::EntityRank>(boundary_rank + 1);

  stk::mesh::EntityVector faces;

  stk::mesh::get_entities(get_bulk_data(), boundary_rank, faces);

  Connectivity connectivity;

  EntityVectorIndex const number_faces = faces.size();

  for (EntityVectorIndex i = 0; i < number_faces; ++i) {
    stk::mesh::Entity const face = faces[i];

    stk::mesh::Entity const* cell_relations =
        get_bulk_data().begin(face, cell_rank);

    size_t const number_connected_cells = get_bulk_data().num_elements(face);

    switch (number_connected_cells) {
      default:
        std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
        std::cerr << '\n';
        std::cerr << "Invalid number of connected cells: ";
        std::cerr << number_connected_cells;
        std::cerr << '\n';
        exit(1);
        break;

      case 1:
        // Make sure connected cell is in bulk and continue.
        assert(is_bulk_cell(cell_relations[0]) == true);
        break;

      case 2:
        // Two connected faces. Determine 1f one of them is a surface element.
        {
          stk::mesh::Entity const cell_0 = cell_relations[0];

          stk::mesh::Entity const cell_1 = cell_relations[1];

          bool const is_in_bulk_0 = is_in_bulk(cell_0);

          bool const is_in_bulk_1 = is_in_bulk(cell_1);

          bool const both_bulk = is_in_bulk_0 && is_in_bulk_1;

          // If internal face do nothing.
          if (both_bulk == true) continue;

          bool const is_in_interface_0 = is_in_interface(cell_0);

          bool const is_in_interface_1 = is_in_interface(cell_1);

          bool const both_interface = is_in_interface_0 && is_in_interface_1;

          if (both_interface == true) {
            std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
            std::cerr << '\n';
            std::cerr << "Cannot be connected to two surface elements: ";
            std::cerr << "Face: " << get_entity_id(face);
            std::cerr << '\n';
            exit(1);
          }

          // One element is bulk and one interface, so it's an internal face.
        }
        break;
    }

    stk::mesh::EntityVector const nodes = getBoundaryEntityNodes(face);

    EntityVectorIndex const number_nodes = nodes.size();

    std::vector<stk::mesh::EntityId> node_ids(number_nodes);

    for (EntityVectorIndex i = 0; i < number_nodes; ++i) {
      node_ids[i] = get_entity_id(nodes[i]);
    }

    connectivity.push_back(node_ids);
  }

  return connectivity;
}

///
/// Compute normal using first 3 nodes of boundary entity.
///
minitensor::Vector<double>
Topology::get_normal(stk::mesh::Entity boundary_entity)
{
  shards::CellTopology const cell_topology = get_cell_topology();

  stk::mesh::EntityRank const boundary_rank =
      get_bulk_data().entity_rank(boundary_entity);

  stk::mesh::ConnectivityOrdinal const* ords =
      get_bulk_data().begin_element_ordinals(boundary_entity);

  EdgeId const face_order = ords[0];

  RelationVectorIndex const num_corner_nodes =
      cell_topology.getVertexCount(boundary_rank, face_order);

  assert(num_corner_nodes >= 3);

  size_t const dimension = get_space_dimension();

  std::vector<minitensor::Vector<double>> nodal_coords(num_corner_nodes);

  stk::mesh::EntityVector nodes = getBoundaryEntityNodes(boundary_entity);

  VectorFieldType& coordinates =
      *(get_stk_mesh_struct()->getCoordinatesField());

  for (EntityVectorIndex i = 0; i < num_corner_nodes; ++i) {
    stk::mesh::Entity node = nodes[i];

    double const* const pointer_coordinates =
        stk::mesh::field_data(coordinates, node);

    minitensor::Vector<double>& X = nodal_coords[i];

    X.set_dimension(dimension);

    for (size_t j = 0; j < dimension; ++j) { X(j) = pointer_coordinates[j]; }
  }

  minitensor::Vector<double> const v1 = nodal_coords[1] - nodal_coords[0];

  minitensor::Vector<double> const v2 = nodal_coords[2] - nodal_coords[0];

  minitensor::Vector<double> normal =
      minitensor::unit(minitensor::cross(v1, v2));

  return normal;
}

//
// Create cohesive connectivity
//
stk::mesh::EntityVector
Topology::createSurfaceElementConnectivity(
    stk::mesh::Entity face_top,
    stk::mesh::Entity face_bottom)
{
  // Check first if normals point in the same direction, just in case.
  minitensor::Vector<double> const normal_top = get_normal(face_top);

  minitensor::Vector<double> const normal_bottom = get_normal(face_bottom);

  if (minitensor::dot(normal_top, normal_bottom) > 0.0) {
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "Face normals have the same instead of opposite directions:\n";
    std::cerr << "Normal top    :" << normal_top << '\n';
    std::cerr << "Normal bottom :" << normal_bottom << '\n';
    std::cerr << '\n';
    exit(1);
  }

  stk::mesh::EntityVector nodes_top = getBoundaryEntityNodes(face_top);

  EntityVectorIndex const num_top = nodes_top.size();

  stk::mesh::EntityVector nodes_bottom = getBoundaryEntityNodes(face_bottom);

  EntityVectorIndex const num_bottom = nodes_bottom.size();

  assert(num_top == num_bottom);

  stk::mesh::EntityVector reordered;

  // Ensure that the order of the bottom nodes is the same as the top ones.
  VectorFieldType& coordinates =
      *(get_stk_mesh_struct()->getCoordinatesField());

  size_t const dimension = get_space_dimension();

  for (EntityVectorIndex i = 0; i < num_top; ++i) {
    stk::mesh::Entity node_top = nodes_top[i];

    double const* const p_top = stk::mesh::field_data(coordinates, node_top);

    minitensor::Vector<double> X(dimension);

    for (size_t n = 0; n < dimension; ++n) { X(n) = p_top[n]; }

    bool found = false;

    for (EntityVectorIndex j = 0; j < num_bottom; ++j) {
      stk::mesh::Entity node_bottom = nodes_bottom[j];

      double const* const p_bottom =
          stk::mesh::field_data(coordinates, node_bottom);

      minitensor::Vector<double> Y(dimension);

      for (size_t n = 0; n < dimension; ++n) { Y(n) = p_bottom[n]; }

      if (X == Y) {
        reordered.push_back(node_bottom);
        found = true;
        break;
      }
    }

    if (found == false) {
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Nodes on top and bottom faces do not match.";
      std::cerr << '\n';
      exit(1);
    }
  }

  stk::mesh::EntityVector nodes;

  nodes.reserve(nodes_top.size() + nodes_bottom.size());

  nodes.insert(nodes.end(), nodes_top.begin(), nodes_top.end());
  nodes.insert(nodes.end(), reordered.begin(), reordered.end());

  return nodes;
}

//
// Create vectors describing the vertices and edges of the star of
// an entity in the stk mesh.
//
void
Topology::createStar(
    stk::mesh::Entity                entity,
    std::set<stk::mesh::Entity>&     subgraph_entities,
    std::set<STKEdge, EdgeLessThan>& subgraph_edges)
{
  subgraph_entities.insert(entity);

  assert(get_space_dimension() == 3);

  stk::mesh::EntityRank const rank = get_bulk_data().entity_rank(entity);

  stk::mesh::EntityRank const one_up =
      static_cast<stk::mesh::EntityRank>(rank + 1);

  stk::mesh::Entity const* relations = get_bulk_data().begin(entity, one_up);

  size_t const num_relations = get_bulk_data().num_connectivity(entity, one_up);

  stk::mesh::ConnectivityOrdinal const* ords =
      get_bulk_data().begin_ordinals(entity, one_up);

  for (size_t i = 0; i < num_relations; ++i) {
    stk::mesh::Entity source = relations[i];

    if (is_interface_cell(source) == true) continue;

    STKEdge edge;

    edge.source   = source;
    edge.target   = entity;
    edge.local_id = ords[i];

    subgraph_edges.insert(edge);
    createStar(source, subgraph_entities, subgraph_edges);
  }

  return;
}

//
// Fractures all open boundary entities of the mesh.
//
void
Topology::splitOpenFaces()
{
  // 3D only for now.
  assert(get_space_dimension() == stk::topology::ELEMENT_RANK);

  stk::mesh::EntityVector points;

  stk::mesh::EntityVector open_points;

  stk::mesh::Selector local_bulk = get_local_bulk_selector();

  std::set<EntityPair> fractured_faces;

  stk::mesh::BulkData& bulk_data = get_bulk_data();

  stk::mesh::BucketVector const& point_buckets =
      bulk_data.buckets(stk::topology::NODE_RANK);

  stk::mesh::get_selected_entities(local_bulk, point_buckets, points);

  // Collect open points
  for (stk::mesh::EntityVector::iterator i = points.begin(); i != points.end();
       ++i) {
    stk::mesh::Entity point = *i;

    if (get_fracture_state(point) == OPEN) { open_points.push_back(point); }
  }

#if defined(DEBUG_LCM_TOPOLOGY)
  {
    std::string const file_name = LCM::parallelize_string("before") + ".dot";

    outputToGraphviz(file_name);

    std::string const boundary_filename =
        LCM::parallelize_string("before") + ".vtk";
    outputBoundary(boundary_filename);
  }
#endif  // DEBUG_LCM_TOPOLOGY

  bulk_data.modification_begin();

  // Iterate over open points and fracture them.
  for (stk::mesh::EntityVector::iterator i = open_points.begin();
       i != open_points.end();
       ++i) {
    stk::mesh::Entity point = *i;

    stk::mesh::Entity const* segment_relations = bulk_data.begin_edges(point);

    size_t const num_segments = bulk_data.num_edges(point);

    stk::mesh::EntityVector open_segments;

    // Collect open segments.
    for (size_t j = 0; j < num_segments; ++j) {
      stk::mesh::Entity segment = segment_relations[j];

      bool const is_local_segment = is_local_entity(segment) == true;

      bool const is_open_segment = get_fracture_state(segment) == OPEN;

      bool const is_local_and_open_segment =
          is_local_segment == true && is_open_segment == true;

      if (is_local_and_open_segment == true) {
        open_segments.push_back(segment);
      }
    }

#if defined(DEBUG_LCM_TOPOLOGY)
    {
      std::string const file_name =
          LCM::parallelize_string(
              "graph-pre-segment-" + entity_string(get_topology(), point)) +
          ".dot";

      outputToGraphviz(file_name);
    }
#endif  // DEBUG_LCM_TOPOLOGY

    // Iterate over open segments and fracture them.
    for (stk::mesh::EntityVector::iterator j = open_segments.begin();
         j != open_segments.end();
         ++j) {
      stk::mesh::Entity segment = *j;

      // Create star of segment
      std::set<stk::mesh::Entity> star_entities;

      std::set<STKEdge, EdgeLessThan> star_edges;

      createStar(segment, star_entities, star_edges);

      // Iterators
      std::set<stk::mesh::Entity>::iterator first_entity =
          star_entities.begin();

      std::set<stk::mesh::Entity>::iterator last_entity = star_entities.end();

      std::set<STKEdge>::iterator first_edge = star_edges.begin();

      std::set<STKEdge>::iterator last_edge = star_edges.end();

      Subgraph segment_star(
          *this, first_entity, last_entity, first_edge, last_edge);

#if defined(DEBUG_LCM_TOPOLOGY)
      {
        std::string const file_name =
            LCM::parallelize_string(
                "graph-pre-clone-" + entity_string(get_topology(), segment)) +
            ".dot";

        outputToGraphviz(file_name);
        segment_star.outputToGraphviz("sub" + file_name);
      }
#endif  // DEBUG_LCM_TOPOLOGY

      // Collect open faces
      stk::mesh::Entity const* face_relations = bulk_data.begin_faces(segment);

      size_t const num_faces = bulk_data.num_faces(segment);

      stk::mesh::EntityVector open_faces;

      for (size_t k = 0; k < num_faces; ++k) {
        stk::mesh::Entity face = face_relations[k];

        bool const is_local_face = is_local_entity(face);

        bool const is_open_face = is_internal_and_open(face) == true;

        bool const is_local_and_open_face =
            is_local_face == true && is_open_face == true;

        if (is_local_and_open_face == true) { open_faces.push_back(face); }
      }

      // Iterate over the open faces
      for (stk::mesh::EntityVector::iterator k = open_faces.begin();
           k != open_faces.end();
           ++k) {
        stk::mesh::Entity face = *k;

        Vertex face_vertex = segment_star.vertexFromEntity(face);

        Vertex new_face_vertex = segment_star.cloneBoundaryVertex(face_vertex);

        stk::mesh::Entity new_face =
            segment_star.entityFromVertex(new_face_vertex);

        // Reset fracture state for both old and new faces
        set_fracture_state(face, CLOSED);
        set_fracture_state(new_face, CLOSED);

        EntityPair face_pair = std::make_pair(face, new_face);

        fractured_faces.insert(face_pair);
      }

      // Split the articulation point (current segment)
      Vertex segment_vertex = segment_star.vertexFromEntity(segment);

#if defined(DEBUG_LCM_TOPOLOGY)
      {
        std::string const file_name =
            LCM::parallelize_string(
                "graph-pre-split-" + entity_string(get_topology(), segment)) +
            ".dot";

        outputToGraphviz(file_name);
        segment_star.outputToGraphviz("sub" + file_name);
      }
#endif  // DEBUG_LCM_TOPOLOGY

      segment_star.splitArticulation(segment_vertex);

      // Reset segment fracture state
      set_fracture_state(segment, CLOSED);

#if defined(DEBUG_LCM_TOPOLOGY)
      {
        std::string const file_name =
            LCM::parallelize_string(
                "graph-post-split-" + entity_string(get_topology(), segment)) +
            ".dot";

        outputToGraphviz(file_name);
        segment_star.outputToGraphviz("sub" + file_name);
      }
#endif  // DEBUG_LCM_TOPOLOGY
    }

    // All open faces and segments have been dealt with.
    // Split the node articulation point
    // Create star of node
    std::set<stk::mesh::Entity> star_entities;

    std::set<STKEdge, EdgeLessThan> star_edges;

    createStar(point, star_entities, star_edges);

    // Iterators
    std::set<stk::mesh::Entity>::iterator first_entity = star_entities.begin();

    std::set<stk::mesh::Entity>::iterator last_entity = star_entities.end();

    std::set<STKEdge>::iterator first_edge = star_edges.begin();

    std::set<STKEdge>::iterator last_edge = star_edges.end();

    Subgraph point_star(
        *this, first_entity, last_entity, first_edge, last_edge);

    Vertex point_vertex = point_star.vertexFromEntity(point);

#if defined(DEBUG_LCM_TOPOLOGY)
    {
      std::string const file_name =
          LCM::parallelize_string(
              "graph-pre-split-" + entity_string(get_topology(), point)) +
          ".dot";

      outputToGraphviz(file_name);
      point_star.outputToGraphviz("sub" + file_name);
    }
#endif  // DEBUG_LCM_TOPOLOGY

    EntityEntityMap new_connectivity =
        point_star.splitArticulation(point_vertex);

    // Reset fracture state of point
    set_fracture_state(point, CLOSED);

#if defined(DEBUG_LCM_TOPOLOGY)
    {
      std::string const file_name =
          LCM::parallelize_string(
              "graph-post-split-" + entity_string(get_topology(), point)) +
          ".dot";

      outputToGraphviz(file_name);
      point_star.outputToGraphviz("sub" + file_name);
    }
#endif  // DEBUG_LCM_TOPOLOGY

    // Update the connectivity
    for (EntityEntityMap::iterator j = new_connectivity.begin();
         j != new_connectivity.end();
         ++j) {
      stk::mesh::Entity new_point = j->second;

      bulk_data.copy_entity_fields(point, new_point);
    }
  }

  Albany::fix_node_sharing(bulk_data);
  bulk_data.modification_end();

#if defined(DEBUG_LCM_TOPOLOGY)
  {
    std::string const file_name =
        LCM::parallelize_string("graph-pre-surface-elements.dot");

    outputToGraphviz(file_name);
  }
#endif  // DEBUG_LCM_TOPOLOGY

  bool const insert_surface_elements = false;

  if (insert_surface_elements == true) {
    insertSurfaceElements(fractured_faces);
  }

  return;
}

//
//
//
void
Topology::insertSurfaceElements(std::set<EntityPair> const& fractured_faces)
{
  stk::mesh::BulkData& bulk_data = get_bulk_data();

  bulk_data.modification_begin();

  // Same rank as bulk cells!
  stk::mesh::EntityRank const interface_rank = stk::topology::ELEMENT_RANK;

  stk::mesh::Part& interface_part = fracture_criterion_->get_interface_part();

  stk::mesh::PartVector interface_parts;

  interface_parts.push_back(&interface_part);

  increase_highest_id(interface_rank);

  stk::mesh::EntityId new_id = get_highest_id(interface_rank);

  // Create the interface connectivity
  for (std::set<EntityPair>::iterator i = fractured_faces.begin();
       i != fractured_faces.end();
       ++i) {
    stk::mesh::Entity face1 = i->first;

    stk::mesh::Entity face2 = i->second;

    stk::mesh::EntityVector interface_points =
        createSurfaceElementConnectivity(face1, face2);

    // Insert the surface element
    stk::mesh::Entity new_surface =
        bulk_data.declare_entity(interface_rank, new_id, interface_parts);

    // Connect to faces
    bulk_data.declare_relation(new_surface, face1, 0);
    bulk_data.declare_relation(new_surface, face2, 1);

    // Connect to points
    for (EntityVectorIndex j = 0; j < interface_points.size(); ++j) {
      stk::mesh::Entity interface_point = interface_points[j];

      bulk_data.declare_relation(new_surface, interface_point, j);
    }

    ++new_id;
  }

  Albany::fix_node_sharing(bulk_data);
  bulk_data.modification_end();

#if defined(DEBUG_LCM_TOPOLOGY)
  {
    std::string const file_name = LCM::parallelize_string("after") + ".dot";

    outputToGraphviz(file_name);

    std::string const boundary_filename =
        LCM::parallelize_string("after") + ".vtk";
    outputBoundary(boundary_filename);
  }
#endif  // DEBUG_LCM_TOPOLOGY

  return;
}

//
//
//
size_t
Topology::setEntitiesOpen()
{
  stk::mesh::EntityVector boundary_entities;

  stk::mesh::Selector local_bulk = get_local_bulk_selector();

  stk::mesh::get_selected_entities(
      local_bulk,
      get_bulk_data().buckets(get_boundary_rank()),
      boundary_entities);

  size_t counter = 0;

  // Iterate over the boundary entities
  for (EntityVectorIndex i = 0; i < boundary_entities.size(); ++i) {
    stk::mesh::Entity entity = boundary_entities[i];

    if (is_internal(entity) == false) continue;

    if (checkOpen(entity) == false) continue;

    set_fracture_state(entity, OPEN);
    ++counter;

    switch (get_space_dimension()) {
      default:
        std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
        std::cerr << '\n';
        std::cerr << "Invalid cells rank in fracture: ";
        std::cerr << stk::topology::ELEMENT_RANK;
        std::cerr << '\n';
        exit(1);
        break;

      case stk::topology::ELEMENT_RANK: {
        stk::mesh::Entity const* segments = get_bulk_data().begin_edges(entity);

        size_t const num_segments = get_bulk_data().num_edges(entity);

        for (size_t j = 0; j < num_segments; ++j) {
          stk::mesh::Entity segment = segments[j];

          set_fracture_state(segment, OPEN);

          stk::mesh::Entity const* points =
              get_bulk_data().begin_nodes(segment);

          size_t const num_points = get_bulk_data().num_nodes(segment);

          for (size_t k = 0; k < num_points; ++k) {
            stk::mesh::Entity point = points[k];

            set_fracture_state(point, OPEN);
          }
        }
      } break;

      case stk::topology::EDGE_RANK: {
        stk::mesh::Entity const* points = get_bulk_data().begin_nodes(entity);

        size_t const num_points = get_bulk_data().num_nodes(entity);

        for (size_t j = 0; j < num_points; ++j) {
          stk::mesh::Entity point = points[j];

          set_fracture_state(point, OPEN);
        }
      } break;
    }
  }

  return counter;
}

//
// Output the graph associated with the mesh to graphviz .dot
// file for visualization purposes.
//
void
Topology::outputToGraphviz(std::string const& output_filename)
{
  // Open output file
  std::ofstream gviz_out;

  gviz_out.open(output_filename.c_str(), std::ios::out);

  if (gviz_out.is_open() == false) {
    std::cout << "Unable to open graphviz output file: ";
    std::cout << output_filename << '\n';
    return;
  }

  std::cout << "Write graph to graphviz dot file: ";
  std::cout << output_filename << '\n';

  // Write beginning of file
  gviz_out << dot_header();

  typedef std::vector<EntityPair> RelationList;

  RelationList relation_list;

  std::vector<EdgeId> relation_local_id;

  // Entities (graph vertices)
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK;
       rank <= stk::topology::ELEMENT_RANK;
       ++rank) {
    stk::mesh::EntityVector entities;

    stk::mesh::get_entities(get_bulk_data(), rank, entities);

    for (EntityVectorIndex i = 0; i < entities.size(); ++i) {
      stk::mesh::Entity source_entity = entities[i];

      FractureState const fracture_state = get_fracture_state(source_entity);

      stk::mesh::EntityId const source_id = get_entity_id(source_entity);

      gviz_out << dot_entity(
          get_space_dimension(),
          get_parallel_rank(),
          source_entity,
          source_id,
          rank,
          fracture_state);

      for (stk::mesh::EntityRank target_rank = stk::topology::NODE_RANK;
           target_rank < get_meta_data().entity_rank_count();
           ++target_rank) {
        unsigned const num_valid_conn =
            get_bulk_data().count_valid_connectivity(
                source_entity, target_rank);

        if (num_valid_conn > 0) {
          stk::mesh::Entity const* relations =
              get_bulk_data().begin(source_entity, target_rank);

          size_t const num_relations =
              get_bulk_data().num_connectivity(source_entity, target_rank);

          stk::mesh::ConnectivityOrdinal const* ords =
              get_bulk_data().begin_ordinals(source_entity, target_rank);

          for (size_t j = 0; j < num_relations; ++j) {
            stk::mesh::Entity target_entity = relations[j];

            bool is_valid_target_rank = false;

            OutputType const output_type = get_output_type();

            switch (output_type) {
              default:
                std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
                std::cerr << '\n';
                std::cerr << "Invalid output type: ";
                std::cerr << output_type;
                std::cerr << '\n';
                exit(1);
                break;

              case UNIDIRECTIONAL_UNILEVEL:
                is_valid_target_rank = target_rank + 1 == rank;
                break;

              case UNIDIRECTIONAL_MULTILEVEL:
                is_valid_target_rank = target_rank < rank;
                break;

              case BIDIRECTIONAL_UNILEVEL:
                is_valid_target_rank =
                    (target_rank == rank + 1) || (target_rank + 1 == rank);
                break;

              case BIDIRECTIONAL_MULTILEVEL:
                is_valid_target_rank = target_rank != rank;
                break;
            }

            if (is_valid_target_rank == false) continue;

            EntityPair entity_pair =
                std::make_pair(source_entity, target_entity);

            EdgeId const edge_id = ords[j];

            relation_list.push_back(entity_pair);
            relation_local_id.push_back(edge_id);
          }
        }
      }
    }
  }

  // Relations (graph edges)
  for (RelationList::size_type i = 0; i < relation_list.size(); ++i) {
    EntityPair entity_pair = relation_list[i];

    stk::mesh::Entity source = entity_pair.first;

    stk::mesh::Entity target = entity_pair.second;

    gviz_out << dot_relation(
        get_entity_id(source),
        get_bulk_data().entity_rank(source),
        get_entity_id(target),
        get_bulk_data().entity_rank(target),
        relation_local_id[i]);
  }

  // File end
  gviz_out << dot_footer();

  gviz_out.close();

  return;
}

//
//
//
void
Topology::initializeTopologies()
{
  size_t const dimension = get_space_dimension();

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK;
       rank <= stk::topology::ELEMENT_RANK;
       ++rank) {
    if (rank > dimension) break;

    std::vector<stk::mesh::Bucket*> buckets = get_bulk_data().buckets(rank);

    stk::mesh::Bucket const& bucket = *(buckets[0]);

    topologies_.push_back(bucket.topology());
  }
  return;
}

//
// Place the entity in the root part that has the stk::topology
// associated with the given rank.
//
void
Topology::AssignTopology(
    stk::mesh::EntityRank const rank,
    stk::mesh::Entity const     entity)
{
  stk::topology stk_topology = get_topology().get_rank_topology(rank);

  shards::CellTopology cell_topology =
      stk::mesh::get_cell_topology(stk_topology);

  stk::mesh::Part& part =
      get_meta_data().get_cell_topology_root_part(cell_topology);

  stk::mesh::PartVector add_parts;

  add_parts.push_back(&part);

  get_bulk_data().change_entity_parts(entity, add_parts);

  return;
}

//
// \brief This returns the number of entities of a given rank
//
EntityVectorIndex
Topology::get_num_entities(stk::mesh::EntityRank const entity_rank)
{
  std::vector<stk::mesh::Bucket*> buckets =
      get_bulk_data().buckets(entity_rank);

  EntityVectorIndex number_entities = 0;

  for (EntityVectorIndex i = 0; i < buckets.size(); ++i) {
    number_entities += buckets[i]->size();
  }

  return number_entities;
}

//
// \brief Determine highest id number for each entity rank.
// Used to assign unique ids to newly created entities
//
void
Topology::initializeHighestIds()
{
  size_t const dimension = get_space_dimension();

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK;
       rank <= stk::topology::ELEMENT_RANK;
       ++rank) {
    if (rank > dimension) break;

    highest_ids_.push_back(get_num_entities(rank));
  }

  return;
}

//
//
//
stk::mesh::EntityId
Topology::get_highest_id(stk::mesh::EntityRank const rank)
{
  return highest_ids_[rank];
}

}  // namespace LCM
