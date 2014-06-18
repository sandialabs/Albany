//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <boost/foreach.hpp>

#include "Subgraph.h"
#include "Topology.h"
#include "Topology_Utils.h"

namespace LCM {

//
// Default constructor
//
Topology::Topology() :
    discretization_(Teuchos::null),
    stk_mesh_struct_(Teuchos::null),
    fracture_criterion_(Teuchos::null)
{
  return;
}

//
// Constructor with input and output files.
//
Topology::Topology(
    std::string const & input_file,
    std::string const & output_file) :
    discretization_(Teuchos::null),
    stk_mesh_struct_(Teuchos::null),
    fracture_criterion_(Teuchos::null)
{
  RCP<Teuchos::ParameterList>
  params = Teuchos::rcp(new Teuchos::ParameterList("params"));

  // Create discretization object
  RCP<Teuchos::ParameterList>
  disc_params = Teuchos::sublist(params, "Discretization");

  // Set Method to Exodus and set input file name
  disc_params->set<std::string>("Method", "Exodus");
  disc_params->set<std::string>("Exodus Input File Name", input_file);
  disc_params->set<std::string>("Exodus Output File Name", output_file);

  RCP<Teuchos::ParameterList>
  problem_params = Teuchos::sublist(params, "Problem");

  RCP<Teuchos::ParameterList>
  adapt_params = Teuchos::sublist(problem_params, "Adaptation");

  RCP<Epetra_Comm>
  communicator = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

  Albany::DiscretizationFactory
  disc_factory(params, communicator);

  // Needed, otherwise segfaults.
  Teuchos::ArrayRCP<RCP<Albany::MeshSpecsStruct> >
  mesh_specs = disc_factory.createMeshSpecs();

  RCP<Albany::StateInfoStruct>
  state_info = Teuchos::rcp(new Albany::StateInfoStruct());

  // The default fields
  Albany::AbstractFieldContainer::FieldContainerRequirements
  req;

  setDiscretization(disc_factory.createDiscretization(3, state_info, req));

  Topology::createDiscretization();

  // Fracture the mesh randomly
  // Probability that fracture_criterion will return true.
  double const
  probability = 0.01;

  setFractureCriterion(
      Teuchos::rcp(new FractureCriterionRandom(probability))
  );

  // Create the full mesh representation. This must be done prior to
  // the adaptation query. We are reading the mesh from a file so do
  // it here.
  Topology::graphInitialization();

  return;
}

//
// This constructor assumes that the full mesh graph has been
// previously created. Topology::graphInitialization();
//
Topology::
Topology(RCP<Albany::AbstractDiscretization> & discretization) :
  discretization_(Teuchos::null),
  stk_mesh_struct_(Teuchos::null),
  fracture_criterion_(Teuchos::null)
{
  setDiscretization(discretization);

  Topology::createDiscretization();

  // Fracture the mesh randomly
  // Probability that fracture_criterion will return true.
  double const
  probability = 0.1;

  setFractureCriterion(
      Teuchos::rcp(new FractureCriterionRandom(probability))
  );

  return;
}

//
// Initialize fracture state field
// It exists for all entities except cells (elements)
//
void
Topology::initializeFractureState()
{
  stk_classic::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  for (EntityRank rank = NODE_RANK; rank < getCellRank(); ++rank) {

    std::vector<Bucket*> const &
    buckets = getBulkData()->buckets(rank);

    EntityVector
    entities;

    stk_classic::mesh::get_selected_entities(local_selector, buckets, entities);

    for (EntityVector::size_type i = 0; i < entities.size(); ++i) {

      Entity const &
      entity = *(entities[i]);

      setFractureState(entity, CLOSED);

    }
  }

  return;
}

//
// Create Albany discretization
//
void
Topology::createDiscretization()
{
  // Need to access the bulk_data and meta_data classes in the mesh
  // data structure
  STKDiscretization &
  stk_discretization = static_cast<STKDiscretization &>(*getDiscretization());

  setSTKMeshStruct(stk_discretization.getSTKMeshStruct());

  // Get the topology of the elements. NOTE: Assumes one element
  // type in mesh.
  stk_classic::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(getCellRank());

  EntityVector
  cells;

  stk_classic::mesh::get_selected_entities(local_selector, buckets, cells);

  Entity const &
  first_cell = *(cells[0]);

  setCellTopology(stk_classic::mesh::fem::get_cell_topology(first_cell));

  return;
}

//
// Initializes the default stk mesh object needed by class.
//
void Topology::graphInitialization()
{
  stk_classic::mesh::PartVector add_parts;
  stk_classic::mesh::create_adjacent_entities(*(getBulkData()), add_parts);

  getBulkData()->modification_begin();

  removeMultiLevelRelations();
  initializeFractureState();

  getBulkData()->modification_end();

  return;
}

//
// Creates temporary nodal connectivity for the elements and removes
// the relationships between the elements and nodes.
//
void Topology::removeNodeRelations()
{
  // Create the nodesorary connectivity array
  EntityVector
  elements;

  stk_classic::mesh::get_entities(*(getBulkData()), getCellRank(), elements);

  getBulkData()->modification_begin();

  for (size_t i = 0; i < elements.size(); ++i) {
    PairIterRelation
    relations = elements[i]->relations(NODE_RANK);

    EntityVector
    nodes;

    for (size_t j = 0; j < relations.size(); ++j) {
      Entity *
      node = relations[j].entity();
      nodes.push_back(node);
    }
    connectivity_.push_back(nodes);

    for (size_t j = 0; j < nodes.size(); ++j) {
      getBulkData()->destroy_relation(*(elements[i]), *(nodes[j]), j);
    }
  }

  getBulkData()->modification_end();

  return;
}

//
// Removes multilevel relations.
//
void Topology::removeMultiLevelRelations()
{
  typedef std::vector<EdgeId> EdgeIdList;

  size_t const
  cell_node_rank_distance = getCellRank() - NODE_RANK;

  // Go from points to cells
  for (EntityRank rank = NODE_RANK; rank <= getCellRank(); ++rank) {

    EntityVector
    entities;

    stk_classic::mesh::get_entities(*(getBulkData()), rank, entities);

    for (size_t i = 0; i < entities.size(); ++i) {

      Entity &
      entity = *(entities[i]);

      PairIterRelation
      relations = entity.relations();

      EntityVector
      far_entities;

      EdgeIdList
      multilevel_relation_ids;

      // Collect relations to delete
      for (PairIterRelation::iterator relation_iter = relations.begin();
          relation_iter != relations.end(); ++relation_iter) {

        Relation const &
        relation = *relation_iter;

        EntityRank
        target_rank = relation.entity_rank();

        size_t const
        rank_distance =
            rank > target_rank ? rank - target_rank : target_rank - rank;

        bool const
        is_valid_relation =
            rank < target_rank ||
            rank_distance == 1 ||
            rank_distance == cell_node_rank_distance;

        if (is_valid_relation == false) {
          far_entities.push_back(relation_iter->entity());
          multilevel_relation_ids.push_back(relation_iter->identifier());
        }

      }

      // Delete them
      for (size_t i = 0; i < multilevel_relation_ids.size(); ++i) {

        Entity &
        far_entity = *(far_entities[i]);

        EdgeId const
        multilevel_relation_id = multilevel_relation_ids[i];

        getBulkData()->destroy_relation(
            entity,
            far_entity,
            multilevel_relation_id);
      }

    }

  }

  return;
}

//
// After mesh manipulations are complete, need to recreate a stk
// mesh understood by Albany_STKDiscretization.
//
void Topology::restoreElementToNodeConnectivity()
{
  EntityVector
  elements;

  stk_classic::mesh::get_entities(*(getBulkData()), getCellRank(), elements);

  getBulkData()->modification_begin();

  // Add relations from element to nodes
  for (size_t i = 0; i < elements.size(); ++i) {
    Entity &
    element = *(elements[i]);

    EntityVector
    element_connectivity = connectivity_[i];

    for (size_t j = 0; j < element_connectivity.size(); ++j) {
      Entity &
      node = *(element_connectivity[j]);
      getBulkData()->declare_relation(element, node, j);
    }
  }

  // Recreate Albany STK Discretization
  STKDiscretization &
  stk_discretization = static_cast<STKDiscretization &>(*discretization_);

  RCP<Epetra_Comm>
  communicator = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

  //stk_discretization.updateMesh(stkMeshStruct_, communicator);
  stk_discretization.updateMesh();

  getBulkData()->modification_end();

  return;
}

//
// Returns array of pointers to Entities for the element to node relations
//
std::vector<EntityVector>
Topology::getElementToNodeConnectivity()
{
  // Create a list of element entities
  EntityVector
  element_list;

  EntityVector
  node_list;

  stk_classic::mesh::get_entities(*(getBulkData()), getCellRank(), element_list);

  // vector to store the entity pointers
  std::vector<EntityVector>
  element_to_node_connectivity;

  // Loop over the elements
  EntityVector::size_type const
  number_of_elements = element_list.size();

  for (EntityVector::size_type i(0); i < number_of_elements; ++i) {

    PairIterRelation
    relations = element_list[i]->relations(NODE_RANK);

    size_t const
    nodes_per_element = relations.size();

    for (size_t j(0); j < nodes_per_element; ++j) {

      Entity*
      node = relations[j].entity();

      node_list.push_back(node);
    }

    element_to_node_connectivity.push_back(node_list);
  }

  return element_to_node_connectivity;
}

//
// Determine nodes associated with a boundary entity
//
EntityVector
Topology::getBoundaryEntityNodes(Entity const & boundary_entity)
{
  EntityRank const
  boundary_rank = boundary_entity.entity_rank();

  assert(boundary_rank == getCellRank() - 1);

  EntityVector
  nodes;

  PairIterRelation
  relations = relations_one_up(boundary_entity);

  Entity const &
  first_cell = *(relations[0].entity());

  EdgeId const
  face_order = relations[0].identifier();

  size_t const
  number_face_nodes =
      getCellTopology().getNodeCount(boundary_rank, face_order);

  for (size_t i = 0; i < number_face_nodes; ++i) {
    EdgeId const
    cell_order = getCellTopology().getNodeMap(boundary_rank, face_order, i);

    // Brute force approach. Maybe there is a better way to do this?
    PairIterRelation
    node_relations = first_cell.relations(NODE_RANK);

    for (size_t i = 0; i < node_relations.size(); ++i) {
      if (node_relations[i].identifier() == cell_order) {
        nodes.push_back(node_relations[i].entity());
      }
    }
  }

  return nodes;
}

//
// Create boundary mesh
//
void
Topology::createBoundary()
{
  stk_classic::mesh::Part &
  boundary_part = *(getMetaData()->get_part("boundary"));

  stk_classic::mesh::PartVector
  add_parts;

  add_parts.push_back(&boundary_part);

  stk_classic::mesh::PartVector const
  part_vector = getMetaData()->get_parts();

  for (size_t i = 0; i < part_vector.size(); ++i) {
    std::cout << part_vector[i]->name() << '\n';
  }

  EntityRank const
  boundary_entity_rank = getCellRank() - 1;

  stk_classic::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(boundary_entity_rank);

  EntityVector
  entities;

  stk_classic::mesh::get_selected_entities(local_selector, buckets, entities);

  getBulkData()->modification_begin();
  for (EntityVector::size_type i = 0; i < entities.size(); ++i) {

    Entity &
    entity = *(entities[i]);

    PairIterRelation
    relations = relations_one_up(entity);

    size_t const
    number_connected_cells = std::distance(relations.begin(), relations.end());

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
      getBulkData()->change_entity_parts(entity, add_parts);
      break;

    case 2:
      // Internal face, do nothing.
      break;

    }

  }
  getBulkData()->modification_end();

  return;
}

//
// Get nodal coordinates
//
std::vector<Intrepid::Vector<double> >
Topology::getNodalCoordinates()
{
  stk_classic::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(NODE_RANK);

  EntityVector
  entities;

  stk_classic::mesh::get_selected_entities(local_selector, buckets, entities);

  EntityVector::size_type const
  number_nodes = entities.size();

  std::vector<Intrepid::Vector<double> >
  coordinates(number_nodes);

  size_t const
  dimension = getSpaceDimension();

  Intrepid::Vector<double>
  X(dimension);

  VectorFieldType &
  node_coordinates = *(getSTKMeshStruct()->getCoordinatesField());

  for (EntityVector::size_type i = 0; i < number_nodes; ++i) {

    Entity const &
    node = *(entities[i]);

    double const * const
    pointer_coordinates = stk_classic::mesh::field_data(node_coordinates, node);

    for (size_t j = 0; j < dimension; ++j) {
      X(j) = pointer_coordinates[j];
    }

    coordinates[i] = X;
  }

  return coordinates;
}

//
// Output of boundary
//
void
Topology::outputBoundary(std::string const & output_filename)
{
  // Open output file
  std::ofstream
  ofs;

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
  std::vector<Intrepid::Vector<double> > const
  coordinates = getNodalCoordinates();

  size_t const
  number_nodes = coordinates.size();

  ofs << "POINTS " << number_nodes << " double\n";

  for (size_t i = 0; i < number_nodes; ++i) {
    Intrepid::Vector<double> const &
    X = coordinates[i];

    for (size_t j = 0; j < X.get_dimension(); ++j) {
      ofs << std::setw(24) << std::scientific << std::setprecision(16) << X(j);
    }
    ofs << '\n';
  }

  std::vector<std::vector<EntityId> > const
  connectivity = getBoundary();

  size_t const
  number_cells = connectivity.size();

  size_t
  cell_list_size = 0;

  for (size_t i = 0; i < number_cells; ++i) {
    cell_list_size += connectivity[i].size() + 1;
  }

  // Boundary cell connectivity
  ofs << "CELLS " << number_cells << " " << cell_list_size << '\n';
  for (size_t i = 0; i < number_cells; ++i) {
    size_t const
    number_cell_nodes = connectivity[i].size();

    ofs << number_cell_nodes;

    for (size_t j = 0; j < number_cell_nodes; ++j) {
      ofs << ' ' << connectivity[i][j] - 1;
    }
    ofs << '\n';
  }

  ofs << "CELL_TYPES " << number_cells << '\n';
  for (size_t i = 0; i < number_cells; ++i) {
    size_t const
    number_cell_nodes = connectivity[i].size();

    VTKCellType
    cell_type = INVALID;

    switch (number_cell_nodes) {
    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Invalid number of nodes in boundary cell: ";
      std::cerr << number_cell_nodes;
      std::cerr << '\n';
      exit(1);
      break;

    case 1:
      cell_type = VERTEX;
      break;

    case 2:
      cell_type = LINE;
      break;

    case 3:
      cell_type = TRIANGLE;
      break;

    case 4:
      cell_type = QUAD;
      break;

    }
    ofs << cell_type << '\n';
  }

  ofs.close();
  return;
}

//
// Create boundary mesh
//
std::vector<std::vector<EntityId> >
Topology::getBoundary()
{
  EntityRank const
  boundary_entity_rank = getCellRank() - 1;

  stk_classic::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(boundary_entity_rank);

  EntityVector
  entities;

  stk_classic::mesh::get_selected_entities(local_selector, buckets, entities);

  std::vector<std::vector<EntityId> >
  connectivity;

  EntityVector::size_type const
  number_entities = entities.size();

  for (EntityVector::size_type i = 0; i < number_entities; ++i) {

    Entity &
    entity = *(entities[i]);

    PairIterRelation
    relations = relations_one_up(entity);

    size_t const
    number_connected_cells = std::distance(relations.begin(), relations.end());

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
      {
        EntityVector const
        nodes = getBoundaryEntityNodes(entity);

        EntityVector::size_type const
        number_nodes = nodes.size();

        std::vector<EntityId>
        node_ids(number_nodes);

        for (EntityVector::size_type i = 0; i < number_nodes; ++i) {
          node_ids[i] = nodes[i]->identifier();
        }
        connectivity.push_back(node_ids);
      }
      break;

    case 2:
      // Internal face, do nothing.
      break;

    }

  }

  return connectivity;
}

//
// Create cohesive connectivity
//
EntityVector
Topology::createSurfaceElementConnectivity(
    Entity const & face_top,
    Entity const & face_bottom)
{
  EntityVector
  top = getBoundaryEntityNodes(face_top);

  EntityVector
  bottom = getBoundaryEntityNodes(face_bottom);

  EntityVector
  both;

  both.reserve(top.size() + bottom.size());

  both.insert(both.end(), top.begin(), top.end());
  both.insert(both.end(), bottom.begin(), bottom.end());

  return both;
}

//
// Create vectors describing the vertices and edges of the star of
// an entity in the stk mesh.
//
void
Topology::createStar(
    Entity & entity,
    std::set<EntityKey> & subgraph_entities,
    std::set<stkEdge, EdgeLessThan> & subgraph_edges)
{
  subgraph_entities.insert(entity.key());

  PairIterRelation
  relations = relations_one_up(entity);

  for (PairIterRelation::iterator i = relations.begin();
      i != relations.end(); ++i) {

    Relation
    relation = *i;

    Entity &
    source = *(relation.entity());

    stkEdge
    edge;

    edge.source = source.key();
    edge.target = entity.key();
    edge.local_id = relation.identifier();

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
  assert(getSpaceDimension() == VOLUME_RANK);

  EntityVector
  points;

  EntityVector
  open_points;

  stk_classic::mesh::Selector
  selector_owned = getMetaData()->locally_owned_part();

  std::set<EntityPair>
  fractured_faces;

  stk_classic::mesh::get_selected_entities(
      selector_owned,
      getBulkData()->buckets(NODE_RANK),
      points);

  // Collect open points
  for (EntityVector::iterator i = points.begin();i != points.end(); ++i) {

    Entity &
    point = *(*i);

    if (getFractureState(point) == OPEN) {
      open_points.push_back(&point);
    }
  }

  getBulkData()->modification_begin();

  // Iterate over open points and fracture them.
  for (EntityVector::iterator i = open_points.begin();
      i != open_points.end(); ++i) {

    Entity &
    point = *(*i);

    PairIterRelation
    relations = relations_one_up(point);

    EntityVector
    open_segments;

    // Collect open segments.
    for (PairIterRelation::iterator j = relations.begin();
        j != relations.end(); ++j) {

      Entity &
      segment = *j->entity();

      bool const
      is_local_and_open_segment =
          isLocalEntity(segment) == true && getFractureState(segment) == OPEN;

      if (is_local_and_open_segment == true) {
        open_segments.push_back(&segment);
      }

    }

#if defined(LCM_GRAPHVIZ)
    {
      std::string const
      file_name = "graph-pre-segment-" + entity_string(point) + ".dot";
      outputToGraphviz(file_name);
    }
#endif

    // Iterate over open segments and fracture them.
    for (EntityVector::iterator j = open_segments.begin();
        j != open_segments.end(); ++j) {

      Entity &
      segment = *(*j);

      // Create star of segment
      std::set<EntityKey>
      subgraph_entities;

      std::set<stkEdge, EdgeLessThan>
      subgraph_edges;

      createStar(segment, subgraph_entities, subgraph_edges);

      // Iterators
      std::set<EntityKey>::iterator
      first_entity = subgraph_entities.begin();

      std::set<EntityKey>::iterator
      last_entity = subgraph_entities.end();

      std::set<stkEdge>::iterator
      first_edge = subgraph_edges.begin();

      std::set<stkEdge>::iterator
      last_edge = subgraph_edges.end();

      Subgraph
      subgraph(getSTKMeshStruct(),
          first_entity, last_entity, first_edge, last_edge);

#if defined(LCM_GRAPHVIZ)
      {
        std::string const
        file_name = "graph-pre-clone-" + entity_string(segment) + ".dot";
        outputToGraphviz(file_name);
        subgraph.outputToGraphviz("sub" + file_name);
      }
#endif

      // Collect open faces
      PairIterRelation
      face_relations = relations_one_up(segment);

      EntityVector
      open_faces;

      for (PairIterRelation::iterator k = face_relations.begin();
          k != face_relations.end(); ++k) {

        Entity *
        face = k->entity();

        bool const
        is_local_and_open_face =
            isLocalEntity(*face) == true && isInternalAndOpen(*face) == true;

        if (is_local_and_open_face == true) {
          open_faces.push_back(face);
        }
      }

      // Iterate over the open faces
      for (EntityVector::iterator k = open_faces.begin();
          k != open_faces.end(); ++k) {

        Entity *
        face = *k;

        Vertex
        face_vertex = subgraph.globalToLocal(face->key());

        Vertex
        new_face_vertex = subgraph.cloneBoundaryEntity(face_vertex);

        EntityKey
        new_face_key = subgraph.localToGlobal(new_face_vertex);

        Entity *
        new_face = getBulkData()->get_entity(new_face_key);

        // Reset fracture state for both old and new faces
        setFractureState(*face, CLOSED);
        setFractureState(*new_face, CLOSED);

        EntityPair
        ff = std::make_pair(face, new_face);

        fractured_faces.insert(ff);
      }

      // Split the articulation point (current segment)
      Vertex
      segment_vertex = subgraph.globalToLocal(segment.key());

#if defined(LCM_GRAPHVIZ)
      {
        std::string const
        file_name = "graph-pre-split-" + entity_string(segment) + ".dot";
        outputToGraphviz(file_name);
        subgraph.outputToGraphviz("sub" + file_name);
      }
#endif

      subgraph.splitArticulationPoint(segment_vertex);

      // Reset segment fracture state
      setFractureState(segment, CLOSED);

#if defined(LCM_GRAPHVIZ)
      {
        std::string const
        file_name = "graph-post-split-" + entity_string(segment) + ".dot";
        outputToGraphviz(file_name);
        subgraph.outputToGraphviz("sub" + file_name);
      }
#endif
    }

    // All open faces and segments have been dealt with.
    // Split the node articulation point
    // Create star of node
    std::set<EntityKey>
    subgraph_entities;

    std::set<stkEdge, EdgeLessThan>
    subgraph_edges;

    createStar(point, subgraph_entities, subgraph_edges);

    // Iterators
    std::set<EntityKey>::iterator
    first_entity = subgraph_entities.begin();

    std::set<EntityKey>::iterator
    last_entity = subgraph_entities.end();

    std::set<stkEdge>::iterator
    first_edge = subgraph_edges.begin();

    std::set<stkEdge>::iterator
    last_edge = subgraph_edges.end();

    Subgraph
    subgraph(
        getSTKMeshStruct(),
        first_entity,
        last_entity,
        first_edge,
        last_edge);

    Vertex
    node = subgraph.globalToLocal(point.key());

#if defined(LCM_GRAPHVIZ)
    {
      std::string const
      file_name = "graph-pre-split-" + entity_string(point) + ".dot";
      outputToGraphviz(file_name);
      subgraph.outputToGraphviz("sub" + file_name);
    }
#endif

    ElementNodeMap
    new_connectivity = subgraph.splitArticulationPoint(node);

    // Reset fracture state of point
    setFractureState(point, CLOSED);

#if defined(LCM_GRAPHVIZ)
    {
      std::string const
      file_name = "graph-post-split-" + entity_string(point) + ".dot";
      outputToGraphviz(file_name);
      subgraph.outputToGraphviz("sub" + file_name);
    }
#endif

    // Update the connectivity
    for (ElementNodeMap::iterator j = new_connectivity.begin();
        j != new_connectivity.end(); ++j) {

      Entity &
      new_node = *((*j).second);

      getBulkData()->copy_entity_fields(point, new_node);
    }

  }

  getBulkData()->modification_end();

  getBulkData()->modification_begin();

  // Create the cohesive connectivity
  for (std::set<EntityPair>::iterator i =
      fractured_faces.begin(); i != fractured_faces.end(); ++i) {

    Entity & face1 = *((*i).first);
    Entity & face2 = *((*i).second);

    EntityVector
    cohesive_connectivity = createSurfaceElementConnectivity(face1, face2);

    // TODO: Insert the surface element element

  }

  getBulkData()->modification_end();
  return;
}

//
//
//
size_t
Topology::setEntitiesOpen()
{
  EntityVector
  boundary_entities;

  stk_classic::mesh::Selector
  select_owned = getMetaData()->locally_owned_part();

  stk_classic::mesh::get_selected_entities(
      select_owned,
      getBulkData()->buckets(getBoundaryRank()) ,
      boundary_entities);

  size_t
  counter = 0;

  // Iterate over the boundary entities
  for (size_t i = 0; i < boundary_entities.size(); ++i) {

    Entity &
    entity = *(boundary_entities[i]);

    if (isInternal(entity) == false) continue;

    if (checkOpen(entity) == false) continue;

    setFractureState(entity, OPEN);
    ++counter;

    switch(getCellRank()) {

    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Invalid cells rank in fracture: ";
      std::cerr << getCellRank();
      std::cerr << '\n';
      exit(1);
      break;

    case VOLUME_RANK:
      {
        PairIterRelation segments = entity.relations(EDGE_RANK);
        for (size_t j = 0; j < segments.size(); ++j) {
          Entity & segment = *(segments[j].entity());
          setFractureState(segment, OPEN);
          PairIterRelation points = segment.relations(NODE_RANK);
          for (size_t k = 0; k < points.size(); ++k) {
            Entity & point = *(points[k].entity());
            setFractureState(point, OPEN);
          }
        }
      }
      break;

    case EDGE_RANK:
      {
        PairIterRelation points = entity.relations(NODE_RANK);
        for (size_t j = 0; j < points.size(); ++j) {
          Entity & point = *(points[j].entity());
          setFractureState(point, OPEN);
        }
      }
      break;
    }

  }

  return counter;
}

//
// Output the graph associated with the mesh to graphviz .dot
// file for visualization purposes. No need for entity_open map
// for this version
//
void
Topology::outputToGraphviz(
    std::string const & output_filename,
    OutputType const output_type)
{
  // Open output file
  std::ofstream
  gviz_out;

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

  RelationList
  relation_list;

  std::vector<EdgeId>
  relation_local_id;

  // Entities (graph vertices)
  for (EntityRank rank = NODE_RANK; rank <= getCellRank(); ++rank) {

    EntityVector
    entities;

    stk_classic::mesh::get_entities(*(getBulkData()), rank, entities);

    for (EntityVector::size_type i = 0; i < entities.size(); ++i) {

      Entity &
      source_entity = *(entities[i]);

      FractureState const
      fracture_state = getFractureState(source_entity);

      PairIterRelation
      relations = relations_all(source_entity);

      gviz_out << dot_entity(source_entity.identifier(), rank, fracture_state);

      for (size_t j = 0; j < relations.size(); ++j) {

        Relation const &
        relation = relations[j];

        Entity &
        target_entity = *(relation.entity());

        EntityRank const
        target_rank = target_entity.entity_rank();

        bool
        is_valid_target_rank = false;

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

        EntityPair
        entity_pair = std::make_pair(&source_entity, &target_entity);

        EdgeId const
        edge_id = relation.identifier();

        relation_list.push_back(entity_pair);
        relation_local_id.push_back(edge_id);
      }

    }

  }

  // Relations (graph edges)
  for (RelationList::size_type i = 0; i < relation_list.size(); ++i) {

    EntityPair
    entity_pair = relation_list[i];

    Entity &
    source = *(entity_pair.first);

    Entity &
    target = *(entity_pair.second);

    gviz_out << dot_relation(
        source.identifier(),
        source.entity_rank(),
        target.identifier(),
        target.entity_rank(),
        relation_local_id[i]
    );

  }

  // File end
  gviz_out << dot_footer();

  gviz_out.close();

  return;
}

} // namespace LCM

