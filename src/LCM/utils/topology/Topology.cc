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
  probability = 0.1;

  setFractureCriterion(
      Teuchos::rcp(new FractureCriterionRandom(
          getSpaceDimension(), probability))
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
      Teuchos::rcp(new FractureCriterionRandom(
          getSpaceDimension(), probability))
  );

  return;
}

//
//
//
Topology::
Topology(RCP<Albany::AbstractDiscretization>& discretization,
    RCP<AbstractFractureCriterion>& fracture_criterion) :
    discretization_(Teuchos::null),
    stk_mesh_struct_(Teuchos::null),
    fracture_criterion_(Teuchos::null)
{
  setDiscretization(discretization);
  setFractureCriterion(fracture_criterion);
  Topology::createDiscretization();

  return;
}

//
// Initialize fracture state field
// It exists for all entities except cells (elements)
//
void
Topology::initializeFractureState()
{
  stk::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  for (EntityRank rank = NODE_RANK; rank < getCellRank(); ++rank) {

    std::vector<Bucket*> const &
    buckets = getBulkData()->buckets(rank);

    EntityVector
    entities;

    stk::mesh::get_selected_entities(local_selector, buckets, entities);

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
  stk::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(getCellRank());

  EntityVector
  cells;

  stk::mesh::get_selected_entities(local_selector, buckets, cells);

  Entity const &
  first_cell = *(cells[0]);

  setCellTopology(stk::mesh::fem::get_cell_topology(first_cell));

  return;
}

//
// Initializes the default stk mesh object needed by class.
//
void Topology::graphInitialization()
{
  stk::mesh::PartVector add_parts;
  stk::mesh::create_adjacent_entities(*(getBulkData()), add_parts);

  getBulkData()->modification_begin();

  removeMultiLevelRelations();
  initializeFractureState();

  getBulkData()->modification_end();

  return;
}

//
// Removes multilevel relations.
//
void Topology::removeMultiLevelRelations()
{
  typedef std::vector<EdgeId> EdgeIdList;

  // Go from cells to segments
  for (EntityRank rank = getCellRank(); rank > NODE_RANK; --rank) {

    EntityVector
    entities;

    stk::mesh::get_entities(*(getBulkData()), rank, entities);

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

        bool const
        is_valid_relation =
            rank - target_rank == 1 ||
            (rank == getCellRank() && target_rank == NODE_RANK);

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

//----------------------------------------------------------------------------
//
// Removes unneeded relations from the mesh.
//
void Topology::removeExtraRelations()
{
  EntityVector element_list;
  stk::mesh::get_entities(*(getBulkData()), getCellRank(), element_list);

  // Remove extra relations from element
  for (int i = 0; i < element_list.size(); ++i) {
    Entity & element = *(element_list[i]);
    PairIterRelation relations = element.relations();
    EntityVector del_relations;
    std::vector<int> del_ids;
    for (PairIterRelation::iterator j = relations.begin();
        j != relations.end(); ++j) {
      // remove all relationships from element unless to faces(segments
      //   in 2D) or nodes
      if (j->entity_rank() != getCellRank() - 1
          && j->entity_rank() != NODE_RANK) {
        del_relations.push_back(j->entity());
        del_ids.push_back(j->identifier());
      }
    }
    for (int j = 0; j < del_relations.size(); ++j) {
      Entity & entity = *(del_relations[j]);
      getBulkData()->destroy_relation(element, entity, del_ids[j]);
    }
  };

  if (getCellRank() == VOLUME_RANK) {
    // Remove extra relations from face
    EntityVector face_list;
    stk::mesh::get_entities(*(getBulkData()), getCellRank() - 1, face_list);
    EntityRank entityRank = face_list[0]->entity_rank();
    for (int i = 0; i < face_list.size(); ++i) {
      Entity & face = *(face_list[i]);
      PairIterRelation relations = face_list[i]->relations();
      EntityVector del_relations;
      std::vector<int> del_ids;
      for (PairIterRelation::iterator j = relations.begin();
          j != relations.end(); ++j) {
        if (j->entity_rank() != entityRank + 1
            && j->entity_rank() != entityRank - 1) {
          del_relations.push_back(j->entity());
          del_ids.push_back(j->identifier());
        }
      }
      for (int j = 0; j < del_relations.size(); ++j) {
        Entity & entity = *(del_relations[j]);
        getBulkData()->destroy_relation(face, entity, del_ids[j]);
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------
//
// Creates temporary nodal connectivity for the elements and removes
// the relationships between the elements and nodes.
//
void Topology::removeNodeRelations()
{
  // Create the temporary connectivity array
  EntityVector element_list;
  stk::mesh::get_entities(*(getBulkData()), getCellRank(), element_list);

  getBulkData()->modification_begin();
  for (int i = 0; i < element_list.size(); ++i) {
    PairIterRelation nodes = element_list[i]->relations(NODE_RANK);
    EntityVector temp;
    for (int j = 0; j < nodes.size(); ++j) {
      Entity* node = nodes[j].entity();
      temp.push_back(node);
    }
    connectivity_temp_.push_back(temp);

    for (int j = 0; j < temp.size(); ++j) {
      getBulkData()->destroy_relation(*(element_list[i]), *(temp[j]), j);
    }
  }

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

  stk::mesh::get_entities(*(getBulkData()), getCellRank(), element_list);

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

//----------------------------------------------------------------------------
void
Topology::
removeElementToNodeConnectivity(std::vector<EntityVector>& oldElemToNode)
{
  // Create the temporary connectivity array
  EntityVector element_list;
  stk::mesh::get_entities(*(getBulkData()), getCellRank(), element_list);

  getBulkData()->modification_begin();
  for (int i = 0; i < element_list.size(); ++i) {
    PairIterRelation nodes = element_list[i]->relations(NODE_RANK);
    EntityVector temp;
    for (int j = 0; j < nodes.size(); ++j) {
      Entity* node = nodes[j].entity();
      temp.push_back(node);
    }

    // save the current element to node connectivity and the local
    // to global numbering
    connectivity_temp_.push_back(temp);
    element_global_to_local_ids_[element_list[i]->identifier()] = i;

    for (int j = 0; j < temp.size(); ++j) {
      getBulkData()->destroy_relation(*(element_list[i]), *(temp[j]), j);
    }
  }

  getBulkData()->modification_end();

  return;
}

//----------------------------------------------------------------------------
//
// After mesh manipulations are complete, need to recreate a stk
// mesh understood by Albany_STKDiscretization.
void Topology::restoreElementToNodeConnectivity()
{
  EntityVector element_list;
  stk::mesh::get_entities(*(getBulkData()), getCellRank(), element_list);

  getBulkData()->modification_begin();

  // Add relations from element to nodes
  for (int i = 0; i < element_list.size(); ++i) {
    Entity & element = *(element_list[i]);
    EntityVector element_connectivity = connectivity_temp_[i];
    for (int j = 0; j < element_connectivity.size(); ++j) {
      Entity & node = *(element_connectivity[j]);
      getBulkData()->declare_relation(element, node, j);
    }
  }

  // Recreate Albany STK Discretization
  STKDiscretization & stk_discretization =
      static_cast<STKDiscretization &>(*discretization_);

  RCP<Epetra_Comm> communicator =
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

  //stk_discretization.updateMesh(stkMeshStruct_, communicator);
  stk_discretization.updateMesh();

  getBulkData()->modification_end();

  return;
}

//----------------------------------------------------------------------------
void
Topology::
restoreElementToNodeConnectivity(std::vector<EntityVector>& oldElemToNode)
{
  EntityVector element_list;
  stk::mesh::get_entities(*(getBulkData()), getCellRank(), element_list);

  //    getBulkData()->modification_begin(); // need to comment GAH?

  // Add relations from element to nodes
  for (int i = 0; i < element_list.size(); ++i) {
    Entity & element = *(element_list[i]);
    EntityVector element_connectivity = oldElemToNode[i];
    for (int j = 0; j < element_connectivity.size(); ++j) {
      Entity & node = *(element_connectivity[j]);
      getBulkData()->declare_relation(element, node, j);
    }
  }

  getBulkData()->modification_end();

  return;
}

//----------------------------------------------------------------------------
//
// Determine the nodes associated with a face.
//
EntityVector Topology::getFaceNodes(Entity * entity)
{
  EntityVector face_nodes;

  PairIterRelation elements = entity->relations(getCellRank());
  // local id for the current face
  unsigned faceId = elements[0].identifier();
  Entity * element = elements[0].entity();
  // number of nodes for the face
  unsigned numFaceNodes = getCellTopology().getNodeCount(entity->entity_rank(),
      faceId);

  // Create the ordered list of nodes for the face
  for (int i = 0; i < numFaceNodes; ++i) {
    // map the local node id for the face to the local node id for the element
    unsigned elem_node = getCellTopology().getNodeMap(entity->entity_rank(),
        faceId, i);
    // map the local element node id to the global node id
    int element_local_id = element_global_to_local_ids_[element->identifier()];
    Entity* node = connectivity_temp_[element_local_id][elem_node];
    face_nodes.push_back(node);
  }

  return face_nodes;
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
  stk::mesh::Part &
  boundary_part = *(getMetaData()->get_part("boundary"));

  stk::mesh::PartVector
  add_parts;

  add_parts.push_back(&boundary_part);

  stk::mesh::PartVector const
  part_vector = getMetaData()->get_parts();

  for (size_t i = 0; i < part_vector.size(); ++i) {
    std::cout << part_vector[i]->name() << '\n';
  }

  EntityRank const
  boundary_entity_rank = getCellRank() - 1;

  stk::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(boundary_entity_rank);

  EntityVector
  entities;

  stk::mesh::get_selected_entities(local_selector, buckets, entities);

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
// Output of boundary
//
void
Topology::outputBoundary()
{
  EntityRank const
  boundary_entity_rank = getCellRank() - 1;

  stk::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(boundary_entity_rank);

  EntityVector
  entities;

  stk::mesh::get_selected_entities(local_selector, buckets, entities);

  for (EntityVector::size_type i = 0; i < entities.size(); ++i) {

    Entity const &
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
        face_nodes = getBoundaryEntityNodes(entity);
        std::cout << entity.identifier() << " ";
        for (EntityVector::size_type i = 0; i < face_nodes.size(); ++i) {
          std::cout << face_nodes[i]->identifier() << " ";
        }
        std::cout << '\n';
      }
      break;

    case 2:
      // Internal face, do nothing.
      break;

    }

  }

  return;
}

//
// Create cohesive connectivity
// bcell: boundary cell
//
EntityVector
Topology::createSurfaceElementConnectivity(Entity const & bcell1,
    Entity const & bcell2)
{
  // number of nodes for the face
  size_t
  number_face_nodes = getCellTopology().getNodeCount(bcell1.entity_rank(), 0);

  // Traverse down the graph from the face. The first node of
  // segment $n$ is node $n$ of the face.
  PairIterRelation
  bcell1_relations = relations_one_down(bcell1);

  PairIterRelation
  bcell2_relations = relations_one_down(bcell2);

  EntityVector
  connectivity(2 * number_face_nodes);

  for (size_t i = 0; i < bcell1_relations.size(); ++i) {

    Entity &
    entity1 = *(bcell1_relations[i].entity());

    Entity &
    entity2 = *(bcell2_relations[i].entity());

    EntityRank const
    cell_rank = getCellRank();

    switch (getCellRank()) {
    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Surface element not implemented for dimension: ";
      std::cerr << cell_rank;
      std::cerr << '\n';
      exit(1);
      break;

    case FACE_RANK:
      connectivity[i] = &entity1;
      connectivity[i + number_face_nodes] = &entity2;
      break;

    case VOLUME_RANK:
    {
      PairIterRelation
      segment1_relations = entity1.relations(entity1.entity_rank() - 1);
      PairIterRelation
      segment2_relations = entity2.relations(entity2.entity_rank() - 1);

      // Check for the correct node to add to the connectivity vector.
      // Each node should be used only once.
      bool const
      unique_node =
          (i == 0) ||
          (i > 0 && connectivity[i - 1] != segment1_relations[0].entity()) ||
          (i == number_face_nodes - 1
              && connectivity[0] != segment1_relations[0].entity());

      if (unique_node == true) {
        connectivity[i] = segment1_relations[0].entity();
        connectivity[i + number_face_nodes] = segment2_relations[0].entity();
      } else {
        connectivity[i] = segment1_relations[1].entity();
        connectivity[i + number_face_nodes] = segment2_relations[1].entity();
      }
    }
    break;

    }

  }

  return connectivity;
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

  stk::mesh::Selector
  selector_owned = getMetaData()->locally_owned_part();

  std::set<EntityPair>
  fractured_faces;

  stk::mesh::get_selected_entities(
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
      is_local_and_open =
          isLocalEntity(segment) == true && getFractureState(segment) == OPEN;

      if (is_local_and_open == true) {
        open_segments.push_back(&segment);
      }

    }

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

      // Collect open faces
      PairIterRelation
      face_relations = relations_one_up(segment);

      EntityVector
      open_faces;

      for (PairIterRelation::iterator k = face_relations.begin();
          k != face_relations.end(); ++k) {

        Entity *
        face = k->entity();

        if (isInternalAndOpen(*face) == true) {
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
        new_face_vertex;
        subgraph.cloneBoundaryEntity(face_vertex);

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

      subgraph.splitArticulationPoint(segment_vertex);

      // Reset segment fracture state
      setFractureState(segment, CLOSED);

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

    std::map<Entity*, Entity*>
    new_connectivity = subgraph.splitArticulationPoint(node);

    // Reset fracture state of point
    setFractureState(point, CLOSED);

    // Update the connectivity
    for (std::map<Entity*, Entity*>::iterator j = new_connectivity.begin();
        j != new_connectivity.end(); ++j) {

      Entity *
      element = (*j).first;

      Entity *
      new_node = (*j).second;

      getBulkData()->copy_entity_fields(point, *new_node);
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

  }

  getBulkData()->modification_end();
  return;
}

//----------------------------------------------------------------------------
//
// Fractures all open boundary entities of the mesh.
//
#if 0  // original
void
Topology::splitOpenFaces(std::map<EntityKey, bool> & entity_open)
{
  int numfractured = 0; //counter for number of fractured faces

  // Get set of open nodes
  EntityVector node_list; //all nodes
  EntityVector open_node_list; //only the open nodes
  stk::mesh::Selector select_owned_or_shared = getMetaData()->locally_owned_part() |
      getMetaData()->globally_shared_part();

  stk::mesh::get_selected_entities( select_owned_or_shared,
      getBulkData()->buckets( NODE_RANK ),
      node_list );
  for (EntityVector::iterator i = node_list.begin();
      i != node_list.end(); ++i) {
    Entity* entity = *i;
    if (entity_open[entity->key()] == true) {
      open_node_list.push_back(entity);
    }
  }

  getBulkData()->modification_begin();

  // Iterate over the open nodes
  for (EntityVector::iterator i = open_node_list.begin();
      i != open_node_list.end(); ++i) {
    // Get set of open segments
    Entity * entity = *i;
    PairIterRelation relations = entity->relations(EDGE_RANK);
    EntityVector open_segment_list;

    for (PairIterRelation::iterator j = relations.begin();
        j != relations.end(); ++j) {
      Entity & source = *j->entity();
      if (entity_open[source.key()] == true) {
        open_segment_list.push_back(&source);
      }
    }

    // Iterate over the open segments
    for (EntityVector::iterator j = open_segment_list.begin();
        j != open_segment_list.end(); ++j) {
      Entity * segment = *j;
      // Create star of segment
      std::set<EntityKey> subgraph_entity_list;
      std::set<stkEdge, EdgeLessThan> subgraph_edge_list;
      Topology::createStar(*segment, subgraph_entity_list, subgraph_edge_list);
      // Iterators
      std::set<EntityKey>::iterator firstEntity = subgraph_entity_list.begin();
      std::set<EntityKey>::iterator lastEntity = subgraph_entity_list.end();
      std::set<stkEdge>::iterator firstEdge = subgraph_edge_list.begin();
      std::set<stkEdge>::iterator lastEdge = subgraph_edge_list.end();

      Subgraph subgraph(getSTKMeshStruct(), firstEntity, lastEntity, firstEdge,
          lastEdge);

      // Clone open faces
      PairIterRelation faces = segment->relations(FACE_RANK);
      EntityVector open_face_list;
      // create a list of open faces
      for (PairIterRelation::iterator k = faces.begin();
          k != faces.end(); ++k) {
        Entity & source = *k->entity();
        if (entity_open[source.key()] == true) {
          open_face_list.push_back(&source);
        }
      }

      // Iterate over the open faces
      for (EntityVector::iterator k = open_face_list.begin();
          k != open_face_list.end(); ++k) {
        Entity * face = *k;
        Vertex faceVertex = subgraph.globalToLocal(face->key());
        Vertex newFaceVertex;
        subgraph.cloneBoundaryEntity(faceVertex, newFaceVertex,
            entity_open);

        EntityKey newFaceKey = subgraph.localToGlobal(newFaceVertex);
        Entity * newFace = getBulkData()->get_entity(newFaceKey);

        // add original and new faces to the fractured face list
        fractured_faces_.insert(std::make_pair(face, newFace));

        ++numfractured;
      }

      // Split the articulation point (current segment)
      Vertex segmentVertex = subgraph.globalToLocal(segment->key());
      subgraph.splitArticulationPoint(segmentVertex, entity_open);
    }
    // All open faces and segments have been dealt with. Split the node articulation point
    // Create star of node
    std::set<EntityKey> subgraph_entity_list;
    std::set<stkEdge, EdgeLessThan> subgraph_edge_list;
    Topology::createStar(*entity, subgraph_entity_list, subgraph_edge_list);
    // Iterators
    std::set<EntityKey>::iterator firstEntity = subgraph_entity_list.begin();
    std::set<EntityKey>::iterator lastEntity = subgraph_entity_list.end();
    std::set<stkEdge>::iterator firstEdge = subgraph_edge_list.begin();
    std::set<stkEdge>::iterator lastEdge = subgraph_edge_list.end();
    Subgraph subgraph(getSTKMeshStruct(),
        firstEntity, lastEntity, firstEdge, lastEdge);

    Vertex node = subgraph.globalToLocal(entity->key());
    std::map<Entity*, Entity*> new_connectivity =
        subgraph.splitArticulationPoint(node, entity_open);

    // Update the connectivity
    for (std::map<Entity*, Entity*>::iterator j = new_connectivity.begin();
        j != new_connectivity.end(); ++j) {
      Entity* element = (*j).first;
      Entity* newNode = (*j).second;

      int element_id = element_global_to_local_ids_[element->identifier()];
      EntityVector & element_connectivity = connectivity_temp_[element_id];
      for (int k = 0; k < element_connectivity.size(); ++k) {
        // Need to subtract 1 from element number as stk indexes from 1
        //   and connectivity_temp indexes from 0
        if (element_connectivity[k] == entity) {
          element_connectivity[k] = newNode;
          // Duplicate the parameters of old node to new node
          getBulkData()->copy_entity_fields(*entity, *newNode);
        }
      }
    }
  }

  getBulkData()->modification_end();
  getBulkData()->modification_begin();

  // Create the cohesive connectivity
  int j = 1;
  for (std::set<EntityPair>::iterator i =
      fractured_faces_.begin(); i != fractured_faces_.end(); ++i, ++j) {
    Entity * face1 = (*i).first;
    Entity * face2 = (*i).second;
    EntityVector cohesive_connectivity;
    cohesive_connectivity =
        Topology::createSurfaceElementConnectivity(*face1, *face2);

    // Output connectivity for testing purposes
    std::cout << "Cohesive Element " << j << ": ";
    for (int j = 0; j < cohesive_connectivity.size(); ++j) {
      std::cout << cohesive_connectivity[j]->identifier() << ":";
    }
    std::cout << "\n";
  }

  getBulkData()->modification_end();
  return;
}
#endif

void Topology::splitOpenFaces(std::map<EntityKey, bool> & global_entity_open)
{
  EntityVector open_node_list; // Global open node list

  std::cout << " \n\nGlobal stuff in fracture_boundary\n\n" << '\n';

  // Build list of open nodes (global)

  std::pair<EntityKey,bool> me; // what a map<EntityKey, bool> is made of

  BOOST_FOREACH(me, global_entity_open) {

    if(stk::mesh::entity_rank( me.first) == NODE_RANK){

      Entity *entity = getBulkData()->get_entity(me.first);
      std::cout << "Found open node: " << entity->identifier() << " belonging to pe: " << entity->owner_rank() << '\n';
      open_node_list.push_back(entity);
    }
  }

  getBulkData()->modification_begin();

  // Iterate over the open nodes
  for (EntityVector::iterator i = open_node_list.begin();
      i != open_node_list.end(); ++i) {
    // Get set of open segments
    Entity * entity = *i;
    PairIterRelation relations = entity->relations(EDGE_RANK);
    EntityVector open_segment_list;

    for (PairIterRelation::iterator j = relations.begin();
        j != relations.end(); ++j) {
      Entity & source = *j->entity();
      if (global_entity_open[source.key()] == true) {
        std::cout << "Found open segment: " << source.identifier() << " belonging to pe: " << source.owner_rank() << '\n';
        open_segment_list.push_back(&source);
      }
    }

    // Iterate over the open segments
    for (EntityVector::iterator j = open_segment_list.begin();
        j != open_segment_list.end(); ++j) {
      Entity * segment = *j;

      // Create star of segment
      std::set<EntityKey> subgraph_entity_list;
      std::set<stkEdge, EdgeLessThan> subgraph_edge_list;
      Topology::createStar(*segment, subgraph_entity_list, subgraph_edge_list);

      // Iterators
      std::set<EntityKey>::iterator first_entity = subgraph_entity_list.begin();
      std::set<EntityKey>::iterator last_entity = subgraph_entity_list.end();
      std::set<stkEdge>::iterator first_edge = subgraph_edge_list.begin();
      std::set<stkEdge>::iterator last_edge = subgraph_edge_list.end();

      Subgraph subgraph(getSTKMeshStruct(),
          first_entity, last_entity, first_edge, last_edge);

      // Clone open faces
      PairIterRelation faces = segment->relations(FACE_RANK);
      EntityVector open_face_list;

      // create a list of open faces
      for (PairIterRelation::iterator k = faces.begin();
          k != faces.end(); ++k) {
        Entity & source = *k->entity();
        if (global_entity_open[source.key()] == true) {
          std::cout << "Found open face: " << source.identifier() << " belonging to pe: " << source.owner_rank() << '\n';
          open_face_list.push_back(&source);
        }
      }
      std::cout << "\n\n\n\n\n" << '\n';

      // Iterate over the open faces
      for (EntityVector::iterator k = open_face_list.begin();
          k != open_face_list.end(); ++k) {
        Entity * face = *k;
        Vertex face_vertex = subgraph.globalToLocal(face->key());
        Vertex new_face_vertex;
        subgraph.cloneBoundaryEntity(face_vertex, new_face_vertex,
            global_entity_open);
        EntityKey new_face_key = subgraph.localToGlobal(new_face_vertex);
        Entity * new_face = getBulkData()->get_entity(new_face_key);

        // add original and new faces to the fractured face list
        fractured_faces_.insert(std::make_pair(face, new_face));

      }

      // Split the articulation point (current segment)
      Vertex segment_vertex = subgraph.globalToLocal(segment->key());
      std::cout << "Calling split_articulation_point with segmentVertex: " << '\n';
      subgraph.splitArticulationPoint(segment_vertex, global_entity_open);
      std::cout << "done Calling split_articulation_point with segmentVertex: " << '\n';
    }
    // All open faces and segments have been dealt with. Split the node articulation point
    // Create star of node
    std::set<EntityKey> subgraph_entity_list;
    std::set<stkEdge, EdgeLessThan> subgraph_edge_list;
    Topology::createStar(*entity, subgraph_entity_list, subgraph_edge_list);
    // Iterators
    std::set<EntityKey>::iterator firstEntity = subgraph_entity_list.begin();
    std::set<EntityKey>::iterator lastEntity = subgraph_entity_list.end();
    std::set<stkEdge>::iterator firstEdge = subgraph_edge_list.begin();
    std::set<stkEdge>::iterator lastEdge = subgraph_edge_list.end();
    Subgraph subgraph(
        getSTKMeshStruct(),
        firstEntity, lastEntity, firstEdge, lastEdge);

    Vertex node = subgraph.globalToLocal(entity->key());
    std::cout << "Calling split_articulation_point with node: " << '\n';
    std::map<Entity*, Entity*> new_connectivity =
        subgraph.splitArticulationPoint(node, global_entity_open);
    std::cout << "done Calling split_articulation_point with node: " << '\n';

    // Update the connectivity
    for (std::map<Entity*, Entity*>::iterator j = new_connectivity.begin();
        j != new_connectivity.end(); ++j) {
      Entity* element = (*j).first;
      Entity* newNode = (*j).second;

      // Need to subtract 1 from element number as stk indexes from 1
      //   and connectivity_temp indexes from 0
      //        int id = static_cast<int>(element->identifier());
      int element_local_id = element_global_to_local_ids_[element->identifier()];
      //        EntityVector & element_connectivity = connectivity_temp_[id - 1];
      EntityVector & element_connectivity = connectivity_temp_[element_local_id];
      for (int k = 0; k < element_connectivity.size(); ++k) {
        if (element_connectivity[k] == entity) {
          element_connectivity[k] = newNode;
          // Duplicate the parameters of old node to new node
          getBulkData()->copy_entity_fields(*entity, *newNode);
        }
      }
    }
  }

  getBulkData()->modification_end();




  getBulkData()->modification_begin();

  // Create the cohesive connectivity
  int j = 1;
  for (std::set<EntityPair>::iterator i =
      fractured_faces_.begin(); i != fractured_faces_.end(); ++i, ++j) {
    Entity & face1 = *((*i).first);
    Entity & face2 = *((*i).second);
    EntityVector cohesive_connectivity;
    cohesive_connectivity =
        Topology::createSurfaceElementConnectivity(face1, face2);

    // Output connectivity for testing purposes
    std::cout << "Cohesive Element " << j << ": ";
    for (int j = 0; j < cohesive_connectivity.size(); ++j) {
      std::cout << cohesive_connectivity[j]->identifier() << ":";
    }
    std::cout << "\n";
  }

  getBulkData()->modification_end();

  return;
}

//
//
//
void Topology::setEntitiesOpen()
{
  EntityVector
  boundary_entities;

  stk::mesh::Selector
  select_owned = getMetaData()->locally_owned_part();

  stk::mesh::get_selected_entities(
      select_owned,
      getBulkData()->buckets(getBoundaryRank()) ,
      boundary_entities);

  // Iterate over the boundary entities
  for (size_t i = 0; i < boundary_entities.size(); ++i) {

    Entity &
    entity = *(boundary_entities[i]);

    if (checkOpen(entity) == false) continue;

    setFractureState(entity, OPEN);

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

  return;
}

/**
 * \brief Iterates over the boundary entities of the mesh of (all entities
 * of rank dimension-1) and checks fracture criterion.
 *
 * \param map of entity and boolean value is entity open
 *
 * If fracture_criterion is met, the entity and all lower order entities
 * associated with it are marked as open.
 */
void Topology::setEntitiesOpen(std::map<EntityKey, bool>& entity_open)
{
  // Fracture occurs at the boundary of the elements in the mesh.
  //   The rank of the boundary elements is one less than the
  //   dimension of the system.
  EntityVector boundary_list;
  //    stk::mesh::Selector select_owned_or_shared = getMetaData()->locally_owned_part() | getMetaData()->globally_shared_part();
  stk::mesh::Selector select_owned = getMetaData()->locally_owned_part();

  //    stk::mesh::get_selected_entities( select_owned_or_shared ,
  stk::mesh::get_selected_entities( select_owned,
      getBulkData()->buckets(getSpaceDimension() - 1 ) ,
      boundary_list );

  // Iterate over the boundary entities
  for (int i = 0; i < boundary_list.size(); ++i) {
    Entity& entity = *(boundary_list[i]);
    bool is_open = fracture_criterion_->check(entity);
    // If the criterion is met, need to set lower rank entities
    //   open as well
    if (is_open == true && getSpaceDimension() == 3) {
      entity_open[entity.key()] = true;
      PairIterRelation segments = entity.relations(
          entity.entity_rank() - 1);
      // iterate over the segments
      for (int j = 0; j < segments.size(); ++j) {
        Entity & segment = *(segments[j].entity());
        entity_open[segment.key()] = true;
        PairIterRelation nodes = segment.relations(
            segment.entity_rank() - 1);
        // iterate over nodes
        for (int k = 0; k < nodes.size(); ++k) {
          Entity& node = *(nodes[k].entity());
          entity_open[node.key()] = true;
        }
      }
    }
    // If the mesh is 2D
    else if (is_open == true && getSpaceDimension() == 2) {
      entity_open[entity.key()] = true;
      PairIterRelation nodes = entity.relations(
          entity.entity_rank() - 1);
      // iterate over nodes
      for (int j = 0; j < nodes.size(); ++j) {
        Entity & node = *(nodes[j].entity());
        entity_open[node.key()] = true;
      }
    }
  }

  return;

}

/**
 * \brief Iterates over the boundary entities contained in the passed-in
 * vector and opens each edge traversed.
 *
 * \param vector of edges to open, map of entity and boolean value is entity opened
 *
 * If entity is in the vector, the entity and all lower order entities
 * associated with it are marked as open.
 */

void Topology::setEntitiesOpen(const EntityVector& fractured_edges,
    std::map<EntityKey, bool>& entity_open)
{

  entity_open.clear();

  // Iterate over the boundary entities
  for (int i = 0; i < fractured_edges.size(); ++i) {
    Entity& entity = *(fractured_edges[i]);
    // Need to set lower rank entities
    //   open as well
    if (getSpaceDimension() == 3) {
      entity_open[entity.key()] = true;
      PairIterRelation segments = entity.relations(
          entity.entity_rank() - 1);
      // iterate over the segments
      for (int j = 0; j < segments.size(); ++j) {
        Entity & segment = *(segments[j].entity());
        entity_open[segment.key()] = true;
        PairIterRelation nodes = segment.relations(
            segment.entity_rank() - 1);
        // iterate over nodes
        for (int k = 0; k < nodes.size(); ++k) {
          Entity& node = *(nodes[k].entity());
          entity_open[node.key()] = true;
        }
      }
    }
    // If the mesh is 2D
    else if (getSpaceDimension() == 2) {
      entity_open[entity.key()] = true;
      PairIterRelation nodes = entity.relations(
          entity.entity_rank() - 1);
      // iterate over nodes
      for (int j = 0; j < nodes.size(); ++j) {
        Entity & node = *(nodes[j].entity());
        entity_open[node.key()] = true;
      }
    }
  }

  return;

}

namespace {

//
// Auxiliary for graphviz output
//
std::string
entity_label(EntityRank const rank)
{
  std::ostringstream
  oss;

  switch (rank) {
  default:
    oss << rank << "-Polytope";
    break;
  case NODE_RANK:
    oss << "Point";
    break;
  case EDGE_RANK:
    oss << "Segment";
    break;
  case FACE_RANK:
    oss << "Polygon";
    break;
  case VOLUME_RANK:
    oss << "Polyhedron";
    break;
  case 4:
    oss << "Polychoron";
    break;
  case 5:
    oss << "Polyteron";
    break;
  case 6:
    oss << "Polypeton";
    break;
  }

  return oss.str();
}

//
// Auxiliary for graphviz output
//
std::string
entity_color(EntityRank const rank, FractureState const fracture_state)
{
  std::ostringstream
  oss;

  switch (fracture_state) {

  default:
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "Fracture state is invalid: " << fracture_state;
    std::cerr << '\n';
    exit(1);
    break;

  case CLOSED:
    switch (rank) {
    default:
      oss << 2 * (rank + 1);
      break;
    case NODE_RANK:
      oss << "6";
      break;
    case EDGE_RANK:
      oss << "4";
      break;
    case FACE_RANK:
      oss << "2";
      break;
    case VOLUME_RANK:
      oss << "8";
      break;
    case 4:
      oss << "10";
      break;
    case 5:
      oss << "12";
      break;
    case 6:
      oss << "14";
      break;
    }
    break;

  case OPEN:
    switch (rank) {
    default:
      oss << 2 * rank + 1;
      break;
    case NODE_RANK:
      oss << "5";
      break;
    case EDGE_RANK:
      oss << "3";
      break;
    case FACE_RANK:
      oss << "1";
      break;
    case VOLUME_RANK:
      oss << "7";
      break;
    case 4:
      oss << "9";
      break;
    case 5:
      oss << "11";
      break;
    case 6:
      oss << "13";
      break;
    }
    break;
  }

  return oss.str();
}

//
// Auxiliary for graphviz output
//
std::string
dot_header()
{
  std::string
  header = "digraph mesh {\n";

  header += "  node [colorscheme=paired12]\n";
  header += "  edge [colorscheme=paired12]\n";

  return header;
}

//
// Auxiliary for graphviz output
//
std::string
dot_footer()
{
  return "}";
}

//
// Auxiliary for graphviz output
//
std::string
dot_entity(
    EntityId const id,
    EntityRank const rank,
    FractureState const fracture_state)
{
  std::ostringstream
  oss;

  oss << "  \"";
  oss << id;
  oss << "_";
  oss << rank;
  oss << "\"";
  oss << " [label=\"";
  //oss << entity_label(rank);
  //oss << " ";
  oss << id;
  oss << "\",style=filled,fillcolor=\"";
  oss << entity_color(rank, fracture_state);
  oss << "\"]\n";

  return oss.str();
}

//
// Auxiliary for graphviz output
//
std::string
relation_color(unsigned int const relation_id)
{
  std::ostringstream
  oss;

  switch (relation_id) {
  default:
    oss << 2 * (relation_id + 1);
    break;
  case 0:
    oss << "6";
    break;
  case 1:
    oss << "4";
    break;
  case 2:
    oss << "2";
    break;
  case 3:
    oss << "8";
    break;
  case 4:
    oss << "10";
    break;
  case 5:
    oss << "12";
    break;
  }

  return oss.str();
}

//
// Auxiliary for graphviz output
//
std::string
dot_relation(
    EntityId const source_id,
    EntityRank const source_rank,
    EntityId const target_id,
    EntityRank const target_rank,
    unsigned int const relation_local_id)
{
  std::ostringstream
  oss;

  oss << "  \"";
  oss << source_id;
  oss << "_";
  oss << source_rank;
  oss << "\" -> \"";
  oss << target_id;
  oss << "_";
  oss << target_rank;
  oss << "\" [color=\"";
  oss << relation_color(relation_local_id);
  oss << "\"]\n";

  return oss.str();
}

} //anonymous namspace

//
// Output the graph associated with the mesh to graphviz .dot
// file for visualization purposes. No need for entity_open map
// for this version
//
void
Topology::outputToGraphviz(std::string const & output_filename)
{
  // Open output file
  std::ofstream gviz_out;
  gviz_out.open(output_filename.c_str(), std::ios::out);

  if (gviz_out.is_open() == false) {
    std::cout << "Unable to open graphviz output file :";
    std::cout << output_filename << '\n';
    return;
  }

  std::cout << "Write graph to graphviz dot file" << '\n';

  // Write beginning of file
  gviz_out << dot_header();

  typedef std::vector<EntityVector> RelationList;

  RelationList
  relation_list;

  std::vector<size_t>
  relation_local_id;

  // Entities (graph vertices)
  for (EntityRank rank = NODE_RANK; rank <= getCellRank(); ++rank) {

    EntityVector
    entities;

    stk::mesh::get_entities(*(getBulkData()), rank, entities);

    for (EntityVector::size_type i = 0; i < entities.size(); ++i) {

      Entity &
      source_entity = *(entities[i]);

      FractureState const
      fracture_state = getFractureState(source_entity);

      PairIterRelation
      relations = relations_one_down(source_entity);

      gviz_out << dot_entity(source_entity.identifier(), rank, fracture_state);

      for (size_t j = 0; j < relations.size(); ++j) {

        EntityVector
        pair;

        pair.push_back(&source_entity);
        pair.push_back(relations[j].entity());

        relation_list.push_back(pair);
        relation_local_id.push_back(relations[j].identifier());
      }

    }

  }

  // Relations (graph edges)
  for (RelationList::size_type i = 0; i < relation_list.size(); ++i) {

    EntityVector
    pair = relation_list[i];

    Entity &
    source = *(pair[0]);

    Entity &
    target = *(pair[1]);

    gviz_out << dot_relation(
        source.identifier(), source.entity_rank(),
        target.identifier(), target.entity_rank(),
        relation_local_id[i]);

  }

  // File end
  gviz_out << dot_footer();

  gviz_out.close();

  return;
}

} // namespace LCM

