//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <sstream>

#include <boost/foreach.hpp>

#include "Teuchos_GlobalMPISession.hpp"

#include "Topology.h"

namespace LCM {

//
// Default constructor
//
Topology::Topology() :
    node_rank_(0),
    edge_rank_(1),
    face_rank_(2),
    cell_rank_(3),
    space_dimension_(3),
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
    node_rank_(0),
    edge_rank_(1),
    face_rank_(2),
    cell_rank_(3),
    space_dimension_(3),
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

  // Fracture the mesh randomly
  // Probability that fracture_criterion will return true.
  double const
  probability = 0.1;

  setFractureCriterion(
      Teuchos::rcp(new FractureCriterionRandom(space_dimension_, probability))
  );

  Topology::createDiscretization();

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
  node_rank_(0),
  edge_rank_(1),
  face_rank_(2),
  cell_rank_(3),
  space_dimension_(3),
  discretization_(Teuchos::null),
  stk_mesh_struct_(Teuchos::null),
  fracture_criterion_(Teuchos::null)
{
  setDiscretization(discretization);

  // Fracture the mesh randomly
  // Probability that fracture_criterion will return true.
  double const
  probability = 0.1;

  setFractureCriterion(
      Teuchos::rcp(new FractureCriterionRandom(space_dimension_, probability))
  );

  Topology::createDiscretization();

  return;
}

//
//
//
Topology::
Topology(RCP<Albany::AbstractDiscretization>& discretization,
    RCP<AbstractFractureCriterion>& fracture_criterion) :
    node_rank_(0),
    edge_rank_(1),
    face_rank_(2),
    cell_rank_(3),
    space_dimension_(3),
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

  EntityRank const
  cell_rank = getCellRank();

  for (EntityRank rank = 0; rank < cell_rank; ++rank) {

    std::vector<Bucket*> const &
    buckets = getBulkData()->buckets(rank);

    std::vector<Entity*>
    entities;

    stk::mesh::get_selected_entities(local_selector, buckets, entities);

    for (std::vector<Entity*>::size_type i = 0; i < entities.size(); ++i) {

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

  // The entity ranks
  setNodeRank(getMetaData()->node_rank());
  setEdgeRank(getMetaData()->edge_rank());
  setFaceRank(getMetaData()->face_rank());
  setCellRank(getMetaData()->element_rank());

  setSpaceDimension(getSTKMeshStruct()->numDim);

  // Get the topology of the elements. NOTE: Assumes one element
  // type in mesh.
  stk::mesh::Selector
  local_selector = getMetaData()->locally_owned_part();

  std::vector<Bucket*> const &
  buckets = getBulkData()->buckets(getCellRank());

  std::vector<Entity*>
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

  removeExtraRelations();
  initializeFractureState();

  getBulkData()->modification_end();

  return;
}

//
// Removes all multilevel relations.
//
void Topology::removeMultiLevelRelations()
{
  typedef std::vector<Entity*> EntityList;
  typedef std::vector<EdgeId> EdgeIdList;

  // Go from segments and above
  for (EntityRank rank = 2; rank <= getCellRank(); ++rank) {

    EntityList
    entities;

    stk::mesh::get_entities(*(getBulkData()), rank, entities);

    for (EntityList::size_type i = 0; i < entities.size(); ++i) {

      Entity &
      entity = *(entities[i]);

      PairIterRelation
      relations = entity.relations();

      EntityList
      far_entities;

      EdgeIdList
      multilevel_relation_ids;

      // Collect relations to delete
      for (PairIterRelation::iterator relation_iter = relations.begin();
          relation_iter != relations.end(); ++relation_iter) {

        EntityRank const
        target_rank = relation_iter->entity_rank();

        if (rank - target_rank > 1) {
          far_entities.push_back(relation_iter->entity());
          multilevel_relation_ids.push_back(relation_iter->identifier());
        }

      }

      // Delete them
      for (EdgeIdList::size_type i = 0;
          i < multilevel_relation_ids.size(); ++i) {

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
  std::vector<Entity*> element_list;
  stk::mesh::get_entities(*(getBulkData()), cell_rank_, element_list);

  // Remove extra relations from element
  for (int i = 0; i < element_list.size(); ++i) {
    Entity & element = *(element_list[i]);
    PairIterRelation relations = element.relations();
    std::vector<Entity*> del_relations;
    std::vector<int> del_ids;
    for (PairIterRelation::iterator j = relations.begin();
        j != relations.end(); ++j) {
      // remove all relationships from element unless to faces(segments
      //   in 2D) or nodes
      if (j->entity_rank() != cell_rank_ - 1
          && j->entity_rank() != node_rank_) {
        del_relations.push_back(j->entity());
        del_ids.push_back(j->identifier());
      }
    }
    for (int j = 0; j < del_relations.size(); ++j) {
      Entity & entity = *(del_relations[j]);
      getBulkData()->destroy_relation(element, entity, del_ids[j]);
    }
  };

  if (cell_rank_ == 3) {
    // Remove extra relations from face
    std::vector<Entity*> face_list;
    stk::mesh::get_entities(*(getBulkData()), cell_rank_ - 1, face_list);
    EntityRank entityRank = face_list[0]->entity_rank();
    for (int i = 0; i < face_list.size(); ++i) {
      Entity & face = *(face_list[i]);
      PairIterRelation relations = face_list[i]->relations();
      std::vector<Entity*> del_relations;
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
  std::vector<Entity*> element_list;
  stk::mesh::get_entities(*(getBulkData()), cell_rank_, element_list);

  getBulkData()->modification_begin();
  for (int i = 0; i < element_list.size(); ++i) {
    PairIterRelation nodes = element_list[i]->relations(node_rank_);
    std::vector<Entity*> temp;
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

//----------------------------------------------------------------------------
std::vector<std::vector<Entity*> >
Topology::getElementToNodeConnectivity()
{
  // Create a list of element entities
  std::vector<Entity*> element_list;
  std::vector<Entity*> node_list;
  stk::mesh::get_entities(*(getBulkData()), cell_rank_, element_list);

  // vector to store the entity pointers
  std::vector<std::vector<Entity*> > element_to_node_connectivity;

  // Loop over the elements
  const int number_of_elements = element_list.size();

  for (int i(0); i < number_of_elements; ++i) {

    PairIterRelation relations =
        element_list[i]->relations(node_rank_);

    const int nodes_per_element = relations.size();

    for (int j(0); j < nodes_per_element; ++j) {
      Entity* node = relations[j].entity();
      node_list.push_back(node);
    }
    element_to_node_connectivity.push_back(node_list);
  }
  return element_to_node_connectivity;
}

//----------------------------------------------------------------------------
void
Topology::
removeElementToNodeConnectivity(std::vector<std::vector<Entity*> >& oldElemToNode)
{
  // Create the temporary connectivity array
  std::vector<Entity*> element_list;
  stk::mesh::get_entities(*(getBulkData()), cell_rank_, element_list);

  getBulkData()->modification_begin();
  for (int i = 0; i < element_list.size(); ++i) {
    PairIterRelation nodes = element_list[i]->relations(node_rank_);
    std::vector<Entity*> temp;
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
  std::vector<Entity*> element_list;
  stk::mesh::get_entities(*(getBulkData()), cell_rank_, element_list);

  getBulkData()->modification_begin();

  // Add relations from element to nodes
  for (int i = 0; i < element_list.size(); ++i) {
    Entity & element = *(element_list[i]);
    std::vector<Entity*> element_connectivity = connectivity_temp_[i];
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
restoreElementToNodeConnectivity(std::vector<std::vector<Entity*> >& oldElemToNode)
{
  std::vector<Entity*> element_list;
  stk::mesh::get_entities(*(getBulkData()), cell_rank_, element_list);

  //    getBulkData()->modification_begin(); // need to comment GAH?

  // Add relations from element to nodes
  for (int i = 0; i < element_list.size(); ++i) {
    Entity & element = *(element_list[i]);
    std::vector<Entity*> element_connectivity = oldElemToNode[i];
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
std::vector<Entity*> Topology::getFaceNodes(Entity * entity)
{
  std::vector<Entity*> face_nodes;

  PairIterRelation elements = entity->relations(cell_rank_);
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

//----------------------------------------------------------------------------
//
// Creates a mesh of the fractured surfaces only.
//
void
Topology::outputSurfaceMesh()
{
  for (std::set<std::pair<Entity*, Entity*> >::iterator i =
      fractured_faces_.begin(); i != fractured_faces_.end(); ++i) {
    Entity * face1 = (*i).first;
    Entity * face2 = (*i).second;
    // create an ordered list of nodes for the faces
    // For now, output the face nodes. TODO: replace with mesh output code
    std::vector<Entity*> face_nodes = Topology::getFaceNodes(face1);
    std::cout << "Nodes of Face " << (face1)->identifier() << ": ";
    for (std::vector<Entity*>::iterator j = face_nodes.begin();
        j != face_nodes.end(); ++j) {
      std::cout << (*j)->identifier() << ":";
    }
    std::cout << "\n";

    face_nodes = Topology::getFaceNodes(face2);
    std::cout << "Nodes of Face " << (face2)->identifier() << ": ";
    for (std::vector<Entity*>::iterator j = face_nodes.begin();
        j != face_nodes.end(); ++j) {
      std::cout << (*j)->identifier() << ":";
    }
    std::cout << "\n";

  }
  return;
}

//----------------------------------------------------------------------------
//
// Create cohesive connectivity
//
std::vector<Entity*>
Topology::createCohesiveConnectivity(Entity* face1,
    Entity* face2)
{
  // number of nodes for the face
  unsigned numFaceNodes =
      getCellTopology().getNodeCount(face1->entity_rank(), 0);

  // Traverse down the graph from the face. The first node of
  // segment $n$ is node $n$ of the face.
  PairIterRelation face1Relations =
      face1->relations(face1->entity_rank() - 1);
  PairIterRelation face2Relations =
      face2->relations(face2->entity_rank() - 1);

  std::vector<Entity*> connectivity(2 * numFaceNodes);

  for (int i = 0; i < face1Relations.size(); ++i) {
    Entity * entity1 = face1Relations[i].entity();
    Entity * entity2 = face2Relations[i].entity();
    // If space_dimension_ = 2, the out edge targets from the
    // faces are nodes
    if (entity1->entity_rank() == node_rank_) {
      connectivity[i] = entity1;
      connectivity[i + numFaceNodes] = entity2;
    }
    // If space_dimension_ = 3, the out edge targets from the
    // faces are segments Take the 1st out edge of the segment
    // relation list
    else {
      PairIterRelation seg1Relations =
          entity1->relations(entity1->entity_rank() - 1);
      PairIterRelation seg2Relations =
          entity2->relations(entity2->entity_rank() - 1);

      // Check for the correct node to add to the connectivity
      // vector Each node should be used once.
      if ((i == 0)
          || (i > 0 && connectivity[i - 1] != seg1Relations[0].entity())
          || (i == numFaceNodes - 1
              && connectivity[0] != seg1Relations[0].entity())) {
        connectivity[i] = seg1Relations[0].entity();
        connectivity[i + numFaceNodes] = seg2Relations[0].entity();
      } else {
        connectivity[i] = seg1Relations[1].entity();
        connectivity[i + numFaceNodes] = seg2Relations[1].entity();
      }
    }
  }

  return connectivity;
}

//----------------------------------------------------------------------------
//
// Create vectors describing the vertices and edges of the star of
// an entity in the stk mesh.
//
void
Topology::createStar(std::set<EntityKey> & subgraph_entity_list,
    std::set<stkEdge, EdgeLessThan> & subgraph_edge_list,
    Entity & entity)
{
  PairIterRelation relations =
      entity.relations(entity.entity_rank() + 1);
  subgraph_entity_list.insert(entity.key());
  for (PairIterRelation::iterator i = relations.begin();
      i != relations.end(); ++i) {
    stk::mesh::Relation relation = *i;
    Entity & source = *(relation.entity());
    stkEdge edge;
    edge.source = source.key();
    edge.target = entity.key();
    edge.local_id = relation.identifier();
    subgraph_edge_list.insert(edge);
    Topology::createStar(subgraph_entity_list, subgraph_edge_list, source);
  }

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
  std::vector<Entity*> node_list; //all nodes
  std::vector<Entity*> open_node_list; //only the open nodes
  stk::mesh::Selector select_owned_or_shared = getMetaData()->locally_owned_part() |
      getMetaData()->globally_shared_part();

  stk::mesh::get_selected_entities( select_owned_or_shared,
      getBulkData()->buckets( node_rank_ ),
      node_list );
  for (std::vector<Entity*>::iterator i = node_list.begin();
      i != node_list.end(); ++i) {
    Entity* entity = *i;
    if (entity_open[entity->key()] == true) {
      open_node_list.push_back(entity);
    }
  }

  getBulkData()->modification_begin();

  // Iterate over the open nodes
  for (std::vector<Entity*>::iterator i = open_node_list.begin();
      i != open_node_list.end(); ++i) {
    // Get set of open segments
    Entity * entity = *i;
    PairIterRelation relations = entity->relations(edge_rank_);
    std::vector<Entity*> open_segment_list;

    for (PairIterRelation::iterator j = relations.begin();
        j != relations.end(); ++j) {
      Entity & source = *j->entity();
      if (entity_open[source.key()] == true) {
        open_segment_list.push_back(&source);
      }
    }

    // Iterate over the open segments
    for (std::vector<Entity*>::iterator j = open_segment_list.begin();
        j != open_segment_list.end(); ++j) {
      Entity * segment = *j;
      // Create star of segment
      std::set<EntityKey> subgraph_entity_list;
      std::set<stkEdge, EdgeLessThan> subgraph_edge_list;
      Topology::createStar(subgraph_entity_list, subgraph_edge_list, *segment);
      // Iterators
      std::set<EntityKey>::iterator firstEntity = subgraph_entity_list.begin();
      std::set<EntityKey>::iterator lastEntity = subgraph_entity_list.end();
      std::set<stkEdge>::iterator firstEdge = subgraph_edge_list.begin();
      std::set<stkEdge>::iterator lastEdge = subgraph_edge_list.end();

      Subgraph subgraph(getBulkData(), firstEntity, lastEntity, firstEdge,
          lastEdge, space_dimension_);

      // Clone open faces
      PairIterRelation faces = segment->relations(face_rank_);
      std::vector<Entity*> open_face_list;
      // create a list of open faces
      for (PairIterRelation::iterator k = faces.begin();
          k != faces.end(); ++k) {
        Entity & source = *k->entity();
        if (entity_open[source.key()] == true) {
          open_face_list.push_back(&source);
        }
      }

      // Iterate over the open faces
      for (std::vector<Entity*>::iterator k = open_face_list.begin();
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
    Topology::createStar(subgraph_entity_list, subgraph_edge_list, *entity);
    // Iterators
    std::set<EntityKey>::iterator firstEntity = subgraph_entity_list.begin();
    std::set<EntityKey>::iterator lastEntity = subgraph_entity_list.end();
    std::set<stkEdge>::iterator firstEdge = subgraph_edge_list.begin();
    std::set<stkEdge>::iterator lastEdge = subgraph_edge_list.end();
    Subgraph subgraph(getBulkData(), firstEntity, lastEntity, firstEdge, lastEdge,
        space_dimension_);

    Vertex node = subgraph.globalToLocal(entity->key());
    std::map<Entity*, Entity*> new_connectivity =
        subgraph.splitArticulationPoint(node, entity_open);

    // Update the connectivity
    for (std::map<Entity*, Entity*>::iterator j = new_connectivity.begin();
        j != new_connectivity.end(); ++j) {
      Entity* element = (*j).first;
      Entity* newNode = (*j).second;

      int element_id = element_global_to_local_ids_[element->identifier()];
      std::vector<Entity*> & element_connectivity = connectivity_temp_[element_id];
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
  for (std::set<std::pair<Entity*, Entity*> >::iterator i =
      fractured_faces_.begin(); i != fractured_faces_.end(); ++i, ++j) {
    Entity * face1 = (*i).first;
    Entity * face2 = (*i).second;
    std::vector<Entity*> cohesive_connectivity;
    cohesive_connectivity = Topology::createCohesiveConnectivity(face1, face2);

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
  std::vector<Entity*> open_node_list; // Global open node list

  std::cout << " \n\nGlobal stuff in fracture_boundary\n\n" << '\n';

  // Build list of open nodes (global)

  std::pair<EntityKey,bool> me; // what a map<EntityKey, bool> is made of

  BOOST_FOREACH(me, global_entity_open) {

    if(stk::mesh::entity_rank( me.first) == node_rank_){

      Entity *entity = getBulkData()->get_entity(me.first);
      std::cout << "Found open node: " << entity->identifier() << " belonging to pe: " << entity->owner_rank() << '\n';
      open_node_list.push_back(entity);
    }
  }

  getBulkData()->modification_begin();

  // Iterate over the open nodes
  for (std::vector<Entity*>::iterator i = open_node_list.begin();
      i != open_node_list.end(); ++i) {
    // Get set of open segments
    Entity * entity = *i;
    PairIterRelation relations = entity->relations(edge_rank_);
    std::vector<Entity*> open_segment_list;

    for (PairIterRelation::iterator j = relations.begin();
        j != relations.end(); ++j) {
      Entity & source = *j->entity();
      if (global_entity_open[source.key()] == true) {
        std::cout << "Found open segment: " << source.identifier() << " belonging to pe: " << source.owner_rank() << '\n';
        open_segment_list.push_back(&source);
      }
    }

    // Iterate over the open segments
    for (std::vector<Entity*>::iterator j = open_segment_list.begin();
        j != open_segment_list.end(); ++j) {
      Entity * segment = *j;

      // Create star of segment
      std::set<EntityKey> subgraph_entity_list;
      std::set<stkEdge, EdgeLessThan> subgraph_edge_list;
      Topology::createStar(subgraph_entity_list, subgraph_edge_list, *segment);

      // Iterators
      std::set<EntityKey>::iterator first_entity = subgraph_entity_list.begin();
      std::set<EntityKey>::iterator last_entity = subgraph_entity_list.end();
      std::set<stkEdge>::iterator first_edge = subgraph_edge_list.begin();
      std::set<stkEdge>::iterator last_edge = subgraph_edge_list.end();

      Subgraph subgraph(getBulkData(), first_entity, last_entity, first_edge,
          last_edge, space_dimension_);

      // Clone open faces
      PairIterRelation faces = segment->relations(face_rank_);
      std::vector<Entity*> open_face_list;

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
      for (std::vector<Entity*>::iterator k = open_face_list.begin();
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
    Topology::createStar(subgraph_entity_list, subgraph_edge_list, *entity);
    // Iterators
    std::set<EntityKey>::iterator firstEntity = subgraph_entity_list.begin();
    std::set<EntityKey>::iterator lastEntity = subgraph_entity_list.end();
    std::set<stkEdge>::iterator firstEdge = subgraph_edge_list.begin();
    std::set<stkEdge>::iterator lastEdge = subgraph_edge_list.end();
    Subgraph subgraph(getBulkData(), firstEntity, lastEntity, firstEdge,
        lastEdge, space_dimension_);

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
      //        std::vector<Entity*> & element_connectivity = connectivity_temp_[id - 1];
      std::vector<Entity*> & element_connectivity = connectivity_temp_[element_local_id];
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
  for (std::set<std::pair<Entity*, Entity*> >::iterator i =
      fractured_faces_.begin(); i != fractured_faces_.end(); ++i, ++j) {
    Entity * face1 = (*i).first;
    Entity * face2 = (*i).second;
    std::vector<Entity*> cohesive_connectivity;
    cohesive_connectivity = Topology::createCohesiveConnectivity(face1, face2);

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
  std::vector<Entity*> boundary_list;
  //    stk::mesh::Selector select_owned_or_shared = getMetaData()->locally_owned_part() | getMetaData()->globally_shared_part();
  stk::mesh::Selector select_owned = getMetaData()->locally_owned_part();

  //    stk::mesh::get_selected_entities( select_owned_or_shared ,
  stk::mesh::get_selected_entities( select_owned,
      getBulkData()->buckets(space_dimension_ - 1 ) ,
      boundary_list );

  // Iterate over the boundary entities
  for (int i = 0; i < boundary_list.size(); ++i) {
    Entity& entity = *(boundary_list[i]);
    bool is_open = fracture_criterion_->check(entity);
    // If the criterion is met, need to set lower rank entities
    //   open as well
    if (is_open == true && space_dimension_ == 3) {
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
    else if (is_open == true && space_dimension_ == 2) {
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

void Topology::setEntitiesOpen(const std::vector<Entity*>& fractured_edges,
    std::map<EntityKey, bool>& entity_open)
{

  entity_open.clear();

  // Iterate over the boundary entities
  for (int i = 0; i < fractured_edges.size(); ++i) {
    Entity& entity = *(fractured_edges[i]);
    // Need to set lower rank entities
    //   open as well
    if (space_dimension_ == 3) {
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
    else if (space_dimension_ == 2) {
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
  case 0:
    oss << "Point";
    break;
  case 1:
    oss << "Segment";
    break;
  case 2:
    oss << "Polygon";
    break;
  case 3:
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
    case 0:
      oss << "5";
      break;
    case 1:
      oss << "3";
      break;
    case 2:
      oss << "1";
      break;
    case 3:
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

  typedef std::vector<Entity*> EntityList;
  typedef std::vector<EntityList> RelationList;

  RelationList
  relation_list;

  std::vector<unsigned int>
  relation_local_id;

  // Entities (graph vertices)
  for (EntityRank rank = 0; rank <= getCellRank(); ++rank) {

    EntityList
    entities;

    stk::mesh::get_entities(*(getBulkData()), rank, entities);

    for (EntityList::size_type i = 0; i < entities.size(); ++i) {

      Entity &
      entity = *(entities[i]);

      FractureState const
      fracture_state = getFractureState(entity);

      PairIterRelation
      relations = entity.relations();

      gviz_out << dot_entity(entity.identifier(), rank, fracture_state);

      for (size_t j = 0; j < relations.size(); ++j) {
        if (relations[j].entity_rank() < entity.entity_rank()) {

          EntityList
          pair;

          pair.push_back(&entity);
          pair.push_back(relations[j].entity());

          relation_list.push_back(pair);
          relation_local_id.push_back(relations[j].identifier());
        }
      }

    }

  }

  // Relations (graph edges)
  for (RelationList::size_type i = 0; i < relation_list.size(); ++i) {

    EntityList
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

//----------------------------------------------------------------------------
//
// Create a subgraph given two vectors: a vertex list and a edge list.
//
Subgraph::Subgraph(stk::mesh::BulkData* bulk_data,
    std::set<EntityKey>::iterator first_vertex,
    std::set<EntityKey>::iterator last_vertex,
    std::set<Topology::stkEdge>::iterator first_edge,
    std::set<Topology::stkEdge>::iterator last_edge, int num_dim)
{
  // stk mesh data
  bulk_data_ = bulk_data;
  num_dim_ = num_dim;

  // Insert vertices and create the vertex map
  std::set<EntityKey>::iterator vertex_iterator;
  for (vertex_iterator = first_vertex;
      vertex_iterator != last_vertex;
      ++vertex_iterator) {
    // get global vertex
    EntityKey global_vertex = *vertex_iterator;
    // get entity rank
    EntityRank vertex_rank =
        bulk_data_->get_entity(global_vertex)->entity_rank();

    // get the new local vertex
    Vertex local_vertex = boost::add_vertex(*this);

    local_global_vertex_map_.
    insert(std::map<Vertex, EntityKey>::value_type(local_vertex,
        global_vertex));
    global_local_vertex_map_.
    insert(std::map<EntityKey, Vertex>::value_type(global_vertex,
        local_vertex));

    // store entity rank to vertex property
    VertexNamePropertyMap vertex_property_map =
        boost::get(VertexName(), *this);
    boost::put(vertex_property_map, local_vertex, vertex_rank);
  }

  // Add edges to the subgraph
  std::set<Topology::stkEdge>::iterator edge_iterator;
  for (edge_iterator = first_edge;
      edge_iterator != last_edge;
      ++edge_iterator) {
    // Get the edge
    Topology::stkEdge global_edge = *edge_iterator;

    // Get global source and target vertices
    EntityKey global_source_vertex = global_edge.source;
    EntityKey global_target_vertex = global_edge.target;

    // Get local source and target vertices
    Vertex localSourceVertex =
        global_local_vertex_map_.find(global_source_vertex)->second;
    Vertex localTargetVertex =
        global_local_vertex_map_.find(global_target_vertex)->second;

    Edge localEdge;
    bool inserted;

    EdgeId edge_id = global_edge.local_id;

    boost::tie(localEdge, inserted) = boost::add_edge(localSourceVertex,
        localTargetVertex, *this);

    assert(inserted);

    // Add edge id to edge property
    EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(), *this);
    boost::put(edgePropertyMap, localEdge, edge_id);
  }
  return;
}

/**
 * \brief Map a vertex in the subgraph to a entity in the stk mesh.
 *
 * \param[in] Vertex in the subgraph
 * \return Global entity key for the stk mesh
 *
 * Return the global entity key (in the stk mesh) given a local
 * subgraph vertex (in the boost subgraph).
 */
EntityKey Subgraph::localToGlobal(Vertex localVertex)
{

  std::map<Vertex, EntityKey>::const_iterator vertexMapIterator =
      local_global_vertex_map_.find(localVertex);

  assert(vertexMapIterator != local_global_vertex_map_.end());

  return (*vertexMapIterator).second;
}

//----------------------------------------------------------------------------
//
// Map a entity in the stk mesh to a vertex in the subgraph.
//
Vertex Subgraph::globalToLocal(EntityKey globalVertexKey)
{
  std::map<EntityKey, Vertex>::const_iterator vertexMapIterator =
      global_local_vertex_map_.find(globalVertexKey);

  assert(vertexMapIterator != global_local_vertex_map_.end());

  return (*vertexMapIterator).second;
}

//----------------------------------------------------------------------------
//
// Add a vertex in the subgraph.
//
Vertex Subgraph::addVertex(EntityRank vertex_rank)
{
  // Insert the vertex into the stk mesh
  // First have to request a new entity of rank N
  // number of entity ranks. 1 + number of dimensions
  std::vector<size_t> requests(num_dim_ + 1, 0);
  requests[vertex_rank] = 1;
  EntityVector new_entity;
  bulk_data_->generate_new_entities(requests, new_entity);
  Entity & globalVertex = *(new_entity[0]);

  // Insert the vertex into the subgraph
  Vertex localVertex = boost::add_vertex(*this);

  // Update maps
  local_global_vertex_map_.
  insert(std::map<Vertex, EntityKey>::value_type(localVertex,
      globalVertex.key()));
  global_local_vertex_map_.
  insert(std::map<EntityKey, Vertex>::value_type(globalVertex.key(),
      localVertex));

  // store entity rank to the vertex property
  VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);
  boost::put(vertexPropertyMap, localVertex, vertex_rank);

  return localVertex;
}


void
Subgraph::communicate_and_create_shared_entities(Entity   & node,
    EntityKey   new_node_key){

  stk::CommAll comm(bulk_data_->parallel());

  {
    stk::mesh::PairIterEntityComm entity_comm = node.sharing();

    for (; entity_comm.first != entity_comm.second; ++entity_comm.first) {

      unsigned proc = entity_comm.first->proc;
      comm.send_buffer(proc).pack<EntityKey>(node.key())
                                  .pack<EntityKey>(new_node_key);

    }
  }

  comm.allocate_buffers(bulk_data_->parallel_size()/4 );

  {
    stk::mesh::PairIterEntityComm entity_comm = node.sharing();

    for (; entity_comm.first != entity_comm.second; ++entity_comm.first) {

      unsigned proc = entity_comm.first->proc;
      comm.send_buffer(proc).pack<EntityKey>(node.key())
                                  .pack<EntityKey>(new_node_key);

    }
  }

  comm.communicate();

  const stk::mesh::PartVector no_parts;

  for (size_t process = 0; process < bulk_data_->parallel_size(); ++process) {
    EntityKey old_key;
    EntityKey new_key;

    while ( comm.recv_buffer(process).remaining()) {

      comm.recv_buffer(process).unpack<EntityKey>(old_key)
                                     .unpack<EntityKey>(new_key);

      Entity * new_entity = & bulk_data_->declare_entity(new_key.rank(), new_key.id(), no_parts);
      //std::cout << " Proc: " << bulk_data_->parallel_rank() << " created entity: (" << new_entity->identifier() << ", " <<
      //new_entity->entity_rank() << ")." << '\n';

    }
  }

}

void
Subgraph::bcast_key(unsigned root, EntityKey&   node_key){

  stk::CommBroadcast comm(bulk_data_->parallel(), root);

  unsigned rank = bulk_data_->parallel_rank();

  if(rank == root)

    comm.send_buffer().pack<EntityKey>(node_key);

  comm.allocate_buffer();

  if(rank == root)

    comm.send_buffer().pack<EntityKey>(node_key);

  comm.communicate();

  comm.recv_buffer().unpack<EntityKey>(node_key);

}

Vertex Subgraph::cloneVertex(Vertex & vertex)
{

  // Get the vertex rank
  EntityRank vertex_rank = Subgraph::getVertexRank(vertex);
  EntityKey vertex_key = Subgraph::localToGlobal(vertex);

  // Determine which processor should create the new vertex
  Entity *  old_vertex = bulk_data_->get_entity(vertex_key);

  //    if(!oldVertex){
  //std::cout << "oldVertex is NULL at line " << __LINE__ << " in file " << __FILE__ << '\n';
  //    }

  // For now, the owner of the new vertex is the same as the owner of the old one
  int owner_proc = old_vertex->owner_rank();

  // The owning processor inserts a new vertex into the stk mesh
  // First have to request a new entity of rank N
  std::vector<size_t> requests(num_dim_ + 1, 0); // number of entity ranks. 1 + number of dimensions
  EntityVector new_entity;
  const stk::mesh::PartVector no_parts;

  int my_proc = bulk_data_->parallel_rank();
  int source;
  Entity *global_vertex;
  EntityKey global_vertex_key;
  EntityKey::raw_key_type gvertkey;

  if(my_proc == owner_proc){

    // Insert the vertex into the stk mesh
    // First have to request a new entity of rank N
    requests[vertex_rank] = 1;

    // have stk build the new entity, then broadcast the key

    bulk_data_->generate_new_entities(requests, new_entity);
    global_vertex = new_entity[0];
    //std::cout << " Proc: " << bulk_data_->parallel_rank() << " created entity: (" << global_vertex->identifier() << ", " <<
    //global_vertex->entity_rank() << ")." << '\n';
    global_vertex_key = global_vertex->key();
    gvertkey = global_vertex_key.raw_key();

  }
  else {

    // All other processors do a no-op

    bulk_data_->generate_new_entities(requests, new_entity);

  }

  Subgraph::bcast_key(owner_proc, global_vertex_key);

  if(my_proc != owner_proc){ // All other processors receive the key

    // Get the vertex from stk

    const stk::mesh::PartVector no_parts;
    Entity * new_entity = & bulk_data_->declare_entity(global_vertex_key.rank(), global_vertex_key.id(), no_parts);

  }

  // Insert the vertex into the subgraph
  Vertex local_vertex = boost::add_vertex(*this);

  // Update maps
  local_global_vertex_map_.insert(
      std::map<Vertex, EntityKey>::value_type(local_vertex,
          global_vertex_key));
  global_local_vertex_map_.insert(
      std::map<EntityKey, Vertex>::value_type(global_vertex_key,
          local_vertex));

  // store entity rank to the vertex property
  VertexNamePropertyMap vertex_property_map = boost::get(VertexName(), *this);
  boost::put(vertex_property_map, local_vertex, vertex_rank);

  return local_vertex;
}

//----------------------------------------------------------------------------
//
// Remove vertex in subgraph
//
void Subgraph::removeVertex(Vertex & vertex)
{
  // get the global entity key of vertex
  EntityKey key = localToGlobal(vertex);

  // look up entity from key
  Entity* entity = bulk_data_->get_entity(key);

  // remove the vertex and key from global_local_vertex_map_ and
  // local_global_vertex_map_
  global_local_vertex_map_.erase(key);
  local_global_vertex_map_.erase(vertex);

  // remove vertex from subgraph
  // first have to ensure that there are no edges in or out of the vertex
  boost::clear_vertex(vertex, *this);
  // remove the vertex
  boost::remove_vertex(vertex, *this);

  // destroy all relations to or from the entity
  PairIterRelation relations = entity->relations();
  for (int i = 0; i < relations.size(); ++i) {
    EdgeId edgeId = relations[i].identifier();

    Entity & target = *(relations[i].entity());

    bulk_data_->destroy_relation(*entity, target, edgeId);
  }
  // remove the entity from stk mesh
  bool deleted = bulk_data_->destroy_entity(entity);
  assert(deleted);

  return;
}

//----------------------------------------------------------------------------
//
// Add edge to local graph.
//
std::pair<Edge, bool>
Subgraph::addEdge(const EdgeId edge_id,
    const Vertex local_source_vertex,
    const Vertex local_target_vertex)
{
  // Add edge to local graph
  std::pair<Edge, bool> local_edge =
      boost::add_edge(local_source_vertex, local_target_vertex, *this);

  if (local_edge.second == false) return local_edge;

  // get global entities
  EntityKey global_source_key = localToGlobal(local_source_vertex);
  EntityKey global_target_key = localToGlobal(local_target_vertex);
  Entity* global_source_vertex = bulk_data_->get_entity(global_source_key);
  Entity* global_target_vertex = bulk_data_->get_entity(global_target_key);

  //testing
  if (global_source_vertex->entity_rank() - global_target_vertex->entity_rank()
      != 1) {
    std::cout << "add edge:" << global_source_vertex->entity_rank() << ","
        << global_source_vertex->identifier() << " "
        << global_target_vertex->entity_rank() << ","
        << global_target_vertex->identifier() << "\n";
  }

  // Add edge to stk mesh
  bulk_data_->declare_relation(*(global_source_vertex),
      *(global_target_vertex),
      edge_id);

  // Add edge id to edge property
  EdgeNamePropertyMap edge_property_map = boost::get(EdgeName(), *this);
  boost::put(edge_property_map, local_edge.first, edge_id);

  return local_edge;
}

//----------------------------------------------------------------------------
void Subgraph::removeEdge(const Vertex & local_source_vertex,
    const Vertex & local_target_vertex)
{
  // Get the local id of the edge in the subgraph

  Edge edge;
  bool inserted;
  boost::tie(edge, inserted) =
      boost::edge(local_source_vertex,
          local_target_vertex, *this);

  assert(inserted);

  EdgeId edge_id = getEdgeId(edge);

  // remove local edge
  boost::remove_edge(local_source_vertex, local_target_vertex, *this);

  // remove relation from stk mesh
  EntityKey global_source_id = Subgraph::localToGlobal(local_source_vertex);
  EntityKey global_target_id = Subgraph::localToGlobal(local_target_vertex);
  Entity* global_source_vertex = bulk_data_->get_entity(global_source_id);
  Entity* global_target_vertex = bulk_data_->get_entity(global_target_id);

  bulk_data_->destroy_relation(*(global_source_vertex),
      *(global_target_vertex),
      edge_id);

  return;
}

//----------------------------------------------------------------------------
EntityRank &
Subgraph::getVertexRank(const Vertex vertex)
{
  VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);
  return boost::get(vertexPropertyMap, vertex);
}

//----------------------------------------------------------------------------
EdgeId &
Subgraph::getEdgeId(const Edge edge)
{
  EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(), *this);
  return boost::get(edgePropertyMap, edge);
}

//----------------------------------------------------------------------------
//
// Function determines whether the input vertex is an articulation
// point of the subgraph.
//
void
Subgraph::testArticulationPoint(Vertex input_vertex, int & num_components,
    std::map<Vertex, int> & subComponent)
{
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> undirectedGraph;
  typedef boost::graph_traits<undirectedGraph>::vertex_descriptor uVertex;
  typedef boost::graph_traits<undirectedGraph>::edge_descriptor uEdge;
  // Map to and from undirected graph and subgraph
  std::map<uVertex, Vertex> undirectedSubVertexMap;
  std::map<Vertex, uVertex> subUndirectedVertexMap;
  undirectedGraph g;
  VertexIterator vertex_begin;
  VertexIterator vertex_end;
  boost::tie(vertex_begin, vertex_end) = vertices(*this);

  // First add the vertices to the graph
  for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
    Vertex vertex = *i;
    if (vertex != input_vertex) {
      uVertex uvertex = boost::add_vertex(g);
      undirectedSubVertexMap[uvertex] = vertex;
      // Add to maps
      undirectedSubVertexMap.insert(
          std::map<uVertex, Vertex>::value_type(uvertex, vertex));
      subUndirectedVertexMap.insert(
          std::map<Vertex, uVertex>::value_type(vertex, uvertex));
    }
  }

  // Then add the edges
  for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
    Vertex source = *i;

    if (source != input_vertex) {
      std::map<Vertex, uVertex>::const_iterator sourceMapIterator =
          subUndirectedVertexMap.find(source);

      uVertex usource = (*sourceMapIterator).second;

      // write the edges in the subgraph
      OutEdgeIterator out_edge_begin;
      OutEdgeIterator out_edge_end;
      boost::tie(out_edge_begin, out_edge_end) = out_edges(*i, *this);

      for (OutEdgeIterator j = out_edge_begin; j != out_edge_end; ++j) {
        Vertex target;
        Edge edge = *j;
        target = boost::target(edge, *this);

        if (target != input_vertex) {
          std::map<Vertex, uVertex>::const_iterator targetMapIterator =
              subUndirectedVertexMap.find(target);

          uVertex utarget = (*targetMapIterator).second;

          boost::add_edge(usource, utarget, g);
        }
      }
    }
  }

  std::vector<int> component(boost::num_vertices(g));
  num_components = boost::connected_components(g, &component[0]);

  for (std::map<uVertex, Vertex>::iterator i = undirectedSubVertexMap.begin();
      i != undirectedSubVertexMap.end(); ++i) {
    Vertex vertex = (*i).second;
    subComponent.insert(
        std::map<Vertex, int>::value_type(vertex, component[(*i).first]));
  }

  return;
}

/**
 * \brief Clones a boundary entity from the subgraph and separates the in-edges
 * of the entity.
 *
 * \param[in] Boundary vertex
 * \param[out] New boundary vertex
 * \param Map of entity and boolean value is open
 *
 * Boundary entities are on boundary of the elements in the mesh. They
 * will thus have either 1 or 2 in-edges to elements.
 *
 * If there is only 1 in-edge, the entity may be on the exterior of the
 * mesh and is not a candidate for fracture for this subgraph. The
 * boundary entity may be a valid candidate in another step. If only 1
 * in edge: Return.
 *
 * Entity must have satisfied the fracture criterion and be labeled open
 * in map is_open. If not open: Return.
 */
void
Subgraph::cloneBoundaryEntity(Vertex & vertex, Vertex & new_vertex,
    std::map<EntityKey, bool> & entity_open)
{
  // Check that number of in_edges = 2
  boost::graph_traits<Graph>::degree_size_type num_in_edges =
      boost::in_degree(vertex, *this);
  if (num_in_edges != 2) return;

  // Check that vertex = open
  EntityKey vertex_key = Subgraph::localToGlobal(vertex);
  assert(entity_open[vertex_key]==true);

  // Get the vertex rank
  //    EntityRank vertex_rank = Subgraph::getVertexRank(vertex);

  // Create a new vertex of same rank as vertex
  //    newVertex = Subgraph::add_vertex(vertex_rank);
  new_vertex = Subgraph::cloneVertex(vertex);

  // Copy the out_edges of vertex to new_vertex
  OutEdgeIterator out_edge_begin;
  OutEdgeIterator out_edge_end;
  boost::tie(out_edge_begin, out_edge_end) = boost::out_edges(vertex, *this);
  for (OutEdgeIterator i = out_edge_begin; i != out_edge_end; ++i) {
    Edge edge = *i;
    EdgeId edgeId = Subgraph::getEdgeId(edge);
    Vertex target = boost::target(edge, *this);
    Subgraph::addEdge(edgeId, new_vertex, target);
  }

  // Copy all out edges not in the subgraph to the new vertex
  Subgraph::cloneOutEdges(vertex, new_vertex);

  // Remove one of the edges from vertex, copy to new_vertex
  // Arbitrarily remove the first edge from original vertex
  InEdgeIterator in_edge_begin;
  InEdgeIterator in_edge_end;
  boost::tie(in_edge_begin, in_edge_end) = boost::in_edges(vertex, *this);
  Edge edge = *(in_edge_begin);
  EdgeId edgeId = Subgraph::getEdgeId(edge);
  Vertex source = boost::source(edge, *this);
  Subgraph::removeEdge(source, vertex);

  // Add edge to new vertex
  Subgraph::addEdge(edgeId, source, new_vertex);

  // Have to clone the out edges of the original entity to the new entity.
  // These edges are not in the subgraph

  // Clone process complete, set entity_open to false
  entity_open[vertex_key] = false;

  return;
}

//----------------------------------------------------------------------------
//
// Splits an articulation point.
//
std::map<Entity*, Entity*>
Subgraph::splitArticulationPoint(Vertex vertex,
    std::map<EntityKey, bool> & entity_open)
{
  // Check that vertex = open
  EntityKey vertex_key = Subgraph::localToGlobal(vertex);
  assert(entity_open[vertex_key]==true);

  // get rank of vertex
  EntityRank vertex_rank = Subgraph::getVertexRank(vertex);

  // Create undirected graph
  int num_components;
  std::map<Vertex, int> components;
  Subgraph::testArticulationPoint(vertex, num_components, components);

  // The function returns an updated connectivity map. If the vertex
  //   rank is not node, then this map will be of size 0.
  std::map<Entity*, Entity*> new_connectivity;

  // Check number of connected components in undirected graph. If =
  // 1, return
  if (num_components == 1) return new_connectivity;

  // If number of connected components > 1, split vertex in subgraph and stk mesh
  // number of new vertices = numComponents - 1
  std::vector<Vertex> new_vertex;
  for (int i = 0; i < num_components - 1; ++i) {
    //      Vertex newVert = Subgraph::add_vertex(vertex_rank);
    Vertex new_vert = Subgraph::cloneVertex(vertex);
    new_vertex.push_back(new_vert);
  }

  // create a map of elements to new node numbers
  // only do this if the input vertex is a node (don't require otherwise)
  if (vertex_rank == 0) {
    for (std::map<Vertex, int>::iterator i = components.begin();
        i != components.end(); ++i) {
      int component_num = (*i).second;
      Vertex current_vertex = (*i).first;
      EntityRank current_rank = Subgraph::getVertexRank(current_vertex);
      // Only add to map if the vertex is an element
      if (current_rank == num_dim_ && component_num != 0) {
        Entity* element =
            bulk_data_->get_entity(Subgraph::localToGlobal(current_vertex));
        Entity* new_node =
            bulk_data_->
            get_entity(Subgraph::localToGlobal(new_vertex[component_num - 1]));
        new_connectivity.
        insert(std::map<Entity*, Entity*>::value_type(element, new_node));
      }
    }
  }

  // Copy the out edges of the original vertex to the new vertex
  for (int i = 0; i < new_vertex.size(); ++i) {
    Subgraph::cloneOutEdges(vertex, new_vertex[i]);
  }

  // vector for edges to be removed. Vertex is source and edgeId the
  // local id of the edge
  std::vector<std::pair<Vertex, EdgeId> > removed;

  // Iterate over the in edges of the vertex to determine which will
  // be removed
  InEdgeIterator in_edge_begin;
  InEdgeIterator in_edge_end;
  boost::tie(in_edge_begin, in_edge_end) = boost::in_edges(vertex, *this);
  for (InEdgeIterator i = in_edge_begin; i != in_edge_end; ++i) {
    Edge edge = *i;
    Vertex source = boost::source(edge, *this);

    std::map<Vertex, int>::const_iterator componentIterator =
        components.find(source);
    int vertComponent = (*componentIterator).second;
    Entity& entity =
        *(bulk_data_->get_entity(Subgraph::localToGlobal(source)));
    // Only replace edge if vertex not in component 0
    if (vertComponent != 0) {
      EdgeId edgeId = Subgraph::getEdgeId(edge);
      removed.push_back(std::make_pair(source, edgeId));
    }
  }

  // remove all edges in vector removed and replace with new edges
  for (std::vector<std::pair<Vertex, EdgeId> >::iterator i = removed.begin();
      i != removed.end(); ++i) {
    std::pair<Vertex, EdgeId> edge = *i;
    Vertex source = edge.first;
    EdgeId edgeId = edge.second;
    std::map<Vertex, int>::const_iterator componentIterator =
        components.find(source);
    int vertComponent = (*componentIterator).second;

    Subgraph::removeEdge(source, vertex);
    std::pair<Edge, bool> inserted =
        Subgraph::addEdge(edgeId, source,new_vertex[vertComponent - 1]);
    assert(inserted.second==true);
  }

  // split process complete, set entity_open to false
  entity_open[vertex_key] = false;

  return new_connectivity;
}

//----------------------------------------------------------------------------
//
// Clone all out edges of a vertex to a new vertex.
//
void Subgraph::cloneOutEdges(Vertex & original_vertex, Vertex & new_vertex)
{
  // Get the entity for the original and new vertices
  EntityKey original_key = Subgraph::localToGlobal(original_vertex);
  EntityKey new_key = Subgraph::localToGlobal(new_vertex);
  Entity & original_entity = *(bulk_data_->get_entity(original_key));
  Entity & new_entity = *(bulk_data_->get_entity(new_key));

  // Iterate over the out edges of the original vertex and check against the
  //   out edges of the new vertex. If the edge does not exist, add.
  PairIterRelation original_relations =
      original_entity.relations(original_entity.entity_rank() - 1);
  for (int i = 0; i < original_relations.size(); ++i) {
    PairIterRelation new_relations =
        new_entity.relations(new_entity.entity_rank() - 1);
    // assume the edge doesn't exist
    bool exists = false;
    for (int j = 0; j < new_relations.size(); ++j) {
      if (original_relations[i].entity() == new_relations[j].entity()) {
        exists = true;
      }
    }
    if (exists == false) {
      EdgeId edgeId = original_relations[i].identifier();
      Entity& target = *(original_relations[i].entity());
      bulk_data_->declare_relation(new_entity, target, edgeId);
    }
  }

  return;
}

/**
 * \brief Output the graph associated with the mesh to graphviz .dot
 * file for visualization purposes.
 *
 * \param[in] output file
 * \param[in] map of entity and boolean value is open
 *
 * Similar to outputToGraphviz function in Topology class.
 * If fracture criterion for entity is satisfied, the entity and all
 * associated lower order entities are marked open. All open entities are
 * displayed as such in output file.
 *
 * To create final output figure, run command below from terminal:
 *   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
 */
void Subgraph::outputToGraphviz(std::string & gviz_output,
    std::map<EntityKey, bool> entity_open)
{
  // Open output file
  std::ofstream gviz_out;
  gviz_out.open(gviz_output.c_str(), std::ios::out);

  std::cout << "Write graph to graphviz dot file\n";

  if (gviz_out.is_open()) {
    // Write beginning of file
    gviz_out << "digraph mesh {\n" << "  node [colorscheme=paired12]\n"
        << "  edge [colorscheme=paired12]\n";

    VertexIterator vertex_begin;
    VertexIterator vertex_end;
    boost::tie(vertex_begin, vertex_end) = vertices(*this);

    for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
      EntityKey key = localToGlobal(*i);
      Entity & entity = *(bulk_data_->get_entity(key));
      std::string label;
      std::string color;

      // Write the entity name
      switch (entity.entity_rank()) {
      // nodes
      case 0:
        label = "Node";
        if (entity_open[entity.key()] == false)
          color = "6";
        else
          color = "5";
        break;
        // segments
      case 1:
        label = "Segment";
        if (entity_open[entity.key()] == false)
          color = "4";
        else
          color = "3";
        break;
        // faces
      case 2:
        label = "Face";
        if (entity_open[entity.key()] == false)
          color = "2";
        else
          color = "1";
        break;
        // volumes
      case 3:
        label = "Element";
        if (entity_open[entity.key()] == false)
          color = "8";
        else
          color = "7";
        break;
      }
      gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
                     << "\" [label=\" " << label << " " << entity.identifier()
                     << "\",style=filled,fillcolor=\"" << color << "\"]\n";

      // write the edges in the subgraph
      OutEdgeIterator out_edge_begin;
      OutEdgeIterator out_edge_end;
      boost::tie(out_edge_begin, out_edge_end) = out_edges(*i, *this);

      for (OutEdgeIterator j = out_edge_begin; j != out_edge_end; ++j) {
        Edge out_edge = *j;
        Vertex source = boost::source(out_edge, *this);
        Vertex target = boost::target(out_edge, *this);

        EntityKey sourceKey = localToGlobal(source);
        Entity & global_source = *(bulk_data_->get_entity(sourceKey));

        EntityKey targetKey = localToGlobal(target);
        Entity & global_target = *(bulk_data_->get_entity(targetKey));

        EdgeId edgeId = getEdgeId(out_edge);

        switch (edgeId) {
        case 0:
          color = "6";
          break;
        case 1:
          color = "4";
          break;
        case 2:
          color = "2";
          break;
        case 3:
          color = "8";
          break;
        case 4:
          color = "10";
          break;
        case 5:
          color = "12";
          break;
        default:
          color = "9";
        }
        gviz_out << "  \"" << global_source.identifier() << "_"
            << global_source.entity_rank() << "\" -> \""
            << global_target.identifier() << "_"
            << global_target.entity_rank() << "\" [color=\"" << color << "\"]"
            << "\n";
      }

    }

    // File end
    gviz_out << "}";
    gviz_out.close();
  } else
    std::cout << "Unable to open graphviz output file 'output.dot'\n";

  return;
}

//
// Output the mesh connectivity
//
void
display_connectivity(Topology & topology)
{
  // Create a list of element entities
  std::vector<Entity*>
  elements;

  stk::mesh::get_entities(
      *(topology.getBulkData()),
      topology.getCellRank(),
      elements);

  typedef std::vector<Entity*>::size_type size_type;

  // Loop over the elements
  size_type const
  number_of_elements = elements.size();

  for (size_type i = 0; i < number_of_elements; ++i) {

    PairIterRelation
    relations = elements[i]->relations(topology.getNodeRank());

    EntityId const
    element_id = elements[i]->identifier();

    std::cout << std::setw(16) << element_id << ":";

    size_t const
    nodes_per_element = relations.size();

    for (size_t j = 0; j < nodes_per_element; ++j) {

      Entity const &
      node = *(relations[j].entity());

      EntityId const
      node_id = node.identifier();

      std::cout << std::setw(16) << node_id;
    }

    std::cout << '\n';
  }

  return;
}

//
// Output relations associated with entity
//
void
display_relation(Entity const & entity)
{
  std::cout << "Relations for entity (identifier,rank): ";
  std::cout << entity.identifier() << "," << entity.entity_rank();
  std::cout << '\n';

  PairIterRelation
  relations = entity.relations();

  for (size_t i = 0; i < relations.size(); ++i) {
    std::cout << "entity:\t";
    std::cout << relations[i].entity()->identifier() << ",";
    std::cout << relations[i].entity()->entity_rank();
    std::cout << "\tlocal id: ";
    std::cout << relations[i].identifier();
    std::cout << '\n';
  }
  return;
}

//
// Output relations of rank associated with entity
//
void
display_relation(Entity const & entity, EntityRank const rank)
{
  std::cout << "Relations of rank ";
  std::cout << rank;
  std::cout << " for entity (identifier,rank): ";
  std::cout << entity.identifier() << "," << entity.entity_rank();
  std::cout << '\n';

  PairIterRelation
  relations = entity.relations(rank);

  for (size_t i = 0; i < relations.size(); ++i) {
    std::cout << "entity:\t";
    std::cout << relations[i].entity()->identifier() << ",";
    std::cout << relations[i].entity()->entity_rank();
    std::cout << "\tlocal id: ";
    std::cout << relations[i].identifier();
    std::cout << '\n';
  }
  return;
}

//
// Add a dash and processor rank to a string. Useful for output
// file names.
//
std::string
parallelize_string(std::string const & string)
{
  std::ostringstream
  oss;

  oss << string;

  int const
  number_processors = Teuchos::GlobalMPISession::getNProc();

  if (number_processors > 1) {

    int const
    number_digits = static_cast<int>(std::log10(number_processors));

    int const
    processor_id = Teuchos::GlobalMPISession::getRank();

    oss << "-";
    oss << std::setfill('0') << std::setw(number_digits) << processor_id;
  }

  return oss.str();
}

} // namespace LCM

