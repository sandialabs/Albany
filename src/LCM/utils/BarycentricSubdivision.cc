///
/// \file BarycentricSubdivision.cc
/// Barycentric subdivision for simplicial meshes
/// Implementation.
/// \author Juan Rojas
///

// Define only if ALbany is enabled
#if defined (ALBANY_LCM)

#include <stk_mesh/base/FieldData.hpp>
#include "Topology.h"

namespace LCM {

//INDENT EVERYTHING AGAIN USING THE SAME SPACE USED IN "Topology.h"!!!

//
// \brief Adds a new entity to the mesh
//
  void topology::add_entity(EntityRank entity_rank)
  {
    // Get space dimension by querying the STK discretization.
    Albany::STKDiscretization & stk_discretization =
        static_cast<Albany::STKDiscretization &>(*discretization_ptr_);
    int number_dimensions = stk_discretization.getSTKMeshStruct()->numDim;
    std::vector<size_t> requests(number_dimensions + 1, 0); // number of entity ranks. 1 + number of dimensions
    requests[entity_rank] = 1;
    stk::mesh::EntityVector newEntity;
    bulkData_->generate_new_entities(requests, newEntity);
    return;
  }

//
// \brief Removes an entity and all its connections
//
  void topology::remove_entity(Entity & entity)
  {
    // Destroy all relations to or from the entity
    Entity * entities = &entity;
    stk::mesh::PairIterRelation relations = entity.relations();
    for (int i = 0; i < relations.size(); ++i) {
      EdgeId edgeId = relations[i].identifier();
      Entity & target = *(relations[i].entity());
      bulkData_->destroy_relation(entity, target, edgeId);
    }
    // remove the entity from stk mesh
    bool deleted = bulkData_->destroy_entity(entities);
    assert(deleted);
    return;
  }

//
// \brief Adds a relation between two entities
//
  void topology::add_relation(Entity & source_entity, Entity & target_entity,
      EdgeId local_relation_id)
  {
    bulkData_->declare_relation(source_entity, target_entity,
        local_relation_id);
    return;
  }

//
// \brief Removes the relation between two entities
//
  void topology::remove_relation(Entity & source_entity, Entity & target_entity,
      EdgeId local_relation_id)
  {

    bulkData_->destroy_relation(source_entity, target_entity,
        local_relation_id);
    return;
  }

//
// \brief Returns a vector with all the mesh entities of a specific rank
//
  std::vector<Entity*> topology::get_entities_by_rank(
      const stk::mesh::BulkData & mesh, EntityRank entity_rank)
  {
    std::vector<Entity*> entities;
    const std::vector<stk::mesh::Bucket*> & ks = mesh.buckets(entity_rank);
    entities.clear();
    size_t count = 0;
    const std::vector<stk::mesh::Bucket*>::const_iterator ie = ks.end();
    std::vector<stk::mesh::Bucket*>::const_iterator ik = ks.begin();

    for (; ik != ie; ++ik) {
      count += (*ik)->size();
    }
    entities.reserve(count);

    ik = ks.begin();
    for (; ik != ie; ++ik) {
      const stk::mesh::Bucket & k = **ik;
      size_t n = k.size();
      for (size_t i = 0; i < n; ++i) {
        entities.push_back(&k[i]);
      }
    }
    return entities;
  }

//
// \brief Gets the local relation id (0,1,2,...) between two entities
//
  EdgeId topology::get_local_relation_id(Entity & source_entity,
      Entity & target_entity)
  {

    EdgeId local_id;
    stk::mesh::PairIterRelation source_relations = source_entity.relations();
    for (int ii = 0; ii < source_relations.size(); ++ii) {
      if (source_relations[ii].entity()->identifier()
          == target_entity.identifier()
          && source_relations[ii].entity()->entity_rank()
              == target_entity.entity_rank()) {
        local_id = source_relations[ii].identifier();
      }
    }
    return local_id;
  }

//
// \brief Returns the total number of lower rank entities connected to a specific entity
//
  unsigned int topology::get_number_lower_rank_entities(Entity & entity)
  {

    unsigned int count = 0;
    stk::mesh::PairIterRelation entity_relations = entity.relations();
    for (int ii = 0; ii < entity_relations.size(); ++ii) {
      if (entity.entity_rank() > entity_relations[ii].entity()->entity_rank()) {
        count++;
      }
    }
    return count;
  }

  /*
   * \brief Returns a group of entities connected directly to a given entity. The input rank is the rank
   *  of the returned entities.
   */
  std::vector<Entity*> topology::get_directly_connected_entities(
      Entity & entity, EntityRank entity_rank)
  {

    std::vector<Entity*> returned_entities;
    stk::mesh::PairIterRelation entity_relations = entity.relations();
    for (int ii = 0; ii < entity_relations.size(); ++ii) {
      if (entity_relations[ii].entity_rank() == entity_rank) {
        returned_entities.push_back(entity_relations[ii].entity());
      }
    }
    return returned_entities;
  }

//
// \brief Checks if an entity exists inside a specific vector. returns "0" for true and "1" for false
//
  unsigned int topology::find_entity_in_vector(std::vector<Entity*> & entities,
      Entity * entity)
  {

    std::vector<Entity*>::iterator iterator_entities;
    unsigned int number = 1;
    for (iterator_entities = entities.begin();
        iterator_entities != entities.end(); ++iterator_entities) {
      if (*iterator_entities == entity) {
        number = 0;
        break;
      }
    }
    return number;
  }

  /*
   *  \brief Returns a group of entities connected indirectly to a given entity.
   *  e.g. of returns: nodes  that belong to a face
   *  segments or nodes that belong to an  element
   *  The input rank is the rank of the returned entities.
   *  The input rank must be lower than that of the input entity
   *
   */
  std::vector<Entity*> topology::get_boundary_entities(Entity & entity,
      EntityRank entity_rank)
  {

    EntityRank given_entity_rank = entity.entity_rank();
    //Get entities of  "given_entity_rank -1"
    std::vector<std::vector<Entity*> > boundary_entities(given_entity_rank + 1);
    boundary_entities[given_entity_rank - 1] = get_directly_connected_entities(
        entity, given_entity_rank - 1);
    std::vector<Entity*>::iterator iterator_entities1;
    std::vector<Entity*>::iterator iterator_entities2;
    std::vector<Entity*> temp_vector1;
    for (int ii = given_entity_rank - 1; ii > entity_rank; ii--) {
      for (iterator_entities1 = boundary_entities[ii].begin();
          iterator_entities1 != boundary_entities[ii].end();
          ++iterator_entities1) {
        temp_vector1 = get_directly_connected_entities(*(*iterator_entities1),
            ii - 1);
        for (iterator_entities2 = temp_vector1.begin();
            iterator_entities2 != temp_vector1.end(); ++iterator_entities2) {
          if (find_entity_in_vector(boundary_entities[ii - 1],
              *iterator_entities2) == 1) {
            boundary_entities[ii - 1].push_back(*iterator_entities2);
          }
        }
        temp_vector1.clear();
      }
    }
    return boundary_entities[entity_rank];
  }

//
// \brief Checks if a segment is connected to an input node. Returns true "0" or false "1"
//
  unsigned int topology::check_segment_connection(Entity & segment,
      Entity * node)
  {

    int number = 1;
    std::vector<Entity*> segment_nodes = get_boundary_entities(segment, 0);
    std::vector<Entity*>::iterator Iterator_nodes;
    for (Iterator_nodes = segment_nodes.begin();
        Iterator_nodes != segment_nodes.end(); ++Iterator_nodes) {
      if (*Iterator_nodes == node) {
        number = 0;
      }
    }
    return number;
  }

  /*
   * \brief Finds the adjacent segments to a given segment. The adjacent segments are connected to a given common point.
   * it returns  adjacent segments
   */
  std::vector<Entity*> topology::find_adjacent_segments(Entity & segment,
      Entity * node)
  {

    std::vector<Entity*>::iterator Iterator_seg_nodes;
    std::vector<Entity*>::iterator Iterator_adj_nodes;
    std::vector<Entity*>::iterator Iterator_adj_seg;
    std::vector<Entity*> adjacent_segments;
    std::vector<Entity*> adjacent_segments_final;

    //Obtain the nodes corresponding to the input segment
    std::vector<Entity*> input_segment_nodes = get_directly_connected_entities(
        segment, 0);
    //Find the segments connected to "input_segment_nodes"
    for (Iterator_adj_nodes = input_segment_nodes.begin();
        Iterator_adj_nodes != input_segment_nodes.end(); ++Iterator_adj_nodes) {
      adjacent_segments = get_directly_connected_entities(
          *(*Iterator_adj_nodes), 1);
      //Which segment is connected to the input node?
      for (Iterator_adj_seg = adjacent_segments.begin();
          Iterator_adj_seg != adjacent_segments.end(); ++Iterator_adj_seg) {
        if (check_segment_connection(*(*Iterator_adj_seg), node) == 0) {
          adjacent_segments_final.push_back(*Iterator_adj_seg);
        }
      }
    }
    return adjacent_segments_final;
  }

//
// \brief Returns all the 3D entities connected to a given face
//
  std::vector<Entity*> topology::find_3D_relations(Entity & face)
  {
    std::vector<Entity*> entities_3D;
    stk::mesh::PairIterRelation _relations = face.relations();
    for (int i = 0; i < _relations.size(); ++i) {
      if (_relations[i].entity()->entity_rank() == 3) {
        entities_3D.push_back(_relations[i].entity());
      }
    }
    return entities_3D;
  }

  /*
   * \brief Returns all the segments at the boundary of a given element. Including those
   * connected between the faces barycenters and the faces boundary nodes
   */
  std::vector<Entity*> topology::find_segments_from_element(Entity & element)
  {
    std::vector<Entity*> element_faces;
    std::vector<Entity*> element_node;
    std::vector<Entity*> node_segments;
    std::vector<Entity*> _segments;
    std::vector<Entity*> outer_segments;
    std::vector<Entity*>::iterator iterator_element_faces;
    std::vector<Entity*>::iterator iterator_node_segments;
    std::vector<Entity*>::iterator iterator_outer_segments;

    element_faces = get_boundary_entities(element, 2);
    for (iterator_element_faces = element_faces.begin();
        iterator_element_faces != element_faces.end();
        ++iterator_element_faces) {
      element_node = get_directly_connected_entities(*(*iterator_element_faces),
          0);
      node_segments = get_directly_connected_entities(*(element_node[0]), 1);
      for (iterator_node_segments = node_segments.begin();
          iterator_node_segments != node_segments.end();
          ++iterator_node_segments) {
        _segments.push_back(*iterator_node_segments);
      }
    }
    outer_segments = get_boundary_entities(element, 1);
    for (iterator_outer_segments = outer_segments.begin();
        iterator_outer_segments != outer_segments.end();
        ++iterator_outer_segments) {
      _segments.push_back(*iterator_outer_segments);
    }
    return _segments;
  }

//
// \brief finds the adjacent faces from a given node
//
  std::vector<Entity*> topology::find_adjacent_faces_from_node(Entity & node)
  {
    std::vector<Entity*> adjacent_segments;
    std::vector<Entity*> adjacent_faces;
    std::vector<Entity*> adjacent_faces_final;
    std::vector<Entity*>::iterator iterator_adjacent_segments;
    std::vector<Entity*>::iterator iterator_adjacent_faces;
    std::vector<Entity*>::iterator iterator_adjacent_faces_final;

    adjacent_segments = get_directly_connected_entities(node, 1);
    for (iterator_adjacent_segments = adjacent_segments.begin();
        iterator_adjacent_segments != adjacent_segments.end();
        ++iterator_adjacent_segments) {
      adjacent_faces = get_directly_connected_entities(
          *(*iterator_adjacent_segments), 2);
      for (iterator_adjacent_faces = adjacent_faces.begin();
          iterator_adjacent_faces != adjacent_faces.end();
          ++iterator_adjacent_faces) {
        if (find_entity_in_vector(adjacent_faces_final,
            *iterator_adjacent_faces) == 1) {
          adjacent_faces_final.push_back(*iterator_adjacent_faces);
        }
      }
    }
    return adjacent_faces_final;
  }

  /*
   * \brief Returns "0" if the input faces have two points in common. Otherwise,
   * it returns "1"
   */
  int topology::compare_faces(Entity & face1, Entity & face2)
  {
    std::vector<Entity*> face1_nodes;
    std::vector<Entity*> face2_nodes;
    std::vector<Entity*> common_nodes;
    face1_nodes = get_boundary_entities(face1, 0);
    face2_nodes = get_boundary_entities(face2, 0);
    int num = 1;
    for (int ii = 0; ii < face2_nodes.size(); ++ii) {
      if (find_entity_in_vector(face1_nodes, face2_nodes[ii]) == 0) {
        common_nodes.push_back(face1_nodes[ii]);
      }
    }
    if (common_nodes.size() == 2) {
      num = 0;
    }
    return num;
  }

  /*
   * \brief Returns the adjacent faces to a given face;
   * "element_centroid" is the centroid of the element to which the face belongs
   */

  std::vector<Entity*> topology::find_adjacent_faces(Entity & face,
      Entity & element_centroid)
  {
    std::vector<Entity*> face_nodes;
    std::vector<Entity*> _element_internal_faces;
    std::vector<Entity*> adjacent_faces;
    std::vector<Entity*>::iterator iterator_element_internal_faces;
    _element_internal_faces = find_adjacent_faces_from_node(element_centroid);
    for (iterator_element_internal_faces = _element_internal_faces.begin();
        iterator_element_internal_faces != _element_internal_faces.end();
        ++iterator_element_internal_faces) {
      if (compare_faces(face, *(*iterator_element_internal_faces)) == 0) {
        adjacent_faces.push_back(*iterator_element_internal_faces);
      }
    }
    return adjacent_faces;
  }

//
// \brief Returns a pointer with the coordinates of a given entity
//
  double*
  topology::get_pointer_of_coordintes(Entity * entity)
  {

    Teuchos::RCP<Albany::AbstractDiscretization> discretization_ptr =
        get_Discretization();
    Albany::STKDiscretization & stk_discretization =
        static_cast<Albany::STKDiscretization &>(*discretization_ptr);
    //Obtain the stkMeshStruct
    Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct =
        stk_discretization.getSTKMeshStruct();
    //Create the pointer of coordinates
    double* pointer_coordinates = stk::mesh::field_data(
        *stkMeshStruct->coordinates_field, *entity);

    return pointer_coordinates;
  }

//
// brief Returns a vector with the corresponding former boundary nodes of an input entity of rank 3
//

  std::vector<Entity*> topology::get_former_element_nodes(Entity & element,
      std::vector<std::vector<Entity*> > & entities)
  {
    std::vector<Entity*> vector_nodes_;
    std::vector<Entity*> _boundary_nodes;
    vector_nodes_ = entities[element.identifier()];
    for (int ii = 0; ii < vector_nodes_.size(); ++ii) {
      _boundary_nodes.push_back(vector_nodes_[ii]);
    }
    return _boundary_nodes;
  }

  /*
   * brief Generates the coordinate of a given barycenter
   * "entities" is a vector with the entities of rank "0" that belong to the same higher rank entity
   *  connected to the barycenter(e.g segment, face, or element)
   */
  std::vector<double*> topology::create_coordinates(
      std::vector<Entity*> & entities, Entity * barycenter)
  {

    //vector of pointers
    std::vector<double*> vector_pointers;
    //Copy all the fields from entity1 to the new middle node called "barycenter"
    stk::mesh::BulkData* bulkData_ = get_BulkData();
    bulkData_->copy_entity_fields(*entities[0], *barycenter);

    //With the barycenter coordinate initialized, take the average between the entities that belong to
    //the vector called: "entities"
    for (int ii = 0; ii < entities.size(); ++ii) {
      vector_pointers.push_back(get_pointer_of_coordintes(entities[ii]));
    }
    //Pointer with coordinates without average
    std::vector<double> coordinates_(3);
    for (int ii = 0; ii < vector_pointers.size(); ++ii) {
      coordinates_[0] += vector_pointers[ii][0];
      coordinates_[1] += vector_pointers[ii][1];
      coordinates_[2] += vector_pointers[ii][2];
    }
    //Pointer with the barycenter coordinates
    double* barycenter_coordinates = get_pointer_of_coordintes(barycenter);
    barycenter_coordinates[0] = coordinates_[0] / (1.0 * entities.size());
    barycenter_coordinates[1] = coordinates_[1] / (1.0 * entities.size());
    barycenter_coordinates[2] = coordinates_[2] / (1.0 * entities.size());

    //THIS IS JUST FOR TESTING THE VALUES OF THE COORDINATES. 2D CASE
    std::vector<double*> final_vec_doubles;
    final_vec_doubles.push_back(vector_pointers[0]);
    final_vec_doubles.push_back(vector_pointers[1]);
    final_vec_doubles.push_back(vector_pointers[2]);
    final_vec_doubles.push_back(vector_pointers[3]);
    final_vec_doubles.push_back(barycenter_coordinates);

    return final_vec_doubles; //delete this return once done with checking coordinates
  }

//
// \brief Update the vector of vectors called: connectivity_temp with the new elements and nodes
// created by doing the barycentric subdivision on the mesh
//
  /*
   void
   topology::update_connectivity(std::vector<std::vector<Entity*> > & _connectivity_temp){
   std::vector<Entity*> modified_entities_3D;
   std::vector<Entity*> nodes_;
   for (int ii = 0; ii < modified_entities_3D.size();++ii){
   nodes_ = get_boundary;
   }
   }
   */

// \brief Barycentric subdivision of simplicial meshes
//
  void topology::barycentric_subdivision()
  {

    // Begin mesh update
    bulkData_->modification_begin();

//-----------------------------------------------------------------------------------------------------------------------------------
// I. Divide all the segments of the mesh by half
// initial_entities_0D: Vector with all the entities of rank "0" (nodes) required to divide the original mesh segments by half
// initial_entities_1D: Vector that contains all the entities of rank "1" (segments) of the original mesh
// modified1_entities_1D: Vector with all the segments required to divide the original mesh segments by half
// initial_entities_3D: vector with all the elements of the former mesh
// Assign coordinates to the new nodes needed to divide the segments by half
//------------------------------------------------------------------------------------------------------------------------------------

    //Get the segments from the original mesh
    std::vector<Entity*> initial_entities_1D = get_entities_by_rank(
        *(bulkData_), 1);
    std::vector<Entity*> vector_nodes;

    //Adding nodes to divide segments by half
    for (int ii = 0; ii < initial_entities_1D.size(); ++ii) {
      add_entity(0);
    }
    std::vector<Entity*> initial_entities_0D = get_entities_by_rank(
        *(bulkData_), 0);

    //vector with all elements from former mesh. This is used in step VI
    std::vector<Entity*> initial_entities_3D = get_entities_by_rank(
        *(bulkData_), 3);
    //Create a vector of vectors that contains all the former boundary nodes of all the elements of the mesh
    std::vector<std::vector<Entity*> > all_elements_boundary_nodes1(
        initial_entities_3D.size() + 1);
    //temporary vector //check the values inside this vector
    for (int ii = 0; ii < initial_entities_3D.size(); ++ii) {
      all_elements_boundary_nodes1[ii + 1] = get_boundary_entities(
          *(initial_entities_3D[ii]), 0);
    }

    for (int ii = 0; ii < initial_entities_1D.size(); ++ii) {
      //Create a vector with all the initial nodes connected to a segment
      vector_nodes = get_directly_connected_entities(*(initial_entities_1D[ii]),
          0);
      //Look for all the relations of each segment
      stk::mesh::PairIterRelation _relations =
          initial_entities_1D[ii]->relations();
      for (int i = 0; i < _relations.size(); ++i) {
        if (_relations[i].entity()->entity_rank() == 0
            && get_local_relation_id(*(initial_entities_1D[ii]),
                *(_relations[i].entity())) == 1) {
          //Add a blue(local relation) connection. This is only for reference
          add_relation(*(initial_entities_1D[ii]), *(_relations[i].entity()),
              2);
          //Remove the relation between the former segment and node
          remove_relation(*(initial_entities_1D[ii]), *(_relations[i].entity()),
              1);
          //Add a relation from the former segment to a new node
          add_relation(*(initial_entities_1D[ii]), *(initial_entities_0D[ii]),
              1);
        }
      }
      //Assign coordinates to the new nodes
      create_coordinates(vector_nodes, initial_entities_0D[ii]);
    }

    //Adding segments
    for (int ii = 0; ii < initial_entities_1D.size(); ++ii) {
      add_entity(1);
    }
    std::vector<Entity*> modified1_entities_1D = get_entities_by_rank(
        *(bulkData_), 1);

    for (int ii = 0; ii < initial_entities_1D.size(); ++ii) {
      //Look for all the relations of each segment
      stk::mesh::PairIterRelation _relations =
          initial_entities_1D[ii]->relations();
      for (int i = 0; i < _relations.size(); ++i) {
        if (_relations[i].entity()->entity_rank() == 0
            && get_local_relation_id(*(initial_entities_1D[ii]),
                *(_relations[i].entity())) == 2) {
          //Add a relation between the new segment and a node
          add_relation(*(modified1_entities_1D[ii]), *(_relations[i].entity()),
              0);
          //Remove this connection. This was only for reference
          remove_relation(*(initial_entities_1D[ii]), *(_relations[i].entity()),
              2);
          //Add a relation between the new segment and the "middle" node
          add_relation(*(modified1_entities_1D[ii]), *(initial_entities_0D[ii]),
              1);
        }
      }
    }

    //Adding the new segments to its corresponding faces
    //The segments can be connected to 1 one or more faces
    for (int ii = 0; ii < initial_entities_1D.size(); ++ii) {
      stk::mesh::PairIterRelation _relations =
          initial_entities_1D[ii]->relations();
      for (int i = 0; i < _relations.size(); ++i) {
        if (_relations[i].entity()->entity_rank() == 2) {
          add_relation(*(_relations[i].entity()), *(modified1_entities_1D[ii]),
              get_number_lower_rank_entities(*(_relations[i].entity())));
        }
      }
    }

    //Get the former faces from the mesh
    std::vector<Entity*> initial_entities_2D = get_entities_by_rank(
        *(bulkData_), 2);
    //Final number of segments per face after the division of the segments
    stk::mesh::PairIterRelation _relations =
        initial_entities_2D[0]->relations();
    std::vector<Entity*> segments;
    for (int i = 0; i < _relations.size(); ++i) {
      if (_relations[i].entity()->entity_rank() == 1) {
        segments.push_back(_relations[i].entity());
      }
    }
    //Number of segments per face after division by half
    unsigned int Num_segments_face = segments.size();

//-----------------------------------------------------------------------------------------------------------------------------------
// II.Connect the new center nodes to the center of the face
// mofified1_entities_0D: Vector of nodes that includes all the ones up the "node centers of the faces"
// initial_entities_2D: Vector with the faces of the original mesh
// Add the corresponding coordinates to the barycenters of all faces
//------------------------------------------------------------------------------------------------------------------------------------
    //Adding new nodes to the centers of the faces of the original mesh
    for (int ii = 0; ii < initial_entities_2D.size(); ++ii) {
      add_entity(0);
    }
    std::vector<Entity*> modified1_entities_0D = get_entities_by_rank(
        *(bulkData_), 0);

    for (int ii = 0; ii < initial_entities_2D.size(); ++ii) {
      //Connect the node to its corresponding face
      add_relation(*(initial_entities_2D[ii]), *(modified1_entities_0D[ii]),
          get_number_lower_rank_entities(*(initial_entities_2D[ii])));
    }

    //Add the corresponding coordinates to the barycenters of all faces
    std::vector<Entity*> boundary_nodes;
    for (int ii = 0; ii < initial_entities_2D.size(); ++ii) {
      boundary_nodes = get_boundary_entities(*(initial_entities_2D[ii]), 0);
      create_coordinates(boundary_nodes, modified1_entities_0D[ii]);
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// III. For each face start creating new segments that will connect the center point
// of the face with with all the points at its boundary
// modified2_entities_1D: Vector with all the segments up the new ones defined in III.
// vector_boundary_points: Vector with all the boundary nodes of all faces of the element
//------------------------------------------------------------------------------------------------------------------------------------

    //Add the new segments that will connect the center point with the points at the boundary
    for (int ii = 0; ii < initial_entities_2D.size(); ++ii) {
      //Get the relations of the face
      stk::mesh::PairIterRelation _relations =
          initial_entities_2D[ii]->relations();
      for (int i = 1; i < _relations.size(); ++i) {
        if (_relations[i].entity()->entity_rank() == 1) {
          //Add the new segments
          add_entity(1);
        }
      }
    }

    //Vector that contains the latest addition of segments
    std::vector<Entity*> modified2_entities_1D = get_entities_by_rank(
        *(bulkData_), 1);

    //Vector with all the boundary nodes of all faces of the element
    std::vector<Entity*>::iterator iterator_entities1;
    std::vector<Entity*>::iterator iterator_entities2;
    std::vector<Entity*> vector_boundary_points1;
    std::vector<Entity*> vector_boundary_points;
    for (iterator_entities1 = initial_entities_2D.begin();
        iterator_entities1 != initial_entities_2D.end(); ++iterator_entities1) {
      //Create a vector with all the "0 rank" boundaries (nodes)
      vector_boundary_points1 = get_boundary_entities(*(*iterator_entities1),
          0);
      //Push all the vector of boundary points into a single one
      for (iterator_entities2 = vector_boundary_points1.begin();
          iterator_entities2 != vector_boundary_points1.end();
          ++iterator_entities2) {
        vector_boundary_points.push_back(*iterator_entities2);
      }
      vector_boundary_points1.clear();
    }

    //Connect the new segments to the corresponding nodes
    for (int ii = 0; ii < Num_segments_face * initial_entities_2D.size();
        ++ii) {
      //Add a relation between the segments and the center nodes
      add_relation(*(modified2_entities_1D[ii]),
          *(modified1_entities_0D[ii / Num_segments_face]), 0);
      //Add a relation between the segments and the boundary nodes
      add_relation(*(modified2_entities_1D[ii]), *(vector_boundary_points[ii]),
          1);
    }

    //Create a vector with all the boundary segments of the elements
    // "All_boundary_segments" and "Number_new_triangles_inside_element" used in step VIII.
    int Number_new_triangles_inside_element;
    std::vector<Entity*> All_boundary_segments;
    std::vector<Entity*> element_segments;
    std::vector<Entity*>::iterator iterator_elements_;
    std::vector<Entity*>::iterator iterator_element_segments;
    for (iterator_elements_ = initial_entities_3D.begin();
        iterator_elements_ != initial_entities_3D.end(); ++iterator_elements_) {
      element_segments = find_segments_from_element(*(*iterator_elements_));
      Number_new_triangles_inside_element = element_segments.size();
      for (iterator_element_segments = element_segments.begin();
          iterator_element_segments != element_segments.end();
          ++iterator_element_segments) {
        All_boundary_segments.push_back(*iterator_element_segments);
      }
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// IV. Define the new faces at the boundary of the elements
// modified1_entities_2D: Vector that contains all the faces up to the new ones at the boundary of the elements
// "all_faces_centroids" and "All_boundary_segments" defined below
//-----------------------------------------------------------------------------------------------------------------------------------
    //Add the new faces
    for (int ii = 0; ii < Num_segments_face * initial_entities_2D.size();
        ++ii) {
      add_entity(2);
    }
    std::vector<Entity*> modified1_entities_2D = get_entities_by_rank(
        *(bulkData_), 2);

    //iterators
    std::vector<Entity*>::iterator iterator_faces;
    std::vector<Entity*>::iterator iterator_segments;
    std::vector<Entity*>::iterator iterator_nodes;
    //vectors
    std::vector<Entity*> vector_segments;
    std::vector<Entity*> face_centroid;
    std::vector<Entity*> all_boundary_segments;
    std::vector<Entity*> all_faces_centroids;

    //Put all the boundary segments in a single vector. Likewise, the nodes.
    for (iterator_faces = initial_entities_2D.begin();
        iterator_faces != initial_entities_2D.end(); ++iterator_faces) {
      //vector_segments, This vector contains the boundary segments that conform a specific face
      vector_segments = get_directly_connected_entities(*(*iterator_faces), 1);
      face_centroid = get_directly_connected_entities(*(*iterator_faces), 0);
      for (iterator_segments = vector_segments.begin();
          iterator_segments != vector_segments.end(); ++iterator_segments) {
        all_boundary_segments.push_back(*iterator_segments);
      }
      for (iterator_nodes = face_centroid.begin();
          iterator_nodes != face_centroid.end(); ++iterator_nodes) {
        all_faces_centroids.push_back(*iterator_nodes);
      }
    }

    //Add the new faces to its corresponding segments and elements
    std::vector<Entity*> adjacent_segments;
    std::vector<Entity*> original_face_relations_3D;
    std::vector<Entity*> original_face;
    std::vector<Entity*>::iterator iterator_entities;
    for (int ii = 0; ii < all_boundary_segments.size(); ++ii) {
      adjacent_segments = find_adjacent_segments(*all_boundary_segments[ii],
          all_faces_centroids[ii / Num_segments_face]);
      add_relation(*(modified1_entities_2D[ii]), *(all_boundary_segments[ii]),
          0);
      add_relation(*(modified1_entities_2D[ii]), *(adjacent_segments[0]), 1);
      add_relation(*(modified1_entities_2D[ii]), *(adjacent_segments[1]), 2);
      //Add the new face to its corresponding element
      original_face = get_directly_connected_entities(
          *(all_faces_centroids[ii / Num_segments_face]), 2);
      //find original_face 3D relations (entities)
      original_face_relations_3D = find_3D_relations(*(original_face[0]));
      for (iterator_entities = original_face_relations_3D.begin();
          iterator_entities != original_face_relations_3D.end();
          iterator_entities++) {
        add_relation(*(*iterator_entities), *(modified1_entities_2D[ii]),
            get_number_lower_rank_entities(*(*iterator_entities)));
      }
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// V. Delete former mesh faces
// initial_entities_3D:  Vector that contains all the former elements of the mesh
// All_boundary_faces:vector with all the boundary faces of all elements.
// This vector doesn't include the faces inside the elements
//-----------------------------------------------------------------------------------------------------------------------------------

    //Because "remove_entity" cannot be used to delete the relation between faces and elements
    //Remove first the relations between elements and faces
    std::vector<Entity*>::iterator iterator_entities_3D;
    for (int ii = 0; ii < initial_entities_2D.size(); ++ii) {
      std::vector<Entity*> former_face = get_directly_connected_entities(
          *(all_faces_centroids[ii]), 2);
      std::vector<Entity*> elements = get_directly_connected_entities(
          *(former_face[0]), 3);
      for (iterator_entities_3D = elements.begin();
          iterator_entities_3D != elements.end(); ++iterator_entities_3D) {
        remove_relation(*(*iterator_entities_3D), *(former_face[0]),
            get_local_relation_id(*(*iterator_entities_3D), *(former_face[0])));
      }
    }

    //Remove the former mesh faces and all their relations
    for (int ii = 0; ii < initial_entities_2D.size(); ++ii) {
      std::vector<Entity*> former_face = get_directly_connected_entities(
          *(all_faces_centroids[ii]), 2);
      remove_entity(*(former_face[0]));
    }

    //The following variables will be used in step IX
    //Number of faces per element "_faces_element". This number doesn't include any faces inside
    //the element.Only the ones at the boundary
    std::vector<Entity*> _faces_element;
    std::vector<Entity*> All_boundary_faces;
    std::vector<Entity*>::iterator iterator_faces_element;
    for (int ii = 0; ii < initial_entities_3D.size(); ++ii) {
      _faces_element = get_directly_connected_entities(
          *(initial_entities_3D[ii]), 2);
      for (iterator_faces_element = _faces_element.begin();
          iterator_faces_element != _faces_element.end();
          ++iterator_faces_element) {
        All_boundary_faces.push_back(*iterator_faces_element);
      }
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// VI. Add a point to each element. Each point represents the centroid of each element
// modified2_entities_0D: Vector that contains all the nodes up to the centroids of all the elements
// Add the coordinates of all the elements centroids
//-----------------------------------------------------------------------------------------------------------------------------------

    //Add a point to each element
    for (int ii = 0; ii < initial_entities_3D.size(); ++ii) {
      add_entity(0);
    }
    std::vector<Entity*> modified2_entities_0D = get_entities_by_rank(
        *(bulkData_), 0);

    //At this point the way the numbers are stored in the vector of nodes has changed
    //Thus, create a new vector with the nodes that has all the centroids of the elements
    std::vector<Entity*> elements_centroids;
    int _start = initial_entities_2D.size();
    int _end = initial_entities_2D.size() + initial_entities_3D.size();
    for (int ii = _start; ii < _end; ++ii) {
      elements_centroids.push_back(modified2_entities_0D[ii]);
    }

    //Add the coordinates of all the elements centroids
    std::vector<Entity*> boundary_nodes_elements;
    for (int ii = 0; ii < initial_entities_3D.size(); ++ii) {
      boundary_nodes_elements = get_former_element_nodes(
          *(initial_entities_3D[ii]), all_elements_boundary_nodes1);
      create_coordinates(boundary_nodes_elements, elements_centroids[ii]);
    }

    //Connect each element to the new added nodes
    for (int ii = 0; ii < initial_entities_3D.size(); ++ii) {
      //Connect the node to its corresponding element
      add_relation(*(initial_entities_3D[ii]), *(elements_centroids[ii]),
          get_number_lower_rank_entities(*(initial_entities_3D[ii])));
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// VII. For each element create new segments to connect its center point to all the points that
// compose its boundary
// modified3_entities_1D: Vector that contains all the segments up to the new ones defined in step VII.
//-----------------------------------------------------------------------------------------------------------------------------------

    //Add the new segments that will connect the center point with the points at the boundary
    //Create a vector with all the boundary points of all the former elements of the mesh
    std::vector<Entity*> element_boundary_nodes;
    std::vector<Entity*> all_elements_boundary_nodes;
    for (int ii = 0; ii < initial_entities_3D.size(); ++ii) {
      element_boundary_nodes = get_boundary_entities(*(initial_entities_3D[ii]),
          0);
      for (int ii = 0; ii < element_boundary_nodes.size(); ++ii) {
        add_entity(1);
        all_elements_boundary_nodes.push_back(element_boundary_nodes[ii]);
      }
    }

    //Vector that contains the latest addition of segments
    std::vector<Entity*> modified3_entities_1D = get_entities_by_rank(
        *(bulkData_), 1);

    //At this point the way the numbers are stored in the vector of segments has changed
    //Thus, create a new vector that contains the segments that connect the element centroid
    //with the element boundary points
    int Start_ = Num_segments_face * initial_entities_2D.size()
        + initial_entities_1D.size();
    int End_ = modified3_entities_1D.size() - initial_entities_1D.size();
    std::vector<Entity*> segments_connected_centroid;
    for (int ii = Start_; ii < End_; ++ii) {
      segments_connected_centroid.push_back(modified3_entities_1D[ii]);
    }

    //Connect the new segments to the corresponding nodes
    for (int ii = 0; ii < segments_connected_centroid.size(); ++ii) {
      //Add a relation between the segments and the center nodes
      add_relation(*(segments_connected_centroid[ii]),
          *(elements_centroids[ii / element_boundary_nodes.size()]), 0);
      //Add a relation between the segments and the boundary nodes
      add_relation(*(segments_connected_centroid[ii]),
          *(all_elements_boundary_nodes[ii]), 1);
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// VIII. Create the new faces inside each element  //REDO SLIDES HERE!!!
// modified2_entities_2D: Vector with all the faces up the ones that are inside the elements
//-----------------------------------------------------------------------------------------------------------------------------------

    //Add the new faces.
    for (int ii = 0; ii < All_boundary_segments.size(); ++ii) {
      add_entity(2);
    }
    std::vector<Entity*> modified2_entities_2D = get_entities_by_rank(
        *(bulkData_), 2);

    //Connect  the face to the corresponding segments
    std::vector<Entity*> adjacent_segments_inside;
    for (int ii = 0; ii < All_boundary_segments.size(); ++ii) {
      adjacent_segments_inside = find_adjacent_segments(
          *(All_boundary_segments[ii]),
          elements_centroids[ii / Number_new_triangles_inside_element]);
      add_relation(*(modified2_entities_2D[ii]), *(All_boundary_segments[ii]),
          0);
      add_relation(*(modified2_entities_2D[ii]), *(adjacent_segments_inside[0]),
          1);
      add_relation(*(modified2_entities_2D[ii]), *(adjacent_segments_inside[1]),
          2);
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// IX. Create the new elements  //ADD SLIDES
// modified1_entities_3D: Vector with all the elements required to carry out the barycentric
// subdivision
//-----------------------------------------------------------------------------------------------------------------------------------

    int number_new_elements = _faces_element.size()
        * initial_entities_3D.size();
    //Add the new elements
    for (int ii = 0; ii < number_new_elements; ++ii) {
      add_entity(3);
    }
    std::vector<Entity*> modified1_entities_3D = get_entities_by_rank(
        *(bulkData_), 3);

    //Connect the the element with its corresponding faces
    std::vector<Entity*> adjacent_faces_inside;
    for (int ii = 0; ii < All_boundary_faces.size(); ++ii) {
      adjacent_faces_inside = find_adjacent_faces(*(All_boundary_faces[ii]),
          *elements_centroids[ii / _faces_element.size()]);
      add_relation(*(modified1_entities_3D[ii]), *(All_boundary_faces[ii]), 0);
      add_relation(*(modified1_entities_3D[ii]), *adjacent_faces_inside[0], 1);
      add_relation(*(modified1_entities_3D[ii]), *adjacent_faces_inside[1], 2);
      add_relation(*(modified1_entities_3D[ii]), *adjacent_faces_inside[2], 3);
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// X. Delete the former elements
//
//-----------------------------------------------------------------------------------------------------------------------------------

    //Remove former elements from the mesh
    for (int ii = 0; ii < initial_entities_3D.size(); ++ii) {
      std::vector<Entity*> former_element = get_directly_connected_entities(
          *(elements_centroids[ii]), 3);
      remove_entity(*(former_element[0]));
    }

//-----------------------------------------------------------------------------------------------------------------------------------
// XI. Update the vector: connectivity_temp
//-----------------------------------------------------------------------------------------------------------------------------------

    //Vector with only the new elements. No former mesh elements appear here
    std::vector<Entity*> modified2_entities_3D = get_entities_by_rank(
        *(bulkData_), 3);

    //Connectivity matrix
    std::vector<std::vector<Entity*> > _connectivity_temp_(
        modified2_entities_3D.size());

    //Add the new entities to "connectivity_temp"
    for (int ii = 0; ii < modified2_entities_3D.size(); ++ii) {
      _connectivity_temp_[ii] = get_boundary_entities(
          *modified2_entities_3D[ii], 0);
    }
    connectivity_temp.clear();
    connectivity_temp = _connectivity_temp_;

    // End mesh update
    bulkData_->modification_end();

    // Recreate Albany STK Discretization
    Albany::STKDiscretization & stk_discretization =
        static_cast<Albany::STKDiscretization &>(*discretization_ptr_);

    Teuchos::RCP<Epetra_Comm> communicator =
        Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    stk_discretization.updateMesh(stkMeshStruct_, communicator);


    return;
  }

} // namespace LCM

#endif // #if defined (ALBANY_LCM)
