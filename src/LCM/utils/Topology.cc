//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Topology.h"

#include <stk_mesh/base/EntityComm.hpp>
#include <boost/foreach.hpp>



namespace LCM {

  //----------------------------------------------------------------------------
  //
  // Default constructor for Topology (private to Topology class)
  //
  Topology::Topology() :
    number_dimensions_(0), discretization_ptr_(Teuchos::null)
  {
    return;
  }

  //----------------------------------------------------------------------------
  Topology::Topology(std::string const & input_file,
                     std::string const & output_file)
  {
    Teuchos::RCP<Teuchos::ParameterList> disc_params = 
      rcp(new Teuchos::ParameterList("params"));

    //set Method to Exodus and set input file name
    disc_params->set<std::string>("Method", "Exodus");
    disc_params->set<std::string>("Exodus Input File Name", input_file);
    disc_params->set<std::string>("Exodus Output File Name", output_file);
    //disc_params->print(std::cout);

    Teuchos::RCP<Epetra_Comm> communicator =
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    Albany::DiscretizationFactory disc_factory(disc_params, Teuchos::null, communicator);

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs =
      disc_factory.createMeshSpecs();

    Teuchos::RCP<Albany::StateInfoStruct> stateInfo = 
      Teuchos::rcp(new Albany::StateInfoStruct());

    Albany::AbstractFieldContainer::FieldContainerRequirements req; // The default fields

    discretization_ptr_ = disc_factory.createDiscretization(3, stateInfo, req);

    Topology::createDiscretization();

    // Create the full mesh representation. This must be done prior to
    // the adaptation query. We are reading the mesh from a file so do
    // it here.
    Topology::graphInitialization();

    // Fracture the mesh randomly
    fracture_criterion_ = 
      Teuchos::rcp(new GenericFractureCriterion(number_dimensions_, 
                                                element_rank_));

    return;
  }

  //----------------------------------------------------------------------------
  //
  // This constructor assumes that the full mesh graph has been
  // previously created. Topology::graphInitialization();
  //
  Topology::
  Topology(Teuchos::RCP<Albany::AbstractDiscretization> & discretization_ptr)
  {
    discretization_ptr_ = discretization_ptr;
    Topology::createDiscretization();
    fracture_criterion_ = 
      Teuchos::rcp(new GenericFractureCriterion(number_dimensions_, element_rank_));
    return;
  }

  //----------------------------------------------------------------------------
  Topology::
  Topology(Teuchos::RCP<Albany::AbstractDiscretization>& discretization_ptr,
           Teuchos::RCP<AbstractFractureCriterion>& fracture_criterion) :
    discretization_ptr_(discretization_ptr),
    fracture_criterion_(fracture_criterion)
  {
    Topology::createDiscretization();
    return;
  }

  //----------------------------------------------------------------------------
  //
  // Create Albany discretization
  //
  void 
  Topology::createDiscretization()
  {
    // Need to access the bulk_data and meta_data classes in the mesh
    // data structure
    Albany::STKDiscretization & stk_discretization =
      static_cast<Albany::STKDiscretization &>(*discretization_ptr_);

    stk_mesh_struct_ = stk_discretization.getSTKMeshStruct();

    bulk_data_ = stk_mesh_struct_->bulkData;
    meta_data_ = stk_mesh_struct_->metaData;

    // The entity ranks
    node_rank_ = meta_data_->NODE_RANK;
    edge_rank_ = meta_data_->EDGE_RANK;
    face_rank_ = meta_data_->FACE_RANK;
    element_rank_ = meta_data_->element_rank();
    number_dimensions_ = stk_mesh_struct_->numDim;

    // Get the topology of the elements. NOTE: Assumes one element
    // type in mesh.
    std::vector<Entity*> element_list;
    stk::mesh::Selector select_owned = meta_data_->locally_owned_part();

    stk::mesh::get_selected_entities( select_owned,
                                      bulk_data_->buckets( element_rank_ ),
                                      element_list );
    element_topology_ = stk::mesh::fem::get_cell_topology(*element_list[0]);

    return;
  }

  //----------------------------------------------------------------------------
  //
  // Output relations associated with entity
  //
  void Topology::displayRelation(Entity const & entity)
  {
    cout << "Relations for entity (identifier,rank): " << entity.identifier()
         << "," << entity.entity_rank() << "\n";
    stk::mesh::PairIterRelation relations = entity.relations();
    for (int i = 0; i < relations.size(); ++i) {
      cout << "entity:\t" << relations[i].entity()->identifier() << ","
           << relations[i].entity()->entity_rank() << "\tlocal id: "
           << relations[i].identifier() << "\n";
    }
    return;
  }

  //----------------------------------------------------------------------------
  //
  // Output relations of rank entityRank associated with entity
  //
  void 
  Topology::displayRelation(Entity const & entity,
                            EntityRank const entityRank)
  {
    cout << "Relations of rank " << entityRank
         << " for entity (identifier,rank): " << entity.identifier() << ","
         << entity.entity_rank() << "\n";
    stk::mesh::PairIterRelation relations = entity.relations(entityRank);
    for (int i = 0; i < relations.size(); ++i) {
      cout << "entity:\t" << relations[i].entity()->identifier() << ","
           << relations[i].entity()->entity_rank() << "\tlocal id: "
           << relations[i].identifier() << "\n";
    }
    return;
  }

  //----------------------------------------------------------------------------
  //
  // Output the mesh connectivity
  //
  void 
  Topology::displayConnectivity()
  {
    // Create a list of element entities
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

    // Loop over the elements
    const int number_of_elements = element_list.size();

    for (int i = 0; i < number_of_elements; ++i) {

      stk::mesh::PairIterRelation relations = 
        element_list[i]->relations(node_rank_);

      const int element_id = element_list[i]->identifier();
      cout << "Nodes of Element " << element_id << std::endl;

      const int nodes_per_element = relations.size();

      for (int j = 0; j < nodes_per_element; ++j) {
        Entity& node = *(relations[j].entity());

        const int node_id = node.identifier();
        cout << ":" << node_id;
      }

      cout << ":" << std::endl;
    }

    return;
  }

  //----------------------------------------------------------------------------
  //
  // Output the graph associated with the mesh to graphviz .dot
  // file for visualization purposes. No need for entity_open map
  // for this version
  //
  void 
  Topology::outputToGraphviz(std::string & output_filename)
  {
    std::map<EntityKey, bool> entity_open;
    outputToGraphviz(output_filename, entity_open);
    return;
  }

  //----------------------------------------------------------------------------
  //
  // Output the graph associated with the mesh to graphviz .dot file
  // for visualization purposes.
  //
  void 
  Topology::outputToGraphviz(std::string & output_filename,
                             std::map<EntityKey, bool> & entity_open)
  {
    // Open output file
    std::ofstream gviz_out;
    gviz_out.open(output_filename.c_str(), std::ios::out);

    cout << "Write graph to graphviz dot file\n";

    if (gviz_out.is_open()) {
      // Write beginning of file
      gviz_out << "digraph mesh {\n" << "  node [colorscheme=paired12]\n"
               << "  edge [colorscheme=paired12]\n";

      std::vector<Entity*> entity_list;
      stk::mesh::get_entities(*(bulk_data_), element_rank_, entity_list);

      std::vector<std::vector<Entity*> > relation_list;
      std::vector<int> relation_local_id;

      // Elements
      for (int i = 0; i < entity_list.size(); ++i) {
        Entity & entity = *(entity_list[i]);
        stk::mesh::PairIterRelation relations = entity.relations();

        gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
                 << "\" [label=\"Element " << entity.identifier()
                 << "\",style=filled,fillcolor=\"8\"]\n";
        for (int j = 0; j < relations.size(); ++j) {
          if (relations[j].entity_rank() < entity.entity_rank()) {
            std::vector<Entity*> temp;
            temp.push_back(&entity);
            temp.push_back(relations[j].entity());
            relation_list.push_back(temp);
            relation_local_id.push_back(relations[j].identifier());
          }
        }
      }

      stk::mesh::get_entities(*(bulk_data_), face_rank_, entity_list);

      // Faces
      for (int i = 0; i < entity_list.size(); ++i) {
        Entity & entity = *(entity_list[i]);
        stk::mesh::PairIterRelation relations = entity.relations();

        if (entity_open[entity.key()] == true)
          gviz_out << "  \"" << entity.identifier() << "_"
                   << entity.entity_rank() << "\" [label=\"Face "
                   << entity.identifier() << "\",style=filled,fillcolor=\"1\"]\n";
        else
          gviz_out << "  \"" << entity.identifier() << "_"
                   << entity.entity_rank() << "\" [label=\"Face "
                   << entity.identifier() << "\",style=filled,fillcolor=\"2\"]\n";
        for (int j = 0; j < relations.size(); ++j) {
          if (relations[j].entity_rank() < entity.entity_rank()) {
            std::vector<Entity*> temp;
            temp.push_back(&entity);
            temp.push_back(relations[j].entity());
            relation_list.push_back(temp);
            relation_local_id.push_back(relations[j].identifier());
          }
        }
      }

      stk::mesh::get_entities(*(bulk_data_), edge_rank_, entity_list);

      // Edges
      for (int i = 0; i < entity_list.size(); ++i) {
        Entity & entity = *(entity_list[i]);
        stk::mesh::PairIterRelation relations = entity.relations();

        if (entity_open[entity.key()] == true)
          gviz_out << "  \"" << entity.identifier() << "_"
                   << entity.entity_rank() << "\" [label=\"Segment "
                   << entity.identifier() << "\",style=filled,fillcolor=\"3\"]\n";
        else
          gviz_out << "  \"" << entity.identifier() << "_"
                   << entity.entity_rank() << "\" [label=\"Segment "
                   << entity.identifier() << "\",style=filled,fillcolor=\"4\"]\n";
        for (int j = 0; j < relations.size(); ++j) {
          if (relations[j].entity_rank() < entity.entity_rank()) {
            std::vector<Entity*> temp;
            temp.push_back(&entity);
            temp.push_back(relations[j].entity());
            relation_list.push_back(temp);
            relation_local_id.push_back(relations[j].identifier());
          }
        }
      }

      stk::mesh::get_entities(*(bulk_data_), node_rank_, entity_list);

      // Nodes
      for (int i = 0; i < entity_list.size(); ++i) {
        Entity & entity = *(entity_list[i]);

        if (entity_open[entity.key()] == true)
          gviz_out << "  \"" << entity.identifier() << "_"
                   << entity.entity_rank() << "\" [label=\"Node "
                   << entity.identifier() << "\",style=filled,fillcolor=\"5\"]\n";
        else
          gviz_out << "  \"" << entity.identifier() << "_"
                   << entity.entity_rank() << "\" [label=\"Node "
                   << entity.identifier() << "\",style=filled,fillcolor=\"6\"]\n";
      }

      for (int i = 0; i < relation_list.size(); ++i) {
        std::vector<Entity*> temp = relation_list[i];
        Entity& origin = *(temp[0]);
        Entity& destination = *(temp[1]);
        std::string color;
        switch (relation_local_id[i]) {
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
        gviz_out << "  \"" << origin.identifier() << "_" << origin.entity_rank()
                 << "\" -> \"" << destination.identifier() << "_"
                 << destination.entity_rank() << "\" [color=\"" << color << "\"]"
                 << "\n";
      }

      // File end
      gviz_out << "}";
      gviz_out.close();
    } else
      cout << "Unable to open graphviz output file 'output.dot'\n";

    return;
  }

  //----------------------------------------------------------------------------
  //
  // Initializes the default stk mesh object needed by class.
  //
  void Topology::graphInitialization()
  {
    stk::mesh::PartVector add_parts;
    stk::mesh::create_adjacent_entities(*(bulk_data_), add_parts);

    bulk_data_->modification_begin();
    Topology::removeExtraRelations();
    bulk_data_->modification_end();

    return;
  }

  //----------------------------------------------------------------------------
  //
  // Removes unneeded relations from the mesh.
  //
#if 1 // GAH - this is the original
  void Topology::removeExtraRelations()
  {
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

    // Remove extra relations from element
    for (int i = 0; i < element_list.size(); ++i) {
      Entity & element = *(element_list[i]);
      stk::mesh::PairIterRelation relations = element.relations();
      std::vector<Entity*> del_relations;
      std::vector<int> del_ids;
      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
           j != relations.end(); ++j) {
        // remove all relationships from element unless to faces(segments
        //   in 2D) or nodes
        if (j->entity_rank() != element_rank_ - 1
            && j->entity_rank() != node_rank_) {
          del_relations.push_back(j->entity());
          del_ids.push_back(j->identifier());
        }
      }
      for (int j = 0; j < del_relations.size(); ++j) {
        Entity & entity = *(del_relations[j]);
        bulk_data_->destroy_relation(element, entity, del_ids[j]);
      }
    };

    if (element_rank_ == 3) {
      // Remove extra relations from face
      std::vector<Entity*> face_list;
      stk::mesh::get_entities(*(bulk_data_), element_rank_ - 1, face_list);
      EntityRank entityRank = face_list[0]->entity_rank();
      for (int i = 0; i < face_list.size(); ++i) {
        Entity & face = *(face_list[i]);
        stk::mesh::PairIterRelation relations = face_list[i]->relations();
        std::vector<Entity*> del_relations;
        std::vector<int> del_ids;
        for (stk::mesh::PairIterRelation::iterator j = relations.begin();
             j != relations.end(); ++j) {
          if (j->entity_rank() != entityRank + 1
              && j->entity_rank() != entityRank - 1) {
            del_relations.push_back(j->entity());
            del_ids.push_back(j->identifier());
          }
        }
        for (int j = 0; j < del_relations.size(); ++j) {
          Entity & entity = *(del_relations[j]);
          bulk_data_->destroy_relation(face, entity, del_ids[j]);
        }
      }
    }

    return;
  }
#else

  void
  Topology::removeExtraRelations()
  {
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulk_data_),element_rank_,element_list);

    // Remove extra relations from element
    for (int i = 0; i < element_list.size(); ++i){
      Entity & element = *(element_list[i]);
      stk::mesh::PairIterRelation relations = element.relations();
      std::vector<Entity*> del_relations;
      std::vector<int> del_ids;
      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
           j != relations.end(); ++j){
        // remove all relationships from element unless to faces(segments
        //   in 2D) or nodes
        //			if (j->entity_rank() != element_rank_-1 && j->entity_rank() != node_rank_){
        // GAH THIS NEEDS TO BE UNCOMMENTED!!!!!
        if (j->entity_rank() != node_rank_){
          del_relations.push_back(j->entity());
          del_ids.push_back(j->identifier());
        }
      }
      for (int j = 0; j < del_relations.size(); ++j){
        Entity & entity = *(del_relations[j]);
        bulk_data_->destroy_relation(element,entity,del_ids[j]);
      }
    };

    if (element_rank_ == 3){
      // Remove extra relations from face
      std::vector<Entity*> face_list;
      stk::mesh::get_entities(*(bulk_data_),element_rank_-1,face_list);
      EntityRank entityRank = face_list[0]->entity_rank();
      for (int i = 0; i < face_list.size(); ++i){
        Entity & face = *(face_list[i]);
        stk::mesh::PairIterRelation relations = face_list[i]->relations();
        std::vector<Entity*> del_relations;
        std::vector<int> del_ids;
        for (stk::mesh::PairIterRelation::iterator j = relations.begin();
             j != relations.end(); ++j){
          if (j->entity_rank() != entityRank+1 &&
              j->entity_rank() != entityRank-1){
            del_relations.push_back(j->entity());
            del_ids.push_back(j->identifier());
          }
        }
        for (int j = 0; j < del_relations.size(); ++j){
          Entity & entity = *(del_relations[j]);
          bulk_data_->destroy_relation(face,entity,del_ids[j]);
        }
      }
    }

    return;
  }

#endif

  //----------------------------------------------------------------------------
  //
  // Creates temporary nodal connectivity for the elements and removes
  // the relationships between the elements and nodes.
  //
  void Topology::removeNodeRelations()
  {
    // Create the temporary connectivity array
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

    bulk_data_->modification_begin();
    for (int i = 0; i < element_list.size(); ++i) {
      stk::mesh::PairIterRelation nodes = element_list[i]->relations(node_rank_);
      std::vector<Entity*> temp;
      for (int j = 0; j < nodes.size(); ++j) {
        Entity* node = nodes[j].entity();
        temp.push_back(node);
      }
      connectivity_temp_.push_back(temp);

      for (int j = 0; j < temp.size(); ++j) {
        bulk_data_->destroy_relation(*(element_list[i]), *(temp[j]), j);
      }
    }

    bulk_data_->modification_end();

    return;
  }

  //----------------------------------------------------------------------------  
  std::vector<std::vector<Entity*> >
  Topology::getElementToNodeConnectivity() 
  {
    // Create a list of element entities
    std::vector<Entity*> element_list;
    std::vector<Entity*> node_list;
    stk::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

    // vector to store the entity pointers
    std::vector<std::vector<Entity*> > element_to_node_connectivity;

    // Loop over the elements
    const int number_of_elements = element_list.size();

    for (int i(0); i < number_of_elements; ++i) {

      stk::mesh::PairIterRelation relations = 
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
    stk::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

    bulk_data_->modification_begin();
    for (int i = 0; i < element_list.size(); ++i) {
      stk::mesh::PairIterRelation nodes = element_list[i]->relations(node_rank_);
      std::vector<Entity*> temp;
      for (int j = 0; j < nodes.size(); ++j) {
        Entity* node = nodes[j].entity();
        temp.push_back(node);
      }

      // save the current element to node connectivity and the local
      // to global numbering
      connectivity_temp.push_back(temp);
      element_global_to_local_ids[element_lst[i]->identifier()] = i;

      for (int j = 0; j < temp.size(); ++j) {
        bulk_data_->destroy_relation(*(element_list[i]), *(temp[j]), j);
      }
    }

    bulk_data_->modification_end();

    return;
  }

  //----------------------------------------------------------------------------
  //
  // After mesh manipulations are complete, need to recreate a stk
  // mesh understood by Albany_STKDiscretization.
  void Topology::restoreElementToNodeConnectivity()
  {
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

    bulkData_->modification_begin();

    // Add relations from element to nodes
    for (int i = 0; i < element_list.size(); ++i) {
      Entity & element = *(element_list[i]);
      std::vector<Entity*> element_connectivity = connectivity_temp_[i];
      for (int j = 0; j < element_connectivity.size(); ++j) {
        Entity & node = *(element_connectivity[j]);
        bulk_data_->declare_relation(element, node, j);
      }
    }

    // Recreate Albany STK Discretization
    Albany::STKDiscretization & stk_discretization =
      static_cast<Albany::STKDiscretization &>(*discretization_ptr_);

    Teuchos::RCP<Epetra_Comm> communicator =
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    //stk_discretization.updateMesh(stkMeshStruct_, communicator);
    stk_discretization.updateMesh();

    bulkData_->modification_end();

    return;
  }

  //----------------------------------------------------------------------------
  void 
  Topology::
  restoreElementToNodeConnectivity(std::vector<std::vector<Entity*> >& oldElemToNode)
  {
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulk_data_), element_rank_, element_list);

    //    bulk_data_->modification_begin(); // need to comment GAH?

    // Add relations from element to nodes
    for (int i = 0; i < element_list.size(); ++i) {
      Entity & element = *(element_list[i]);
      std::vector<Entity*> element_connectivity = oldElemToNode[i];
      for (int j = 0; j < element_connectivity.size(); ++j) {
        Entity & node = *(element_connectivity[j]);
        bulk_data_->declare_relation(element, node, j);
      }
    }

    bulkData_->modification_end();

    return;
  }

  //----------------------------------------------------------------------------
  //
  // Determine the nodes associated with a face.
  //
  std::vector<Entity*> Topology::getFaceNodes(Entity * entity)
  {
    std::vector<Entity*> face_nodes;

    stk::mesh::PairIterRelation elements = entity->relations(element_rank_);
    // local id for the current face
    unsigned faceId = elements[0].identifier();
    Entity * element = elements[0].entity();
    // number of nodes for the face
    unsigned numFaceNodes = element_topology_.getNodeCount(entity->entity_rank(),
                                                           faceId);

    // Create the ordered list of nodes for the face
    for (int i = 0; i < numFaceNodes; ++i) {
      // map the local node id for the face to the local node id for the element
      unsigned elem_node = element_topology_.getNodeMap(entity->entity_rank(),
                                                        faceId, i);
      // map the local element node id to the global node id
      int element_local_id = element_global_to_local_ids[element->identifier()];
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
      cout << "Nodes of Face " << (face1)->identifier() << ": ";
      for (std::vector<Entity*>::iterator j = face_nodes.begin();
           j != face_nodes.end(); ++j) {
        cout << (*j)->identifier() << ":";
      }
      cout << "\n";

      face_nodes = Topology::getFaceNodes(face2);
      cout << "Nodes of Face " << (face2)->identifier() << ": ";
      for (std::vector<Entity*>::iterator j = face_nodes.begin();
           j != face_nodes.end(); ++j) {
        cout << (*j)->identifier() << ":";
      }
      cout << "\n";

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
      element_topology_.getNodeCount(face1->entity_rank(), 0);

    // Traverse down the graph from the face. The first node of
    // segment $n$ is node $n$ of the face.
    stk::mesh::PairIterRelation face1Relations =
      face1->relations(face1->entity_rank() - 1);
    stk::mesh::PairIterRelation face2Relations =
      face2->relations(face2->entity_rank() - 1);

    std::vector<Entity*> connectivity(2 * numFaceNodes);

    for (int i = 0; i < face1Relations.size(); ++i) {
      Entity * entity1 = face1Relations[i].entity();
      Entity * entity2 = face2Relations[i].entity();
      // If number_dimensions_ = 2, the out edge targets from the
      // faces are nodes
      if (entity1->entity_rank() == node_rank_) {
        connectivity[i] = entity1;
        connectivity[i + numFaceNodes] = entity2;
      }
      // Id number_dimensions_ = 3, the out edge targets from the
      // faces are segments Take the 1st out edge of the segment
      // relation list
      else {
        stk::mesh::PairIterRelation seg1Relations =
          entity1->relations(entity1->entity_rank() - 1);
        stk::mesh::PairIterRelation seg2Relations =
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
    stk::mesh::PairIterRelation relations = 
      entity.relations(entity.entity_rank() + 1);
    subgraph_entity_list.insert(entity.key());
    for (stk::mesh::PairIterRelation::iterator i = relations.begin();
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
    stk::mesh::Selector select_owned_or_shared = meta_data_->locally_owned_part() |
      meta_data_->globally_shared_part();

    stk::mesh::get_selected_entities( select_owned_or_shared,
                                      bulk_data_->buckets( node_rank_ ),
                                      node_list );
    for (std::vector<Entity*>::iterator i = node_list.begin();
         i != node_list.end(); ++i) {
      Entity* entity = *i;
      if (entity_open[entity->key()] == true) {
        open_node_list.push_back(entity);
      }
    }

    bulkData_->modification_begin();

    // Iterate over the open nodes
    for (std::vector<Entity*>::iterator i = open_node_list.begin();
         i != open_node_list.end(); ++i) {
      // Get set of open segments
      Entity * entity = *i;
      stk::mesh::PairIterRelation relations = entity->relations(edge_rank_);
      std::vector<Entity*> open_segment_list;

      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
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

        Subgraph subgraph(bulk_data_, firstEntity, lastEntity, firstEdge,
                          lastEdge, number_dimensions_);

        // Clone open faces
        stk::mesh::PairIterRelation faces = segment->relations(face_rank_);
        std::vector<Entity*> open_face_list;
        // create a list of open faces
        for (stk::mesh::PairIterRelation::iterator k = faces.begin();
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
          Entity * newFace = bulk_data_->get_entity(newFaceKey);

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
      Subgraph subgraph(bulk_data_, firstEntity, lastEntity, firstEdge, lastEdge,
                        number_dimensions_);

      Vertex node = subgraph.globalToLocal(entity->key());
      std::map<Entity*, Entity*> new_connectivity =
        subgraph.splitArticulationPoint(node, entity_open);

      // Update the connectivity
      for (std::map<Entity*, Entity*>::iterator j = new_connectivity.begin();
           j != new_connectivity.end(); ++j) {
        Entity* element = (*j).first;
        Entity* newNode = (*j).second;

        int element_id = element_global_to_local_ids[element->identifier()];
        std::vector<Entity*> & element_connectivity = connectivity_temp_[element_id];
        for (int k = 0; k < element_connectivity.size(); ++k) {
          // Need to subtract 1 from element number as stk indexes from 1
          //   and connectivity_temp indexes from 0
          if (element_connectivity[k] == entity) {
            element_connectivity[k] = newNode;
            // Duplicate the parameters of old node to new node
            bulk_data_->copy_entity_fields(*entity, *newNode);
          }
        }
      }
    }

    bulk_data_->modification_end();
    bulk_data_->modification_begin();

    // Create the cohesive connectivity
    int j = 1;
    for (std::set<std::pair<Entity*, Entity*> >::iterator i =
           fractured_faces_.begin(); i != fractured_faces_.end(); ++i, ++j) {
      Entity * face1 = (*i).first;
      Entity * face2 = (*i).second;
      std::vector<Entity*> cohesive_connectivity;
      cohesive_connectivity = Topology::createCohesiveConnectivity(face1, face2);

      // Output connectivity for testing purposes
      cout << "Cohesive Element " << j << ": ";
      for (int j = 0; j < cohesive_connectivity.size(); ++j) {
        cout << cohesive_connectivity[j]->identifier() << ":";
      }
      cout << "\n";
    }

    bulkData_->modification_end();
    return;
  }
#endif

  void topology::splitOpenFaces(std::map<EntityKey, bool> & global_entity_open)
  {
    std::vector<Entity*> open_node_list; // Global open node list
    
    std::cout << " \n\nGlobal stuff in fracture_boundary\n\n" << std::endl;
    
    // Build list of open nodes (global)

    std::pair<EntityKey,bool> me; // what a map<EntityKey, bool> is made of

    BOOST_FOREACH(me, global_entity_open) {

      if(stk::mesh::entity_rank( me.first) == nodeRank){

        stk::mesh::Entity *entity = bulkData_->get_entity(me.first);
        std::cout << "Found open node: " << entity->identifier() << " belonging to pe: " << entity->owner_rank() << std::endl;
        open_node_lst.push_back(entity);
      }
    }

    bulkData_->modification_begin();

    // Iterate over the open nodes
    for (std::vector<Entity*>::iterator i = open_node_list.begin();
         i != open_node_list.end(); ++i) {
      // Get set of open segments
      Entity * entity = *i;
      stk::mesh::PairIterRelation relations = entity->relations(edge_rank_);
      std::vector<Entity*> open_segment_list;

      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
           j != relations.end(); ++j) {
        Entity & source = *j->entity();
        if (global_entity_open[source.key()] == true) {
std::cout << "Found open segment: " << source.identifier() << " belonging to pe: " << source.owner_rank() << std::endl;
          open_segment_lst.push_back(&source);
        }
      }

      // Iterate over the open segments
      for (std::vector<Entity*>::iterator j = open_segment_list.begin();
           j != open_segment_list.end(); ++j) {
        Entity * segment = *j;

        // Create star of segment
        std::set<EntityKey> subgraph_entity_lst;
        std::set<stkEdge, EdgeLessThan> subgraph_edge_lst;
        topology::star(subgraph_entity_lst, subgraph_edge_lst, *segment);

        // Iterators
        std::set<EntityKey>::iterator firstEntity = subgraph_entity_list.begin();
        std::set<EntityKey>::iterator lastEntity = subgraph_entity_list.end();
        std::set<stkEdge>::iterator firstEdge = subgraph_edge_list.begin();
        std::set<stkEdge>::iterator lastEdge = subgraph_edge_list.end();

        Subgraph subgraph(bulkData_, firstEntity, lastEntity, firstEdge, 
            lastEdge, number_dimensions_);

        // Clone open faces
        stk::mesh::PairIterRelation faces = segment->relations(faceRank);
        std::vector<Entity*> open_face_lst;

        // create a list of open faces
        for (stk::mesh::PairIterRelation::iterator k = faces.begin();
             k != faces.end(); ++k) {
          Entity & source = *k->entity();
          if (global_entity_open[source.key()] == true) {
std::cout << "Found open face: " << source.identifier() << " belonging to pe: " << source.owner_rank() << std::endl;
            open_face_lst.push_back(&source);
          }
        }
std::cout << "\n\n\n\n\n" << std::endl;

        // Iterate over the open faces
        for (std::vector<Entity*>::iterator k = open_face_list.begin();
             k != open_face_list.end(); ++k) {
          Entity * face = *k;
          Vertex faceVertex = subgraph.globalToLocal(face->key());
          Vertex newFaceVertex;
          subgraph.clone_boundary_entity(faceVertex, newFaceVertex,
              global_entity_open);
          EntityKey newFaceKey = subgraph.localToGlobal(newFaceVertex);
          Entity * newFace = bulk_data_->get_entity(newFaceKey);

          // add original and new faces to the fractured face list
          fractured_faces_.insert(std::make_pair(face, newFace));

        }

        // Split the articulation point (current segment)
        Vertex segmentVertex = subgraph.global_to_local(segment->key());
        std::cout << "Calling split_articulation_point with segmentVertex: " << std::endl;
        subgraph.split_articulation_point(segmentVertex, global_entity_open);
        std::cout << "done Calling split_articulation_point with segmentVertex: " << std::endl;
      }
      // All open faces and segments have been dealt with. Split the node articulation point
      // Create star of node
      std::set<EntityKey> subgraph_entity_list;
      std::set<stkEdge, EdgeLessThan> subgraph_edge_list;
      Topology::createStar(subgraph_entity_list, subgraph_edge_list, *entity);
      // Iterators
      std::set<EntityKey>::iterator firstEntity = subgraph_entity_lst.begin();
      std::set<EntityKey>::iterator lastEntity = subgraph_entity_lst.end();
      std::set<stkEdge>::iterator firstEdge = subgraph_edge_lst.begin();
      std::set<stkEdge>::iterator lastEdge = subgraph_edge_lst.end();
      Subgraph subgraph(bulkData_, firstEntity, lastEntity, firstEdge, 
          lastEdge, number_dimensions_);

      Vertex node = subgraph.global_to_local(entity->key());
std::cout << "Calling split_articulation_point with node: " << std::endl;
      std::map<Entity*, Entity*> new_connectivity =
          subgraph.split_articulation_point(node, global_entity_open);
std::cout << "done Calling split_articulation_point with node: " << std::endl;

      // Update the connectivity
      for (std::map<Entity*, Entity*>::iterator j = new_connectivity.begin();
           j != new_connectivity.end(); ++j) {
        Entity* element = (*j).first;
        Entity* newNode = (*j).second;

          // Need to subtract 1 from element number as stk indexes from 1
          //   and connectivity_temp indexes from 0
//        int id = static_cast<int>(element->identifier());
        int element_local_id = element_global_to_local_ids[element->identifier()];
//        std::vector<Entity*> & element_connectivity = connectivity_temp[id - 1];
        std::vector<Entity*> & element_connectivity = connectivity_temp[element_local_id];
        for (int k = 0; k < element_connectivity.size(); ++k) {
          if (element_connectivity[k] == entity) {
            element_connectivity[k] = newNode;
            // Duplicate the parameters of old node to new node
            bulk_data_->copy_entity_fields(*entity, *newNode);
          }
        }
      }
    }

    bulkData_->modification_end();




    bulkData_->modification_begin();

    // Create the cohesive connectivity
    int j = 1;
    for (std::set<std::pair<Entity*, Entity*> >::iterator i =
           fractured_faces_.begin(); i != fractured_faces_.end(); ++i, ++j) {
      Entity * face1 = (*i).first;
      Entity * face2 = (*i).second;
      std::vector<Entity*> cohesive_connectivity;
      cohesive_connectivity = Topology::createCohesiveConnectivity(face1, face2);

      // Output connectivity for testing purposes
      cout << "Cohesive Element " << j << ": ";
      for (int j = 0; j < cohesive_connectivity.size(); ++j) {
        cout << cohesive_connectivity[j]->identifier() << ":";
      }
      cout << "\n";
    }

    bulkData_->modification_end();

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
    std::vector<Entity*> boundary_lst;
//    stk::mesh::Selector select_owned_or_shared = metaData_->locally_owned_part() | metaData_->globally_shared_part();
    stk::mesh::Selector select_owned = metaData_->locally_owned_part();

//    stk::mesh::get_selected_entities( select_owned_or_shared ,
    stk::mesh::get_selected_entities( select_owned,
				    bulkData_->buckets( number_dimensions_ - 1 ) ,
				    boundary_lst );

    // Probability that fracture_criterion will return true.
    double p = 1.0;

    // Iterate over the boundary entities
    for (int i = 0; i < boundary_list.size(); ++i) {
      Entity& entity = *(boundary_list[i]);
      bool is_open = fracture_criterion_->computeFractureCriterion(entity, p);
      // If the criterion is met, need to set lower rank entities
      //   open as well
      if (is_open == true && number_dimensions_ == 3) {
        entity_open[entity.key()] = true;
        stk::mesh::PairIterRelation segments = entity.relations(
                                                                entity.entity_rank() - 1);
        // iterate over the segments
        for (int j = 0; j < segments.size(); ++j) {
          Entity & segment = *(segments[j].entity());
          entity_open[segment.key()] = true;
          stk::mesh::PairIterRelation nodes = segment.relations(
                                                                segment.entity_rank() - 1);
          // iterate over nodes
          for (int k = 0; k < nodes.size(); ++k) {
            Entity& node = *(nodes[k].entity());
            entity_open[node.key()] = true;
          }
        }
      }
      // If the mesh is 2D
      else if (is_open == true && number_dimensions_ == 2) {
        entity_open[entity.key()] = true;
        stk::mesh::PairIterRelation nodes = entity.relations(
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

  void Topology::setEntitiesOpen(const std::vector<stk::mesh::Entity*>& fractured_edges,
                                 std::map<EntityKey, bool>& entity_open)
  {

    entity_open.clear();

    // Iterate over the boundary entities
    for (int i = 0; i < fractured_edges.size(); ++i) {
      Entity& entity = *(fractured_edges[i]);
      // Need to set lower rank entities
      //   open as well
      if (number_dimensions_ == 3) {
        entity_open[entity.key()] = true;
        stk::mesh::PairIterRelation segments = entity.relations(
                                                                entity.entity_rank() - 1);
        // iterate over the segments
        for (int j = 0; j < segments.size(); ++j) {
          Entity & segment = *(segments[j].entity());
          entity_open[segment.key()] = true;
          stk::mesh::PairIterRelation nodes = segment.relations(
                                                                segment.entity_rank() - 1);
          // iterate over nodes
          for (int k = 0; k < nodes.size(); ++k) {
            Entity& node = *(nodes[k].entity());
            entity_open[node.key()] = true;
          }
        }
      }
      // If the mesh is 2D
      else if (number_dimensions_ == 2) {
        entity_open[entity.key()] = true;
        stk::mesh::PairIterRelation nodes = entity.relations(
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

  //----------------------------------------------------------------------------
  //
  // Default constructor
  //
  Subgraph::Subgraph()
  {
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
    stk::mesh::EntityVector new_entity;
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
  Subgraph::communicate_and_create_shared_entities(stk::mesh::Entity   & node,
                                          stk::mesh::EntityKey   new_node_key){

    stk::CommAll comm(bulkData_->parallel());

    {
    stk::mesh::PairIterEntityComm entity_comm = node.sharing();
  
    for (; entity_comm.first != entity_comm.second; ++entity_comm.first) {
  
        unsigned proc = entity_comm.first->proc;
        comm.send_buffer(proc).pack<stk::mesh::EntityKey>(node.key())
                              .pack<stk::mesh::EntityKey>(new_node_key);
  
    }
    }

    comm.allocate_buffers(bulkData_->parallel_size()/4 );
  
    {
    stk::mesh::PairIterEntityComm entity_comm = node.sharing();
  
    for (; entity_comm.first != entity_comm.second; ++entity_comm.first) {
  
        unsigned proc = entity_comm.first->proc;
        comm.send_buffer(proc).pack<stk::mesh::EntityKey>(node.key())
                              .pack<stk::mesh::EntityKey>(new_node_key);
  
    }
    }
  
    comm.communicate();
  
    const stk::mesh::PartVector no_parts;
  
    for (size_t process = 0; process < bulkData_->parallel_size(); ++process) {
      stk::mesh::EntityKey old_key;
      stk::mesh::EntityKey new_key;
  
      while ( comm.recv_buffer(process).remaining()) {
  
        comm.recv_buffer(process).unpack<stk::mesh::EntityKey>(old_key)
                                 .unpack<stk::mesh::EntityKey>(new_key);
  
        stk::mesh::Entity * new_entity = & bulkData_->declare_entity(new_key.rank(), new_key.id(), no_parts);
//std::cout << " Proc: " << bulkData_->parallel_rank() << " created entity: (" << new_entity->identifier() << ", " <<
//new_entity->entity_rank() << ")." << std::endl;
  
      }
    }

  }

  void
  Subgraph::bcast_key(unsigned root, stk::mesh::EntityKey&   node_key){

    stk::CommBroadcast comm(bulkData_->parallel(), root);

    unsigned rank = bulkData_->parallel_rank();

    if(rank == root)

      comm.send_buffer().pack<stk::mesh::EntityKey>(node_key);
  
    comm.allocate_buffer();

    if(rank == root)

      comm.send_buffer().pack<stk::mesh::EntityKey>(node_key);
  
    comm.communicate();
  
    comm.recv_buffer().unpack<stk::mesh::EntityKey>(node_key);

  }

  Vertex Subgraph::cloneVertex(Vertex & vertex)
  {

    // Get the vertex rank
    EntityRank vertexRank = Subgraph::get_vertex_rank(vertex);
    EntityKey vertKey = Subgraph::local_to_global(vertex);

    // Determine which processor should create the new vertex
    Entity *  oldVertex = bulkData_->get_entity(vertKey);

//    if(!oldVertex){
//std::cout << "oldVertex is NULL at line " << __LINE__ << " in file " << __FILE__ << std::endl;
//    }

    // For now, the owner of the new vertex is the same as the owner of the old one
    int owner_proc = oldVertex->owner_rank();

    // The owning processor inserts a new vertex into the stk mesh
    // First have to request a new entity of rank N
    std::vector<size_t> requests(numDim_ + 1, 0); // number of entity ranks. 1 + number of dimensions
    stk::mesh::EntityVector newEntity;
    const stk::mesh::PartVector no_parts;

    int my_proc = bulkData_->parallel_rank();
    int source;
    Entity *globalVertex;
    stk::mesh::EntityKey globalVertexKey;
    stk::mesh::EntityKey::raw_key_type gvertkey;

    if(my_proc == owner_proc){

      // Insert the vertex into the stk mesh
      // First have to request a new entity of rank N
      requests[vertexRank] = 1;

      // have stk build the new entity, then broadcast the key

      bulkData_->generate_new_entities(requests, newEntity);
      globalVertex = newEntity[0];
//std::cout << " Proc: " << bulkData_->parallel_rank() << " created entity: (" << globalVertex->identifier() << ", " <<
//globalVertex->entity_rank() << ")." << std::endl;
      globalVertexKey = globalVertex->key();
      gvertkey = globalVertexKey.raw_key();

    }
    else {

      // All other processors do a no-op

      bulkData_->generate_new_entities(requests, newEntity);

    }

    Subgraph::bcast_key(owner_proc, globalVertexKey);

    if(my_proc != owner_proc){ // All other processors receive the key

      // Get the vertex from stk

        const stk::mesh::PartVector no_parts;
        stk::mesh::Entity * new_entity = & bulkData_->declare_entity(globalVertexKey.rank(), globalVertexKey.id(), no_parts);

    }

    // Insert the vertex into the subgraph
    Vertex localVertex = boost::add_vertex(*this);

    // Update maps
    localGlobalVertexMap.insert(
        std::map<Vertex, EntityKey>::value_type(localVertex,
            globalVertexKey));
    globalLocalVertexMap.insert(
        std::map<EntityKey, Vertex>::value_type(globalVertexKey,
            localVertex));

    // store entity rank to the vertex property
    VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);
    boost::put(vertexPropertyMap, localVertex, vertexRank);

    return localVertex;
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
    stk::mesh::PairIterRelation relations = entity->relations();
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
      cout << "add edge:" << global_source_vertex->entity_rank() << ","
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
    boost::graph_traits<BoostGraph>::degree_size_type num_in_edges =
      boost::in_degree(vertex, *this);
    if (num_in_edges != 2) return;

    // Check that vertex = open
    EntityKey vertex_key = Subgraph::localToGlobal(vertex);
    assert(entity_open[vertex_key]==true);

    // Get the vertex rank
//    EntityRank vertexRank = Subgraph::get_vertex_rank(vertex);

    // Create a new vertex of same rank as vertex
//    newVertex = Subgraph::add_vertex(vertexRank);
    newVertex = Subgraph::cloneVertex(vertex);

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
    std::vector<Vertex> newVertex;
    for (int i = 0; i < numComponents - 1; ++i) {
//      Vertex newVert = Subgraph::add_vertex(vertexRank);
      Vertex newVert = Subgraph::cloneVertex(vertex);
      newVertex.push_back(newVert);
    }

    // create a map of elements to new node numbers
    // only do this if the input vertex is a node (don't require otherwise)
    if (vertex_rank == 0) {
      for (std::map<Vertex, int>::iterator i = components.begin();
           i != components.end(); ++i) {
        int componentNum = (*i).second;
        Vertex currentVertex = (*i).first;
        EntityRank currentRank = Subgraph::getVertexRank(currentVertex);
        // Only add to map if the vertex is an element
        if (currentRank == num_dim_ && componentNum != 0) {
          Entity* element = 
            bulk_data_->get_entity(Subgraph::localToGlobal(currentVertex));
          Entity* newNode = 
            bulk_data_->
            get_entity(Subgraph::localToGlobal(new_vertex[componentNum - 1]));
          new_connectivity.
            insert(std::map<Entity*, Entity*>::value_type(element, newNode));
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
    stk::mesh::PairIterRelation original_relations = 
      original_entity.relations(original_entity.entity_rank() - 1);
    for (int i = 0; i < original_relations.size(); ++i) {
      stk::mesh::PairIterRelation new_relations = 
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

    cout << "Write graph to graphviz dot file\n";

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
      cout << "Unable to open graphviz output file 'output.dot'\n";

    return;
  }

} // namespace LCM

