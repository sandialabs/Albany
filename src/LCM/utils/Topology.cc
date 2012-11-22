//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Topology.h"

namespace LCM {

  /**
   * \brief Default constructor for topology (private to topology class)
   */
  topology::topology() :
      number_dimensions_(0), discretization_ptr_(Teuchos::null)
  {
    return;
  }

  topology::topology(std::string const & input_file,
      std::string const & output_file)
  {
    Teuchos::RCP<Teuchos::ParameterList> disc_params = rcp(
        new Teuchos::ParameterList("params"));

    //set Method to Exodus and set input file name
    disc_params->set<std::string>("Method", "Exodus");
    disc_params->set<std::string>("Exodus Input File Name", input_file);
    disc_params->set<std::string>("Exodus Output File Name", output_file);
    //disc_params->print(std::cout);

    Teuchos::RCP<Epetra_Comm> communicator =
        Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    Albany::DiscretizationFactory disc_factory(disc_params, false, communicator);

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs =
        disc_factory.createMeshSpecs();

    Teuchos::RCP<Albany::StateInfoStruct> stateInfo = Teuchos::rcp(
        new Albany::StateInfoStruct());

    discretization_ptr_ = disc_factory.createDiscretization(3, stateInfo);

    topology::create_discretization();

    // Create the full mesh representation. This must be done prior to the adaptation query. We are
    // reading the mesh from a file so do it here.
    topology::graph_initialization();

    // Fracture the mesh randomly
    fracObject = Teuchos::rcp(new GenericFractureCriterion(number_dimensions_, elementRank));

    return;
  }

  /**
   * \brief Create mesh data structure
   *
   * \param[in] Albany discretization object
   *
   * Use if already have an Albany mesh object
   */
  topology::topology(
      Teuchos::RCP<Albany::AbstractDiscretization> & discretization_ptr)
  {

    discretization_ptr_ = discretization_ptr;

    topology::create_discretization();

    // This constructor assumes that the full mesh graph has been previously created.
    //topology::graph_initialization();

    // Fracture the mesh randomly
    fracObject = Teuchos::rcp(new GenericFractureCriterion(number_dimensions_, elementRank));

    return;
  }

  /**
   * \brief Create mesh data structure
   *
   * \param[in] Albany discretization object
   * \param[in] Fracture criterion object
   *
   * Use if already have an Albany mesh object, and want to fracture the mesh based
   * on a user-created fracture criterion object.
   */
  topology::topology(Teuchos::RCP<Albany::AbstractDiscretization>& discretization_ptr,
    Teuchos::RCP<AbstractFractureCriterion>& frac) :
	    discretization_ptr_(discretization_ptr),
      fracObject(frac)
  {

	  topology::create_discretization();

    // This function assumes that the full mesh graph has already been created.
    //topology::graph_initialization(); 

	  return;

  }

  /**
   * \brief Create Albany discretization
   */
  void topology::create_discretization()
  {

    // Need to access the bulkData and metaData classes in the mesh data structure
    Albany::STKDiscretization & stk_discretization =
        static_cast<Albany::STKDiscretization &>(*discretization_ptr_);

    stkMeshStruct_ = stk_discretization.getSTKMeshStruct();

    bulkData_ = stkMeshStruct_->bulkData;
    stk::mesh::fem::FEMMetaData * metaData = stkMeshStruct_->metaData;

    // The entity ranks
    nodeRank = metaData->NODE_RANK;
    edgeRank = metaData->EDGE_RANK;
    faceRank = metaData->FACE_RANK;
    elementRank = metaData->element_rank();
    number_dimensions_ = stkMeshStruct_->numDim;

    // Get the topology of the elements. NOTE: Assumes one element type in
    //   mesh.
    element_topology = stk::mesh::fem::get_cell_topology(
        *(bulkData_->get_entity(elementRank, 1)));

    return;

  }

  /**
   * \brief Output relations associated with entity
   *        The entity may be of any rank
   * \param[in] entity
   */
  void topology::disp_relation(Entity const & entity)
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

  /**
   * \brief Output relations of rank entityRank associated with entity
   *        the entity may be of any rank
   * \param[in] entity
   * \param[in] the rank of the entity
   */
  void topology::disp_relation(Entity const & entity,
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

  /**
   * \brief Output the mesh connectivity
   *
   * Outputs the nodal connectivity of the elements as stored by
   * bulkData. Assumes that relationships between the elements and
   * nodes exist.
   */

  void topology::disp_connectivity()
  {
    // Create a list of element entities
    std::vector<Entity*> element_list;
    stk::mesh::get_entities(*(bulkData_), elementRank, element_list);

    // Loop over the elements
    const int number_of_elements = element_list.size();

    for (int i = 0; i < number_of_elements; ++i) {

      stk::mesh::PairIterRelation relations = element_list[i]->relations(
          nodeRank);

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

//
// \brief Output the graph associated with the mesh to graphviz .dot
// file for visualization purposes. No need for entity_open map
// for this version
//
// \param[in] output file
// \param[in] map of entity and boolean value is open
//
// To create final output figure, run command below from terminal:
//   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
//
  void topology::output_to_graphviz(std::string & gviz_output)
  {
    std::map<EntityKey, bool> entity_open;
    output_to_graphviz(gviz_output, entity_open);
    return;
  }

  /**
   * \brief Output the graph associated with the mesh to graphviz .dot
   * file for visualization purposes.
   *
   * \param[in] output file
   * \param[in] map of entity and boolean value is open
   *
   * If fracture criterion for entity is satisfied, the entity and all
   * associated lower order entities are marked open. All open entities are
   * displayed as such in output file.
   *
   * To create final output figure, run command below from terminal:
   *   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
   */
  void topology::output_to_graphviz(std::string & gviz_output,
      std::map<EntityKey, bool> & entity_open)
  {
    // Open output file
    std::ofstream gviz_out;
    gviz_out.open(gviz_output.c_str(), std::ios::out);

    cout << "Write graph to graphviz dot file\n";

    if (gviz_out.is_open()) {
      // Write beginning of file
      gviz_out << "digraph mesh {\n" << "  node [colorscheme=paired12]\n"
          << "  edge [colorscheme=paired12]\n";

      std::vector<Entity*> entity_lst;
      stk::mesh::get_entities(*(bulkData_), elementRank, entity_lst);

      std::vector<std::vector<Entity*> > relation_lst;
      std::vector<int> relation_local_id;

      // Elements
      for (int i = 0; i < entity_lst.size(); ++i) {
        Entity & entity = *(entity_lst[i]);
        stk::mesh::PairIterRelation relations = entity.relations();

        gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
            << "\" [label=\"Element " << entity.identifier()
            << "\",style=filled,fillcolor=\"8\"]\n";
        for (int j = 0; j < relations.size(); ++j) {
          if (relations[j].entity_rank() < entity.entity_rank()) {
            std::vector<Entity*> temp;
            temp.push_back(&entity);
            temp.push_back(relations[j].entity());
            relation_lst.push_back(temp);
            relation_local_id.push_back(relations[j].identifier());
          }
        }
      }

      stk::mesh::get_entities(*(bulkData_), faceRank, entity_lst);

      // Faces
      for (int i = 0; i < entity_lst.size(); ++i) {
        Entity & entity = *(entity_lst[i]);
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
            relation_lst.push_back(temp);
            relation_local_id.push_back(relations[j].identifier());
          }
        }
      }

      stk::mesh::get_entities(*(bulkData_), edgeRank, entity_lst);

      // Edges
      for (int i = 0; i < entity_lst.size(); ++i) {
        Entity & entity = *(entity_lst[i]);
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
            relation_lst.push_back(temp);
            relation_local_id.push_back(relations[j].identifier());
          }
        }
      }

      stk::mesh::get_entities(*(bulkData_), nodeRank, entity_lst);

      // Nodes
      for (int i = 0; i < entity_lst.size(); ++i) {
        Entity & entity = *(entity_lst[i]);

        if (entity_open[entity.key()] == true)
          gviz_out << "  \"" << entity.identifier() << "_"
              << entity.entity_rank() << "\" [label=\"Node "
              << entity.identifier() << "\",style=filled,fillcolor=\"5\"]\n";
        else
          gviz_out << "  \"" << entity.identifier() << "_"
              << entity.entity_rank() << "\" [label=\"Node "
              << entity.identifier() << "\",style=filled,fillcolor=\"6\"]\n";
      }

      for (int i = 0; i < relation_lst.size(); ++i) {
        std::vector<Entity*> temp = relation_lst[i];
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

  /**
   * \brief Initializes the default stk mesh object needed by class.
   *
   * Creates the full mesh representation of the mesh. Default stk mesh
   * object has only elements and nodes. Function will delete unneeded
   * relations between as described in topology::remove_extra_relations().
   *
   * \attention Function must be called before mesh modification begins.
   *
   * \attention Call function once. Creation of extra entities and relations
   * is slow.
   */
  void topology::graph_initialization()
  {
    stk::mesh::PartVector add_parts;
    stk::mesh::create_adjacent_entities(*(bulkData_), add_parts);

    bulkData_->modification_begin();
    topology::remove_extra_relations();
    bulkData_->modification_end();

    return;
  }

  /**
   * \brief Removes unneeded relations from the mesh.
   *
   * stk::mesh::create_adjacent_entities creates full mesh representation of
   * the mesh instead of the default of only the elements and nodes. All
   * entities created by the function are connected through relationships.
   * Graph algorithms require relationships to only exist between entities
   * separated by one degree, e.g. elements and faces in a 3D graph.
   * Function removes all other relationships.
   *
   * \note Valid for 2D and 3D meshes.
   */
#if 1 // GAH - this is the original
  void topology::remove_extra_relations()
  {
    std::vector<Entity*> element_lst;
    stk::mesh::get_entities(*(bulkData_), elementRank, element_lst);

    // Remove extra relations from element
    for (int i = 0; i < element_lst.size(); ++i) {
      Entity & element = *(element_lst[i]);
      stk::mesh::PairIterRelation relations = element.relations();
      std::vector<Entity*> del_relations;
      std::vector<int> del_ids;
      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
          j != relations.end(); ++j) {
        // remove all relationships from element unless to faces(segments
        //   in 2D) or nodes
        if (j->entity_rank() != elementRank - 1
            && j->entity_rank() != nodeRank) {
          del_relations.push_back(j->entity());
          del_ids.push_back(j->identifier());
        }
      }
      for (int j = 0; j < del_relations.size(); ++j) {
        Entity & entity = *(del_relations[j]);
        bulkData_->destroy_relation(element, entity, del_ids[j]);
      }
    };

    if (elementRank == 3) {
      // Remove extra relations from face
      std::vector<Entity*> face_lst;
      stk::mesh::get_entities(*(bulkData_), elementRank - 1, face_lst);
      EntityRank entityRank = face_lst[0]->entity_rank();
      for (int i = 0; i < face_lst.size(); ++i) {
        Entity & face = *(face_lst[i]);
        stk::mesh::PairIterRelation relations = face_lst[i]->relations();
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
          bulkData_->destroy_relation(face, entity, del_ids[j]);
        }
      }
    }

    return;
  }
#else

void
topology::remove_extra_relations()
{
	std::vector<Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData_),elementRank,element_lst);

	// Remove extra relations from element
	for (int i = 0; i < element_lst.size(); ++i){
		Entity & element = *(element_lst[i]);
		stk::mesh::PairIterRelation relations = element.relations();
		std::vector<Entity*> del_relations;
		std::vector<int> del_ids;
		for (stk::mesh::PairIterRelation::iterator j = relations.begin();
				j != relations.end(); ++j){
			// remove all relationships from element unless to faces(segments
			//   in 2D) or nodes
//			if (j->entity_rank() != elementRank-1 && j->entity_rank() != nodeRank){
// GAH THIS NEEDS TO BE UNCOMMENTED!!!!!
			if (j->entity_rank() != nodeRank){
				del_relations.push_back(j->entity());
				del_ids.push_back(j->identifier());
			}
		}
		for (int j = 0; j < del_relations.size(); ++j){
			Entity & entity = *(del_relations[j]);
			bulkData_->destroy_relation(element,entity,del_ids[j]);
		}
	};

	if (elementRank == 3){
		// Remove extra relations from face
		std::vector<Entity*> face_lst;
		stk::mesh::get_entities(*(bulkData_),elementRank-1,face_lst);
		EntityRank entityRank = face_lst[0]->entity_rank();
		for (int i = 0; i < face_lst.size(); ++i){
			Entity & face = *(face_lst[i]);
			stk::mesh::PairIterRelation relations = face_lst[i]->relations();
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
				bulkData_->destroy_relation(face,entity,del_ids[j]);
			}
		}
	}

	return;
}

#endif

  /**
   * \brief Creates temporary nodal connectivity for the elements and removes the
   * relationships between the elements and nodes.
   *
   * \attention Must be called every time before mesh topology changes begin.
   */
  void topology::remove_node_relations()
  {
    // Create the temporary connectivity array
    std::vector<Entity*> element_lst;
    stk::mesh::get_entities(*(bulkData_), elementRank, element_lst);

    bulkData_->modification_begin();
    for (int i = 0; i < element_lst.size(); ++i) {
      stk::mesh::PairIterRelation nodes = element_lst[i]->relations(nodeRank);
      std::vector<Entity*> temp;
      for (int j = 0; j < nodes.size(); ++j) {
        Entity* node = nodes[j].entity();
        temp.push_back(node);
      }
      connectivity_temp.push_back(temp);

      for (int j = 0; j < temp.size(); ++j) {
        bulkData_->destroy_relation(*(element_lst[i]), *(temp[j]), j);
      }
    }

    bulkData_->modification_end();

    return;
  }

  /**
   * \brief After mesh manipulations are complete, need to recreate a stk
   * mesh understood by Albany_STKDiscretization.
   *
   * Recreates the nodal connectivity using connectivity_temp.
   *
   * \attention must be called before mesh modification has ended
   */
  void topology::graph_cleanup()
  {
    std::vector<Entity*> element_lst;
    stk::mesh::get_entities(*(bulkData_), elementRank, element_lst);

//    bulkData_->modification_begin(); // need to comment GAH?

    // Add relations from element to nodes
    for (int i = 0; i < element_lst.size(); ++i) {
      Entity & element = *(element_lst[i]);
      std::vector<Entity*> element_connectivity = connectivity_temp[i];
      for (int j = 0; j < element_connectivity.size(); ++j) {
        Entity & node = *(element_connectivity[j]);
        bulkData_->declare_relation(element, node, j);
      }
    }

    return;
  }

  /**
   * \brief Determine the nodes associated with a face.
   *
   * \param[in] Face entity
   * \return vector of nodes for the face
   *
   * Return an ordered list of nodes which describe the input face. In 2D,
   * the face of the element is a line segment. In 3D, the face is a surface.
   * Generalized for all element types valid in stk_mesh. Valid in 2D and 3D.
   *
   * \attention Assumes all mesh elements are same type.
   */
  std::vector<Entity*> topology::get_face_nodes(Entity * entity)
  {
    std::vector<Entity*> face_nodes;

    stk::mesh::PairIterRelation elements = entity->relations(elementRank);
    // local id for the current face
    unsigned faceId = elements[0].identifier();
    Entity * element = elements[0].entity();
    // number of nodes for the face
    unsigned numFaceNodes = element_topology.getNodeCount(entity->entity_rank(),
        faceId);

    // Create the ordered list of nodes for the face
    for (int i = 0; i < numFaceNodes; ++i) {
      // map the local node id for the face to the local node id for the element
      unsigned elemNode = element_topology.getNodeMap(entity->entity_rank(),
          faceId, i);
      // map the local element node id to the global node id
      Entity* node = connectivity_temp[element->identifier() - 1][elemNode];
      face_nodes.push_back(node);
    }

    return face_nodes;
  }

  /**
   * \brief Creates a mesh of the fractured surfaces only.
   *
   *  Outputs the mesh as an exodus file for visual representation of split faces.
   *
   *  \todo output the exodus file
   */
  void topology::output_surface_mesh()
  {
    for (std::set<std::pair<Entity*, Entity*> >::iterator i =
        fractured_face.begin(); i != fractured_face.end(); ++i) {
      Entity * face1 = (*i).first;
      Entity * face2 = (*i).second;
      // create an ordered list of nodes for the faces
      // For now, output the face nodes. TODO: replace with mesh output code
      std::vector<Entity*> face_nodes = topology::get_face_nodes(face1);
      cout << "Nodes of Face " << (face1)->identifier() << ": ";
      for (std::vector<Entity*>::iterator j = face_nodes.begin();
          j != face_nodes.end(); ++j) {
        cout << (*j)->identifier() << ":";
      }
      cout << "\n";

      face_nodes = topology::get_face_nodes(face2);
      cout << "Nodes of Face " << (face2)->identifier() << ": ";
      for (std::vector<Entity*>::iterator j = face_nodes.begin();
          j != face_nodes.end(); ++j) {
        cout << (*j)->identifier() << ":";
      }
      cout << "\n";

    }
    return;
  }

  /**
   * \brief Create cohesive connectivity
   *
   * \param[in] Face 1
   * \param[in] Face 2
   * \return Cohesive connectivity
   *
   * Given the two faces after insertion process, create the connectivity
   * of the cohesive element.
   *
   * \attention Assumes that all elements have the same topology
   */
  std::vector<Entity*> topology::create_cohesive_conn(Entity* face1,
      Entity* face2)
  {
    // number of nodes for the face
    unsigned numFaceNodes = element_topology.getNodeCount(face1->entity_rank(),
        0);

    // Traverse down the graph from the face. The first node of segment $n$ is
    // node $n$ of the face.
    stk::mesh::PairIterRelation face1Relations = face1->relations(
        face1->entity_rank() - 1);
    stk::mesh::PairIterRelation face2Relations = face2->relations(
        face2->entity_rank() - 1);

    std::vector<Entity*> connectivity(2 * numFaceNodes);

    for (int i = 0; i < face1Relations.size(); ++i) {
      Entity * entity1 = face1Relations[i].entity();
      Entity * entity2 = face2Relations[i].entity();
      // If number_dimensions_ = 2, the out edge targets from the faces are nodes
      if (entity1->entity_rank() == nodeRank) {
        connectivity[i] = entity1;
        connectivity[i + numFaceNodes] = entity2;
      }
      // Id number_dimensions_ = 3, the out edge targets from the faces are segments
      // Take the 1st out edge of the segment relation list
      else {
        stk::mesh::PairIterRelation seg1Relations = entity1->relations(
            entity1->entity_rank() - 1);
        stk::mesh::PairIterRelation seg2Relations = entity2->relations(
            entity2->entity_rank() - 1);

        // Check for the correct node to add to the connectivity vector
        // Each node should be used once.
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

  /**
   * \brief Create vectors describing the vertices and edges of the star of an entity
   *   in the stk mesh.
   *
   *   \param list of entities in the star
   *   \param list of edges in the star
   *   \param[in] source entity of the star
   *
   *   The star of a graph vertex is defined as the vertex and all higher order
   *   vertices which are connected to it when traversing up the graph from the
   *   input vertex.
   *
   *   \attention Valid for entities of all ranks
   */
  void topology::star(std::set<EntityKey> & subgraph_entity_lst,
      std::set<stkEdge, EdgeLessThan> & subgraph_edge_lst, Entity & entity)
  {
    stk::mesh::PairIterRelation relations = entity.relations(
        entity.entity_rank() + 1);
    subgraph_entity_lst.insert(entity.key());
    for (stk::mesh::PairIterRelation::iterator i = relations.begin();
        i != relations.end(); ++i) {
      stk::mesh::Relation relation = *i;
      Entity & source = *(relation.entity());
      stkEdge edge;
      edge.source = source.key();
      edge.target = entity.key();
      edge.localId = relation.identifier();
      subgraph_edge_lst.insert(edge);
      topology::star(subgraph_entity_lst, subgraph_edge_lst, source);
    }

    return;
  }

  /**
   * \brief Fractures all open boundary entities of the mesh.
   *
   * \param[in] map of entity and boolean value is entity open
   *
   * Iterate through the faces of the mesh and split into two faces if marked
   * as open. The elements associated with an open face are separated. All
   * lower order entities of the face are updated for a consistent mesh.
   *
   * \todo generalize the function for 2D meshes
   */
  void topology::fracture_boundary(std::map<EntityKey, bool> & entity_open)
  {
    int numfractured = 0; //counter for number of fractured faces

    // Get set of open nodes
    std::vector<Entity*> node_lst; //all nodes
    std::vector<Entity*> open_node_lst; //only the open nodes
    stk::mesh::get_entities(*bulkData_, nodeRank, node_lst);

    for (std::vector<Entity*>::iterator i = node_lst.begin();
        i != node_lst.end(); ++i) {
      Entity* entity = *i;
      if (entity_open[entity->key()] == true) {
        open_node_lst.push_back(entity);
      }
    }

    // Iterate over the open nodes
    for (std::vector<Entity*>::iterator i = open_node_lst.begin();
        i != open_node_lst.end(); ++i) {
      // Get set of open segments
      Entity * entity = *i;
      stk::mesh::PairIterRelation relations = entity->relations(edgeRank);
      std::vector<Entity*> open_segment_lst;

      for (stk::mesh::PairIterRelation::iterator j = relations.begin();
          j != relations.end(); ++j) {
        Entity & source = *j->entity();
        if (entity_open[source.key()] == true) {
          open_segment_lst.push_back(&source);
        }
      }

      // Iterate over the open segments
      for (std::vector<Entity*>::iterator j = open_segment_lst.begin();
          j != open_segment_lst.end(); ++j) {
        Entity * segment = *j;
        // Create star of segment
        std::set<EntityKey> subgraph_entity_lst;
        std::set<stkEdge, EdgeLessThan> subgraph_edge_lst;
        topology::star(subgraph_entity_lst, subgraph_edge_lst, *segment);
        // Iterators
        std::set<EntityKey>::iterator firstEntity = subgraph_entity_lst.begin();
        std::set<EntityKey>::iterator lastEntity = subgraph_entity_lst.end();
        std::set<stkEdge>::iterator firstEdge = subgraph_edge_lst.begin();
        std::set<stkEdge>::iterator lastEdge = subgraph_edge_lst.end();

        Subgraph subgraph(bulkData_, firstEntity, lastEntity, firstEdge,
            lastEdge, number_dimensions_);

        // Clone open faces
        stk::mesh::PairIterRelation faces = segment->relations(faceRank);
        std::vector<Entity*> open_face_lst;
        // create a list of open faces
        for (stk::mesh::PairIterRelation::iterator k = faces.begin();
            k != faces.end(); ++k) {
          Entity & source = *k->entity();
          if (entity_open[source.key()] == true) {
            open_face_lst.push_back(&source);
          }
        }

        // Iterate over the open faces
        for (std::vector<Entity*>::iterator k = open_face_lst.begin();
            k != open_face_lst.end(); ++k) {
          Entity * face = *k;
          Vertex faceVertex = subgraph.global_to_local(face->key());
          Vertex newFaceVertex;
          subgraph.clone_boundary_entity(faceVertex, newFaceVertex,
              entity_open);

          EntityKey newFaceKey = subgraph.local_to_global(newFaceVertex);
          Entity * newFace = bulkData_->get_entity(newFaceKey);

          // add original and new faces to the fractured face list
          fractured_face.insert(std::make_pair(face, newFace));

          ++numfractured;
        }

        // Split the articulation point (current segment)
        Vertex segmentVertex = subgraph.global_to_local(segment->key());
        subgraph.split_articulation_point(segmentVertex, entity_open);
      }
      // All open faces and segments have been dealt with. Split the node articulation point
      // Create star of node
      std::set<EntityKey> subgraph_entity_lst;
      std::set<stkEdge, EdgeLessThan> subgraph_edge_lst;
      topology::star(subgraph_entity_lst, subgraph_edge_lst, *entity);
      // Iterators
      std::set<EntityKey>::iterator firstEntity = subgraph_entity_lst.begin();
      std::set<EntityKey>::iterator lastEntity = subgraph_entity_lst.end();
      std::set<stkEdge>::iterator firstEdge = subgraph_edge_lst.begin();
      std::set<stkEdge>::iterator lastEdge = subgraph_edge_lst.end();
      Subgraph subgraph(bulkData_, firstEntity, lastEntity, firstEdge, lastEdge,
          number_dimensions_);

      Vertex node = subgraph.global_to_local(entity->key());
      std::map<Entity*, Entity*> new_connectivity =
          subgraph.split_articulation_point(node, entity_open);

      // Update the connectivity
      for (std::map<Entity*, Entity*>::iterator j = new_connectivity.begin();
          j != new_connectivity.end(); ++j) {
        Entity* element = (*j).first;
        Entity* newNode = (*j).second;

        int id = static_cast<int>(element->identifier());
        std::vector<Entity*> & element_connectivity = connectivity_temp[id - 1];
        for (int k = 0; k < element_connectivity.size(); ++k) {
          // Need to subtract 1 from element number as stk indexes from 1
          //   and connectivity_temp indexes from 0
          if (element_connectivity[k] == entity) {
            element_connectivity[k] = newNode;
            // Duplicate the parameters of old node to new node
            bulkData_->copy_entity_fields(*entity, *newNode);
          }
        }
      }
    }

    // Create the cohesive connectivity
    int j = 1;
    for (std::set<std::pair<Entity*, Entity*> >::iterator i =
        fractured_face.begin(); i != fractured_face.end(); ++i, ++j) {
      Entity * face1 = (*i).first;
      Entity * face2 = (*i).second;
      std::vector<Entity*> cohesive_connectivity;
      cohesive_connectivity = topology::create_cohesive_conn(face1, face2);

      // Output connectivity for testing purposes
      cout << "Cohesive Element " << j << ": ";
      for (int j = 0; j < cohesive_connectivity.size(); ++j) {
        cout << cohesive_connectivity[j]->identifier() << ":";
      }
      cout << "\n";
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
  void topology::set_entities_open(std::map<EntityKey, bool>& entity_open)
  {
    // Fracture occurs at the boundary of the elements in the mesh.
    //   The rank of the boundary elements is one less than the
    //   dimension of the system.
    std::vector<Entity*> boundary_lst;
    stk::mesh::get_entities(*(bulkData_), number_dimensions_ - 1, boundary_lst);

    // Probability that fracture_criterion will return true.
    float p = 1.0;

    // Iterate over the boundary entities
    for (int i = 0; i < boundary_lst.size(); ++i) {
      Entity& entity = *(boundary_lst[i]);
      bool is_open = fracObject->fracture_criterion(entity, p);
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

  void topology::set_entities_open(const std::vector<stk::mesh::Entity*>& fractured_edges,
        std::map<EntityKey, bool>& entity_open)
  {

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

  /*
   * Default constructor
   */
  Subgraph::Subgraph()
  {
    return;
  }

  /**
   * \brief Create a subgraph given two vectors: a vertex list and a edge list.
   *
   * \param[in] bulkData for the stk mesh object
   * \param[in] start of the vertex list
   * \param[in] end of the vertex list
   * \param[in] start of the edge list
   * \param[in] end of the edge list
   * \param[in] number of dimensions in the analysis
   *
   * Subgraph stored as a boost adjacency list.
   * Maps are created to associate the subgraph to the global stk mesh graph.
   * Any changes to the subgraph are automatically mirrored in the stk mesh.
   */
  Subgraph::Subgraph(stk::mesh::BulkData* bulkData,
      std::set<EntityKey>::iterator firstVertex,
      std::set<EntityKey>::iterator lastVertex,
      std::set<topology::stkEdge>::iterator firstEdge,
      std::set<topology::stkEdge>::iterator lastEdge, int numDim)
  {
    // stk mesh data
    bulkData_ = bulkData;
    numDim_ = numDim;

    // Insert vertices and create the vertex map
    std::set<EntityKey>::iterator vertexIterator;
    for (vertexIterator = firstVertex; vertexIterator != lastVertex;
        ++vertexIterator) {
      // get global vertex
      EntityKey globalVertex = *vertexIterator;
      // get entity rank
      EntityRank vertexRank =
          bulkData_->get_entity(globalVertex)->entity_rank();

      // get the new local vertex
      Vertex localVertex = boost::add_vertex(*this);

      localGlobalVertexMap.insert(
          std::map<Vertex, EntityKey>::value_type(localVertex, globalVertex));
      globalLocalVertexMap.insert(
          std::map<EntityKey, Vertex>::value_type(globalVertex, localVertex));

      // store entity rank to vertex property
      VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);
      boost::put(vertexPropertyMap, localVertex, vertexRank);

    }

    // Add edges to the subgraph
    std::set<topology::stkEdge>::iterator edgeIterator;
    for (edgeIterator = firstEdge; edgeIterator != lastEdge; ++edgeIterator) {
      // Get the edge
      topology::stkEdge globalEdge = *edgeIterator;

      // Get global source and target vertices
      EntityKey globalSourceVertex = globalEdge.source;
      EntityKey globalTargetVertex = globalEdge.target;

      // Get local source and target vertices
      Vertex localSourceVertex =
          globalLocalVertexMap.find(globalSourceVertex)->second;
      Vertex localTargetVertex =
          globalLocalVertexMap.find(globalTargetVertex)->second;

      Edge localEdge;
      bool inserted;

      EdgeId edge_id = globalEdge.localId;

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
  EntityKey Subgraph::local_to_global(Vertex localVertex)
  {

    std::map<Vertex, EntityKey>::const_iterator vertexMapIterator =
        localGlobalVertexMap.find(localVertex);

    assert(vertexMapIterator != localGlobalVertexMap.end());

    return (*vertexMapIterator).second;
  }

  /**
   * \brief Map a entity in the stk mesh to a vertex in the subgraph.
   *
   * \param[in] Global entity key for the stk mesh
   * \return Vertex in the subgraph
   *
   * Return local vertex (in the boost graph) given global entity key (in the
   *   stk mesh).
   */
  Vertex Subgraph::global_to_local(EntityKey globalVertexKey)
  {

    std::map<EntityKey, Vertex>::const_iterator vertexMapIterator =
        globalLocalVertexMap.find(globalVertexKey);

    assert(vertexMapIterator != globalLocalVertexMap.end());

    return (*vertexMapIterator).second;
  }

  /**
   * \brief Add a vertex in the subgraph.
   *
   * \param[in] Rank of vertex to be added
   * \return New vertex
   *
   *   Mirrors the change in the subgraph by adding a corresponding entity
   *   to the stk mesh. Adds the relationship between the vertex and entity
   *   to the maps localGlobalVertexMap and globalLocalVertexMap.
   */
  Vertex Subgraph::add_vertex(EntityRank vertex_rank)
  {
    // Insert the vertex into the stk mesh
    // First have to request a new entity of rank N
    std::vector<size_t> requests(numDim_ + 1, 0); // number of entity ranks. 1 + number of dimensions
    requests[vertex_rank] = 1;
    stk::mesh::EntityVector newEntity;
    bulkData_->generate_new_entities(requests, newEntity);
    Entity & globalVertex = *(newEntity[0]);

    // Insert the vertex into the subgraph
    Vertex localVertex = boost::add_vertex(*this);

    // Update maps
    localGlobalVertexMap.insert(
        std::map<Vertex, EntityKey>::value_type(localVertex,
            globalVertex.key()));
    globalLocalVertexMap.insert(
        std::map<EntityKey, Vertex>::value_type(globalVertex.key(),
            localVertex));

    // store entity rank to the vertex property
    VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);
    boost::put(vertexPropertyMap, localVertex, vertex_rank);

    return localVertex;
  }

  /**
   * \brief Remove vertex in subgraph
   *
   * \param[in] Vertex to be removed
   *
   * When the vertex is removed from the subgraph the corresponding entity
   * from the stk mesh is also removed.
   *
   * Both boost and stk require that all edges to and from the vertex/entity
   * are removed before deletion. If any edges remain, will be removed
   * before the vertex/entity deletion.
   */
  void Subgraph::remove_vertex(Vertex & vertex)
  {
    // get the global entity key of vertex
    EntityKey key = local_to_global(vertex);

    // look up entity from key
    Entity* entity = bulkData_->get_entity(key);

    // remove the vertex and key from globalLocalVertexMap and localGlobalVertexMap
    globalLocalVertexMap.erase(key);
    localGlobalVertexMap.erase(vertex);

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

      bulkData_->destroy_relation(*entity, target, edgeId);
    }
    // remove the entity from stk mesh
    bool deleted = bulkData_->destroy_entity(entity);
    assert(deleted);

    return;
  }

  /**
   * \brief Add edge to local graph.
   *
   * \param[in] Local ID of the target vertex with respect to the srouce vertex
   * \param[in] Source vertex in the subgraph
   * \param[in] Target vertex in the subgraph
   * \return New edge and boolean value. If true, edge was inserted, if false
   *  not inserted
   *
   * The edge insertion is mirrored in stk mesh. The edge is only inserted
   * into the stk mesh object if it was inserted into the subgraph.
   */
  std::pair<Edge, bool> Subgraph::add_edge(const EdgeId edge_id,
      const Vertex localSourceVertex, const Vertex localTargetVertex)
  {
    // Add edge to local graph
    std::pair<Edge, bool> localEdge = boost::add_edge(localSourceVertex,
        localTargetVertex, *this);

    if (localEdge.second == false) return localEdge;

    // get global entities
    EntityKey globalSourceKey = local_to_global(localSourceVertex);
    EntityKey globalTargetKey = local_to_global(localTargetVertex);
    Entity* globalSourceVertex = bulkData_->get_entity(globalSourceKey);
    Entity* globalTargetVertex = bulkData_->get_entity(globalTargetKey);

    //testing
    if (globalSourceVertex->entity_rank() - globalTargetVertex->entity_rank()
        != 1) {
      cout << "add edge:" << globalSourceVertex->entity_rank() << ","
          << globalSourceVertex->identifier() << " "
          << globalTargetVertex->entity_rank() << ","
          << globalTargetVertex->identifier() << "\n";
    }

    // Add edge to stk mesh
    bulkData_->declare_relation(*(globalSourceVertex), *(globalTargetVertex),
        edge_id);

    // Add edge id to edge property
    EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(), *this);
    boost::put(edgePropertyMap, localEdge.first, edge_id);

    return localEdge;
  }

  /**
   * \brief Remove edge from graph
   *
   * \param[in] Source vertex in subgraph
   * \param[in] Target vertex in subgraph
   *
   * Edge removal is mirrored in the stk mesh.
   *
   */
  void Subgraph::remove_edge(const Vertex & localSourceVertex,
      const Vertex & localTargetVertex)
  {
    // Get the local id of the edge in the subgraph

    Edge edge;
    bool inserted;
    boost::tie(edge, inserted) = boost::edge(localSourceVertex,
        localTargetVertex, *this);

    assert(inserted);

    EdgeId edge_id = get_edge_id(edge);

    // remove local edge
    boost::remove_edge(localSourceVertex, localTargetVertex, *this);

    // remove relation from stk mesh
    EntityKey globalSourceId = Subgraph::local_to_global(localSourceVertex);
    EntityKey globalTargetId = Subgraph::local_to_global(localTargetVertex);
    Entity* globalSourceVertex = bulkData_->get_entity(globalSourceId);
    Entity* globalTargetVertex = bulkData_->get_entity(globalTargetId);

    bulkData_->destroy_relation(*(globalSourceVertex), *(globalTargetVertex),
        edge_id);

    return;
  }

  /**
   * \param[in] Vertex in subgraph
   * \return Rank of vertex
   */
  EntityRank &
  Subgraph::get_vertex_rank(const Vertex vertex)
  {
    VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);

    return boost::get(vertexPropertyMap, vertex);
  }

  /**
   * \param[in] Edge in subgraph
   * \return Local numbering of edge target with respect to edge source
   *
   * In stk mesh, all relationships between entities have a local Id
   * representing the correct ordering. Need this information to create
   * or delete relations in the stk mesh.
   */
  EdgeId &
  Subgraph::get_edge_id(const Edge edge)
  {
    EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(), *this);

    return boost::get(edgePropertyMap, edge);
  }

  /**
   * \brief Function determines whether the input vertex is an articulation
   * point of the subgraph.
   *
   * \param[in] Input vertex
   * \param[out] Number of components
   * \param[out] map of vertex and associated component number
   *
   * Function checks vertex by the boost connected components algorithm to a
   * copy of the subgraph. The copy does not include the input vertex.
   * Copy is an undirected graph as required by the connected components
   * algorithm.
   *
   * Returns the number of connected components as well as a map of the
   * vertex in the subgraph and the component number.
   */
  void Subgraph::undirected_graph(Vertex input_vertex, int & numComponents,
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
        out_edge_iterator out_edge_begin;
        out_edge_iterator out_edge_end;
        boost::tie(out_edge_begin, out_edge_end) = out_edges(*i, *this);

        for (out_edge_iterator j = out_edge_begin; j != out_edge_end; ++j) {
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
    numComponents = boost::connected_components(g, &component[0]);

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
  void Subgraph::clone_boundary_entity(Vertex & vertex, Vertex & newVertex,
      std::map<EntityKey, bool> & entity_open)
  {
    // Check that number of in_edges = 2
    boost::graph_traits<boostGraph>::degree_size_type num_in_edges =
        boost::in_degree(vertex, *this);
    if (num_in_edges != 2) return;

    // Check that vertex = open
    EntityKey vertKey = Subgraph::local_to_global(vertex);
    assert(entity_open[vertKey]==true);

    // Get the vertex rank
    EntityRank vertexRank = Subgraph::get_vertex_rank(vertex);

    // Create a new vertex of same rank as vertex
    newVertex = Subgraph::add_vertex(vertexRank);

    // Copy the out_edges of vertex to newVertex
    out_edge_iterator out_edge_begin;
    out_edge_iterator out_edge_end;
    boost::tie(out_edge_begin, out_edge_end) = boost::out_edges(vertex, *this);
    for (out_edge_iterator i = out_edge_begin; i != out_edge_end; ++i) {
      Edge edge = *i;
      EdgeId edgeId = Subgraph::get_edge_id(edge);
      Vertex target = boost::target(edge, *this);
      Subgraph::add_edge(edgeId, newVertex, target);
    }

    // Copy all out edges not in the subgraph to the new vertex
    Subgraph::clone_out_edges(vertex, newVertex);

    // Remove one of the edges from vertex, copy to newVertex
    // Arbitrarily remove the first edge from original vertex
    in_edge_iterator in_edge_begin;
    in_edge_iterator in_edge_end;
    boost::tie(in_edge_begin, in_edge_end) = boost::in_edges(vertex, *this);
    Edge edge = *(in_edge_begin);
    EdgeId edgeId = Subgraph::get_edge_id(edge);
    Vertex source = boost::source(edge, *this);
    Subgraph::remove_edge(source, vertex);

    // Add edge to new vertex
    Subgraph::add_edge(edgeId, source, newVertex);

    // Have to clone the out edges of the original entity to the new entity.
    // These edges are not in the subgraph

    // Clone process complete, set entity_open to false
    entity_open[vertKey] = false;

    return;
  }

  /**
   * \brief Splits an articulation point.
   *
   * \param[in] Input vertex
   * \param Map of entity and boolean value is open
   * \return Map of element and new node
   *
   * An articulation point is defined as a vertex which if removed
   * yields a graph with more than 1 connected components. Creates
   * an undirected graph and checks connected components of graph without
   * vertex. Check if vertex is articulation point.
   *
   * Clones articulation point and splits in-edges between original and new
   * vertices. The out-edges of the vertex are not in the subgraph. For
   * a consistent global graph, add the out-edges of the vertex to the new
   * vertex/vertices.
   *
   * If the vertex is a node, create a map between the element and the new
   * node. If the nodal connectivity of an element does not change, do not
   * add to the map.
   */
  std::map<Entity*, Entity*> Subgraph::split_articulation_point(Vertex vertex,
      std::map<EntityKey, bool> & entity_open)
  {
    // Check that vertex = open
    EntityKey vertKey = Subgraph::local_to_global(vertex);
    assert(entity_open[vertKey]==true);

    // get rank of vertex
    EntityRank vertexRank = Subgraph::get_vertex_rank(vertex);

    // Create undirected graph
    int numComponents;
    std::map<Vertex, int> components;
    Subgraph::undirected_graph(vertex, numComponents, components);

    // The function returns an updated connectivity map. If the vertex rank is
    //   not node, then this map will be of size 0.
    std::map<Entity*, Entity*> new_connectivity;

    // Check number of connected components in undirected graph. If = 1, return
    if (numComponents == 1) return new_connectivity;

    // If number of connected components > 1, split vertex in subgraph and stk mesh
    // number of new vertices = numComponents - 1
    std::vector<Vertex> newVertex;
    for (int i = 0; i < numComponents - 1; ++i) {
      Vertex newVert = Subgraph::add_vertex(vertexRank);
      newVertex.push_back(newVert);
    }

    // create a map of elements to new node numbers
    // only do this if the input vertex is a node (don't require otherwise)
    if (vertexRank == 0) {
      for (std::map<Vertex, int>::iterator i = components.begin();
          i != components.end(); ++i) {
        int componentNum = (*i).second;
        Vertex currentVertex = (*i).first;
        EntityRank currentRank = Subgraph::get_vertex_rank(currentVertex);
        // Only add to map if the vertex is an element
        if (currentRank == numDim_ && componentNum != 0) {
          Entity* element = bulkData_->get_entity(
              Subgraph::local_to_global(currentVertex));
          Entity* newNode = bulkData_->get_entity(
              Subgraph::local_to_global(newVertex[componentNum - 1]));
          new_connectivity.insert(
              std::map<Entity*, Entity*>::value_type(element, newNode));
        }
      }
    }

    // Copy the out edges of the original vertex to the new vertex
    for (int i = 0; i < newVertex.size(); ++i) {
      Subgraph::clone_out_edges(vertex, newVertex[i]);
    }

    // vector for edges to be removed. Vertex is source and edgeId the local id of the edge
    std::vector<std::pair<Vertex, EdgeId> > removed;

    // Iterate over the in edges of the vertex to determine which will be removed
    in_edge_iterator in_edge_begin;
    in_edge_iterator in_edge_end;
    boost::tie(in_edge_begin, in_edge_end) = boost::in_edges(vertex, *this);
    for (in_edge_iterator i = in_edge_begin; i != in_edge_end; ++i) {
      Edge edge = *i;
      Vertex source = boost::source(edge, *this);

      std::map<Vertex, int>::const_iterator componentIterator = components.find(
          source);
      int vertComponent = (*componentIterator).second;
      Entity& entity = *(bulkData_->get_entity(
          Subgraph::local_to_global(source)));
      // Only replace edge if vertex not in component 0
      if (vertComponent != 0) {
        EdgeId edgeId = Subgraph::get_edge_id(edge);
        removed.push_back(std::make_pair(source, edgeId));
      }
    }

    // remove all edges in vector removed and replace with new edges
    for (std::vector<std::pair<Vertex, EdgeId> >::iterator i = removed.begin();
        i != removed.end(); ++i) {
      std::pair<Vertex, EdgeId> edge = *i;
      Vertex source = edge.first;
      EdgeId edgeId = edge.second;
      std::map<Vertex, int>::const_iterator componentIterator = components.find(
          source);
      int vertComponent = (*componentIterator).second;

      Subgraph::remove_edge(source, vertex);
      std::pair<Edge, bool> inserted = Subgraph::add_edge(edgeId, source,
          newVertex[vertComponent - 1]);
      assert(inserted.second==true);
    }

    // split process complete, set entity_open to false
    entity_open[vertKey] = false;

    return new_connectivity;
  }

  /**
   * \brief Clone all out edges of a vertex to a new vertex.
   *
   * \param[in] Original vertex
   * \param[in] New vertex
   *
   * The global graph must remain consistent when new vertices are added. In
   * split_articulation_point and clone_boundary_entity, all out-edges of
   * the original vertex may not be in the subgraph.
   *
   * If there are missing edges in the subgraph, clone them from the original
   * vertex to the new vertex. Edges not originally in the subgraph are added
   * to the global graph only.
   */
  void Subgraph::clone_out_edges(Vertex & originalVertex, Vertex & newVertex)
  {
    // Get the entity for the original and new vertices
    EntityKey originalKey = Subgraph::local_to_global(originalVertex);
    EntityKey newKey = Subgraph::local_to_global(newVertex);
    Entity & originalEntity = *(bulkData_->get_entity(originalKey));
    Entity & newEntity = *(bulkData_->get_entity(newKey));

    // Iterate over the out edges of the original vertex and check against the
    //   out edges of the new vertex. If the edge does not exist, add.
    stk::mesh::PairIterRelation originalRelations = originalEntity.relations(
        originalEntity.entity_rank() - 1);
    for (int i = 0; i < originalRelations.size(); ++i) {
      stk::mesh::PairIterRelation newRelations = newEntity.relations(
          newEntity.entity_rank() - 1);
      // assume the edge doesn't exist
      bool exists = false;
      for (int j = 0; j < newRelations.size(); ++j) {
        if (originalRelations[i].entity() == newRelations[j].entity()) {
          exists = true;
        }
      }
      if (exists == false) {
        EdgeId edgeId = originalRelations[i].identifier();
        Entity& target = *(originalRelations[i].entity());
        bulkData_->declare_relation(newEntity, target, edgeId);
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
   * Similar to output_to_graphviz function in topology class.
   * If fracture criterion for entity is satisfied, the entity and all
   * associated lower order entities are marked open. All open entities are
   * displayed as such in output file.
   *
   * To create final output figure, run command below from terminal:
   *   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
   */
  void Subgraph::output_to_graphviz(std::string & gviz_output,
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
        EntityKey key = local_to_global(*i);
        Entity & entity = *(bulkData_->get_entity(key));
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
        out_edge_iterator out_edge_begin;
        out_edge_iterator out_edge_end;
        boost::tie(out_edge_begin, out_edge_end) = out_edges(*i, *this);

        for (out_edge_iterator j = out_edge_begin; j != out_edge_end; ++j) {
          Edge out_edge = *j;
          Vertex source = boost::source(out_edge, *this);
          Vertex target = boost::target(out_edge, *this);

          EntityKey sourceKey = local_to_global(source);
          Entity & global_source = *(bulkData_->get_entity(sourceKey));

          EntityKey targetKey = local_to_global(target);
          Entity & global_target = *(bulkData_->get_entity(targetKey));

          EdgeId edgeId = get_edge_id(out_edge);

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

