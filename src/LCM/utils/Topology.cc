/*
 * Topology.cc
 *  Set of topology manipulation functions for 2D and 3D stk meshes.
 *  Created on: Jul 11, 2011
 *      Author: jrthune
 */

#include "Topology.h"

namespace LCM{

/*
 * Default constructor for topology
 */
topology::topology():
		numDim(0),
		discretization_ptr_(Teuchos::null)
{
	return;
}

/*
 * Create mesh data structure
 *   Inputs:
 *     input_file is exodus II input file name
 *     output_file is exodus II output file name
 */
topology::topology(
		std::string const & input_file,
		std::string const & output_file)
{
	Teuchos::RCP<Teuchos::ParameterList>
	  disc_params = rcp(new Teuchos::ParameterList("params"));

	//set Method to Exodus and set input file name
	disc_params->set<std::string>("Method", "Exodus");
	disc_params->set<std::string>("Exodus Input File Name", input_file);
	disc_params->set<std::string>("Exodus Output File Name", output_file);
	//disc_params->print(std::cout);

	Teuchos::RCP<Epetra_Comm>
	  communicator = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

	Albany::DiscretizationFactory
	  disc_factory(disc_params, communicator);

	const Teuchos::RCP<Albany::MeshSpecsStruct>
	meshSpecs = disc_factory.createMeshSpecs();

	Teuchos::RCP<Albany::StateInfoStruct>
	stateInfo = Teuchos::rcp(new Albany::StateInfoStruct());

	discretization_ptr_ = disc_factory.createDiscretization(3, stateInfo);

	// Dimensioned: Workset, Cell, Local Node
	Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >
	  element_connectivity = discretization_ptr_->getWsElNodeID();

	Teuchos::ArrayRCP<double>
	  coordinates = discretization_ptr_->getCoordinates();

	// Need to access the bulkData and metaData classes in the mesh datastructure
	Albany::STKDiscretization &
	  stk_discretization = static_cast<Albany::STKDiscretization &>(*discretization_ptr_);

	stkMeshStruct_ = stk_discretization.getSTKMeshStruct();

	bulkData_ = stkMeshStruct_->bulkData;
	stk::mesh::fem::FEMMetaData & metaData = *stkMeshStruct_->metaData;

	// The entity ranks
	nodeRank = metaData.NODE_RANK;
	edgeRank = metaData.EDGE_RANK;
	faceRank = metaData.FACE_RANK;
	elementRank = metaData.element_rank();
	numDim = stkMeshStruct_->numDim;

	return;
}

/*
 * Output relations associated with entity
 */
void
topology::disp_relation(
		stk::mesh::Entity & entity)
{
	cout << "Relations for entity (identifier,rank): " << entity.identifier()
			<< "," << entity.entity_rank() << "\n";
	stk::mesh::PairIterRelation relations = entity.relations();
		for (int i = 0; i < relations.size(); ++i){
			cout << "entity:\t" << relations[i].entity()->identifier() << ","
				 << relations[i].entity()->entity_rank() << "\tlocal id: "
				 << relations[i].identifier() << "\n";
		}
		return;
}

/*
 * Output relations of rank entityRank associated with entity
 */
void
topology::disp_relation(
		stk::mesh::Entity & entity,
		stk::mesh::EntityRank entityRank)
{
	cout << "Relations of rank " << entityRank << " for entity (identifier,rank): "
			<< entity.identifier() << "," << entity.entity_rank() << "\n";
	stk::mesh::PairIterRelation relations = entity.relations(entityRank);
		for (int i = 0; i < relations.size(); ++i){
			cout << "entity:\t" << relations[i].entity()->identifier() << ","
				 << relations[i].entity()->entity_rank() << "\tlocal id: "
				 << relations[i].identifier() << "\n";
		}
		return;
}

/*
 * Output the mesh connectivity
 *   stk mesh must contain relations between the elements and the nodes (as in
 *   the original stk mesh object)
 */
void
topology::disp_connectivity()
{
	// Create a list of element entities
	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData_),elementRank,element_lst);

	// Loop over the elements
	for (int i = 0; i < element_lst.size(); ++i){
		stk::mesh::PairIterRelation relations = element_lst[i]->relations(nodeRank);
		cout << "Nodes of Element " << element_lst[i]->identifier() << "\n";

		for (int j = 0; j < relations.size(); ++j){
			stk::mesh::Entity& node = *(relations[j].entity());
			cout << ":"  << node.identifier();
		}
		cout << ":\n";
	}

	//topology::disp_relation(*(element_lst[0]));

	//std::vector<stk::mesh::Entity*> face_lst;
	//stk::mesh::get_entities(*(bulkData_),elementRank-1,face_lst);
	//topology::disp_relation(*(face_lst[1]));

	return;
}

/*
 * Generic fracture criterion function. Given an entity and
 *   probability, will determine if criterion is met.
 *   Will return true if fracture criterion is met, else false.
 *   Fracture only defined on surface of elements. Thus, input entity
 *   must be of rank dimension-1, else error.
 */
bool
topology::fracture_criterion(
		stk::mesh::Entity& entity,
		float p)
{
	// Fracture only defined on the boundary of the elements
	stk::mesh::EntityRank rank = entity.entity_rank();
	assert(rank==numDim-1);

	bool is_open = false;
	// Check criterion
	float random = 0.5 + 0.5*Teuchos::ScalarTraits<double>::random();
	if (random < p)
		is_open = true;

	return is_open;
}

/*
 * Iterates over the boundary entities of the mesh of (all entities of rank
 *   dimension-1) and checks fracture criterion. If fracture_criterion is
 *   met, the entity and all lower order entities associated with it are
 *   marked as open.
 */
void
topology::set_entities_open(
		std::map<stk::mesh::EntityKey,bool>& entity_open)
{
	// Fracture occurs at the boundary of the elements in the mesh.
	//   The rank of the boundary elements is one less than the
	//   dimension of the system.
	std::vector<stk::mesh::Entity*> boundary_lst;
	stk::mesh::get_entities(*(bulkData_),numDim-1,boundary_lst);

	// Probability that fracture_criterion will return true.
	float p = 0.5;

	// Iterate over the boundary entities
	for (int i = 0; i < boundary_lst.size(); ++i){
		stk::mesh::Entity& entity = *(boundary_lst[i]);
		bool is_open = topology::fracture_criterion(entity,p);
		// If the criterion is met, need to set lower rank entities
		//   open as well
		if (is_open == true && numDim == 3){
			entity_open[entity.key()] = true;
			stk::mesh::PairIterRelation segments =
					entity.relations(entity.entity_rank()-1);
			// iterate over the segments
			for (int j = 0; j < segments.size(); ++j){
				stk::mesh::Entity & segment = *(segments[j].entity());
				entity_open[segment.key()] = true;
				stk::mesh::PairIterRelation nodes =
						segment.relations(segment.entity_rank()-1);
				// iterate over nodes
				for (int k = 0; k < nodes.size(); ++k){
					stk::mesh::Entity& node = *(nodes[k].entity());
					entity_open[node.key()] = true;
				}
			}
		}
		// If the mesh is 2D
		else if (is_open == true && numDim == 2){
			entity_open[entity.key()] = true;
			stk::mesh::PairIterRelation nodes =
					entity.relations(entity.entity_rank()-1);
			// iterate over nodes
			for (int j = 0; j < nodes.size(); ++j){
				stk::mesh::Entity & node = *(nodes[j].entity());
				entity_open[node.key()] = true;
			}
		}
	}

	return;
}

/*
 * Output the graph associated with the mesh to graphviz .dot
 *   file for visualization purposes. If fracture criterion for
 *   entity is satisfied, the entity and all associated lower order
 *   entities are marked open. All open entities are displayed as
 *   such in output file.
 * Can create figure using:
 *   dot -Tpng output.dot -o output.png
 */
void
topology::output_to_graphviz(
		std::map<stk::mesh::EntityKey, bool> & entity_open)
{
	// Open output file
	std::ofstream gviz_out;
	gviz_out.open ("output.dot", std::ios::out);

	cout << "Write graph to graphviz dot file\n";

	if (gviz_out.is_open()){
		// Write beginning of file
		gviz_out << "digraph mesh {\n"
				 << "node [colorscheme=paired12]"
				 << "edge [colorscheme=paired12]";

		std::vector<stk::mesh::Entity*> entity_lst;
		stk::mesh::get_entities(*(bulkData_),elementRank,entity_lst);

		std::vector<std::vector<stk::mesh::Entity*> > relation_lst;
		std::vector<int> relation_local_id;

		// Elements
		for (int i = 0; i < entity_lst.size(); ++i){
			stk::mesh::Entity & entity = *(entity_lst[i]);
			stk::mesh::PairIterRelation relations = entity.relations();

			gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
					 << "\" [label=\"Element " << entity.identifier()
					 << "\",style=filled,fillcolor=\"8\"]\n";
			for (int j = 0; j < relations.size(); ++j){
				if (relations[j].entity_rank() < entity.entity_rank()){
					std::vector<stk::mesh::Entity*> temp;
					temp.push_back(&entity);
					temp.push_back(relations[j].entity());
					relation_lst.push_back(temp);
					relation_local_id.push_back(relations[j].identifier());
				}
			}
		}

		stk::mesh::get_entities(*(bulkData_),faceRank,entity_lst);

		// Faces
		for (int i = 0; i < entity_lst.size(); ++i){
			stk::mesh::Entity & entity = *(entity_lst[i]);
			stk::mesh::PairIterRelation relations = entity.relations();

			if (entity_open[entity.key()] == true)
				gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
					<< "\" [label=\"Face " << entity.identifier()
					<< "\",style=filled,fillcolor=\"1\"]\n";
			else
				gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
					<< "\" [label=\"Face " << entity.identifier()
					<< "\",style=filled,fillcolor=\"2\"]\n";
			for (int j = 0; j < relations.size(); ++j){
				if (relations[j].entity_rank() < entity.entity_rank()){
					std::vector<stk::mesh::Entity*> temp;
					temp.push_back(&entity);
					temp.push_back(relations[j].entity());
					relation_lst.push_back(temp);
					relation_local_id.push_back(relations[j].identifier());
				}
			}
		}

		stk::mesh::get_entities(*(bulkData_),edgeRank,entity_lst);

		// Edges
		for (int i = 0; i < entity_lst.size(); ++i){
			stk::mesh::Entity & entity = *(entity_lst[i]);
			stk::mesh::PairIterRelation relations = entity.relations();

			if (entity_open[entity.key()] == true)
				gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
					<< "\" [label=\"Segment " << entity.identifier()
					<< "\",style=filled,fillcolor=\"3\"]\n";
			else
				gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
					<< "\" [label=\"Segment " << entity.identifier()
					<< "\",style=filled,fillcolor=\"4\"]\n";
			for (int j = 0; j < relations.size(); ++j){
				if (relations[j].entity_rank() < entity.entity_rank()){
					std::vector<stk::mesh::Entity*> temp;
					temp.push_back(&entity);
					temp.push_back(relations[j].entity());
					relation_lst.push_back(temp);
					relation_local_id.push_back(relations[j].identifier());
				}
			}
		}

		stk::mesh::get_entities(*(bulkData_),nodeRank,entity_lst);

		// Nodes
		for (int i = 0; i < entity_lst.size(); ++i){
			stk::mesh::Entity & entity = *(entity_lst[i]);

			if (entity_open[entity.key()] == true)
				gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
					<< "\" [label=\"Node " << entity.identifier()
					<< "\",style=filled,fillcolor=\"5\"]\n";
			else
				gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
					<< "\" [label=\"Node " << entity.identifier()
					<< "\",style=filled,fillcolor=\"6\"]\n";
		}

		for (int i = 0; i < relation_lst.size(); ++i){
			std::vector<stk::mesh::Entity*> temp = relation_lst[i];
			stk::mesh::Entity& origin = *(temp[0]);
			stk::mesh::Entity& destination = *(temp[1]);
			std::string color;
			switch(relation_local_id[i]){
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
					 << "\" -> \"" << destination.identifier() << "_" << destination.entity_rank()
					 << "\" [color=\"" << color << "\"]" << "\n";
		}

		// File end
		gviz_out << "}";
		gviz_out.close();
	}
	else
		cout << "Unable to open graphviz output file 'output.dot'\n";

	return;
}

/*
 * Creates the full graph representation of the mesh. Default graph has only
 *   elements and nodes. The original node connectivity will be deleted in
 *   later steps, store the connectivity in temporary array.
 *
 * Note: Function must be called before mesh modification begins
 */
void
topology::graph_initialization(
		std::vector<std::vector<stk::mesh::Entity*> >  & connectivity_temp)
{
	stk::mesh::PartVector add_parts;
	stk::mesh::create_adjacent_entities(*(bulkData_), add_parts);

	// Create the temporary connectivity array
	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData_),elementRank,element_lst);

	for (int i = 0; i < element_lst.size(); ++i){
		stk::mesh::PairIterRelation nodes = element_lst[i]->relations(nodeRank);
		std::vector<stk::mesh::Entity*> temp;
		for (int j = 0; j < nodes.size(); ++j){
			stk::mesh::Entity* node = nodes[j].entity();
			temp.push_back(node);
		}
		connectivity_temp.push_back(temp);
	}

	// Allow for mesh adaptation without entire graph algorithm implemented.
	//   Once done, change to remove_relations = 1
	int remove_relations = 0;
	if (remove_relations == 1){
		bulkData_->modification_begin();
		topology::remove_extra_relations();
		bulkData_->modification_end();
		//topology::output_to_graphviz(*(bulkData));
	}

	return;
}

/*
 * stk::mesh::create_adjacent_entities creates all entities in
 *   graph instead of default elements and nodes. All entities are
 *   connected through relationships. Graph algorithms require
 *   relationships only between entities separated by one degree, e.g. elements
 *   and faces in a 3D graph.
 *   Function removes all other relationships, e.g. between elements
 *   and nodes.
 *
 * Note: Valid for 2D and 3D meshes.
 */
void
topology::remove_extra_relations()
{
	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData_),elementRank,element_lst);

	// Remove extra relations from element
	for (int i = 0; i < element_lst.size(); ++i){
		stk::mesh::Entity & element = *(element_lst[i]);
		stk::mesh::PairIterRelation relations = element.relations();
		std::vector<stk::mesh::Entity*> del_relations;
		std::vector<int> del_ids;
		for (stk::mesh::PairIterRelation::iterator j = relations.begin();
				j != relations.end(); ++j){
			if (j->entity_rank() != elementRank-1){
				del_relations.push_back(j->entity());
				del_ids.push_back(j->identifier());
			}
		}
		for (int j = 0; j < del_relations.size(); ++j){
			stk::mesh::Entity & entity = *(del_relations[j]);
			bulkData_->destroy_relation(element,entity,del_ids[j]);
		}
	};

	if (elementRank == 3){
		// Remove extra relations from face
		std::vector<stk::mesh::Entity*> face_lst;
		stk::mesh::get_entities(*(bulkData_),elementRank-1,face_lst);
		stk::mesh::EntityRank entityRank = face_lst[0]->entity_rank();
		for (int i = 0; i < face_lst.size(); ++i){
			stk::mesh:: Entity & face = *(face_lst[i]);
			stk::mesh::PairIterRelation relations = face_lst[i]->relations();
			std::vector<stk::mesh::Entity*> del_relations;
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
				stk::mesh::Entity & entity = *(del_relations[j]);
				bulkData_->destroy_relation(face,entity,del_ids[j]);
			}
		}
	}

	return;
}

/*
 * After mesh manipulations are complete, need to recreate the original
 *   mesh representation as expected by Albany_STKDiscretization. Remove
 *   all extra entities (faces and edges for a 3D mesh) and recreate relationships
 *   between elements and nodes. Nodal connectivity data for each element is stored
 *   in connectivity_temp.
 *
 * Note: must be called before mesh modification has ended
 */
void
topology::graph_cleanup(
		std::vector<std::vector<stk::mesh::Entity*> >  & connectivity_temp)
{
	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(*(bulkData_),elementRank,element_lst);

	// Remove faces from graph
	std::vector<stk::mesh::Entity*> face_lst;
	stk::mesh::get_entities(*(bulkData_),faceRank,face_lst);
	for (int i = 0; i < face_lst.size(); ++i){
		//bulkData::destroy_entity() requires entity has no relations
		stk::mesh::PairIterRelation relations = face_lst[i]->relations();
		for (int j = 0; j < relations.size(); ++j){
			// relation must be from higher to lower rank entity
			if (face_lst[i]->entity_rank() > relations[j].entity_rank())
				bulkData_->destroy_relation(*(face_lst[i]),
						*(relations[j].entity()),relations[j].identifier());
			else
				bulkData_->destroy_relation(*(relations[j].entity()),
						*(face_lst[i]),relations[j].identifier());
		}
		bulkData_->destroy_entity(face_lst[i]);
	}

	// Remove edges from graph
	std::vector<stk::mesh::Entity*> edge_lst;
	stk::mesh::get_entities(*(bulkData_),edgeRank,edge_lst);
	for (int i = 0; i < edge_lst.size(); ++i){
		//bulkData::destroy_entity() requires entity has no relations
		stk::mesh::PairIterRelation relations = edge_lst[i]->relations();
		for (int j = 0; j < relations.size(); ++j){
			// relation must be from higher to lower rank entity
			if (edge_lst[i]->entity_rank() > relations[j].entity_rank())
				bulkData_->destroy_relation(*(edge_lst[i]),
						*(relations[j].entity()),relations[j].identifier());
			else
				bulkData_->destroy_relation(*(relations[j].entity()),
						*(edge_lst[i]),relations[j].identifier());
		}
		bulkData_->destroy_entity(edge_lst[i]);
	}

	// Add relations from element to nodes
	for (int i = 0; i < element_lst.size(); ++i){
		stk::mesh::Entity & element = *(element_lst[i]);
		for (int j = 0; j < connectivity_temp.size(); ++j){
			stk::mesh::Entity & node = *(connectivity_temp[i][j]);
			bulkData_->declare_relation(element,node,j);
		}
	}

	return;
}

/*
 * Create vectors describing the vertices and edges of the star of an entity
 *   in the stk mesh. Currently, only allowed input is a node.
 *   TODO: allow arbitrary rank input.
 *
 *   The star of a graph vertex is defined as the vertex and all higher order
 *   vertices which are connected to it when traversing up the graph from the
 *   input vertex.
 */
void
topology::star(std::set<entityKey> & subgraph_entity_lst,
		std::set<stkEdge,EdgeLessThan> & subgraph_edge_lst,
		stk::mesh::Entity & entity)
{

	// Create the vector and edge lists for the subgraph
	stk::mesh::PairIterRelation segments = entity.relations(edgeRank);
	subgraph_entity_lst.insert(entity.key());

	for (int i = 0; i < segments.size(); ++i){
		stk::mesh::Entity & segment = *(segments[i].entity());
		stk::mesh::PairIterRelation faces = segment.relations(faceRank);
		subgraph_entity_lst.insert(segment.key());
		stkEdge edge;
		edge.source = segment.key();
		edge.target = entity.key();
		edge.localId = segments[i].identifier();
		subgraph_edge_lst.insert(edge);


		for (int j = 0; j < faces.size(); ++j){
			stk::mesh::Entity & face = *(faces[j].entity());
			stk::mesh::PairIterRelation elements = face.relations(elementRank);
			subgraph_entity_lst.insert(face.key());
			stkEdge edge;
			edge.source = face.key();
			edge.target = segment.key();
			edge.localId = faces[j].identifier();
			subgraph_edge_lst.insert(edge);

			for (int k = 0; k < elements.size(); ++k){
				stk::mesh::Entity & element = *(elements[k].entity());
				subgraph_entity_lst.insert(element.key());
				stkEdge edge;
				edge.source = element.key();
				edge.target = face.key();
				edge.localId = elements[k].identifier();
				subgraph_edge_lst.insert(edge);
			}
		}
	}
	return;
}

/*
 * Used to split elements of a mesh along faces.
 *   Duplicates old_entity and removes connection between in_entity and old_entity.
 *   A new entity with the same out_edges (i.e. connectivity to lower order
 *   entities) is created.
 *   old_entity is the original entity before the duplication
 *   in_entity is a higher order entity with an out_edge connected to old_entity
 *
 * Note: Must be called after mesh modification has begun.
 *
 *       Does *Not* utilize the method from Mota, et al. 2008. Proof of concept for
 *       mesh modification only.
 */
void
topology::duplicate_entity(
		stk::mesh::Entity & in_entity,
		stk::mesh::Entity & old_entity,
		std::vector<std::vector<stk::mesh::Entity*> > & connectivity_temp,
		int local_id,
		stk::mesh::Entity & element)
{
	// Rank of the current entity
	const stk::mesh::EntityRank entity_rank = old_entity.entity_rank();

	stk::mesh::PairIterRelation elem_conn = old_entity.relations(elementRank);
	bool in_element=false;
	int id = 0;
	for (int i =0; i < elem_conn.size(); ++i){
		if (elem_conn[i].entity()->identifier() == element.identifier()){
			in_element = true;
			id = elem_conn[i].identifier();
		}
	}

	// Only duplicate an entity if:
	//   1) the number of elements connected to it are greater than one
	//   2 ) the entity is in the current element. (May have been removed in previous step)
	if ((old_entity.relations(elementRank)).size() > 1 && in_element == true){
		// Get the out_edges of old_entity if the entity is not a node (nodes have no out edges)
		stk::mesh::PairIterRelation out_edges;
		if (entity_rank > 0){
			out_edges = old_entity.relations(entity_rank - 1);
		}

		std::vector<size_t> requests(elementRank+1, 0);
		requests[entity_rank] = 1;
		stk::mesh::EntityVector newEntities;

		// Generate the new entity
		bulkData_->generate_new_entities(requests,newEntities);

		stk::mesh::Entity & new_entity = *(newEntities[0]);

		// Remove relation between in_entity and old_entity (old_entity is no
		//   longer associated with in_entity)
		bulkData_->destroy_relation(in_entity,old_entity,local_id);

		// Declare relation between in_entity and new_entity
		bulkData_->declare_relation(in_entity,new_entity,local_id);

		// The connection to the element
		bulkData_->destroy_relation(element,old_entity,id);
		bulkData_->declare_relation(element,new_entity,id);

		// Duplicate the parameters of old_entity to new_entity
		bulkData_->copy_entity_fields(old_entity,new_entity);


		if (entity_rank > 0){
			// Copy the out_edges of old_entity to new_entity
			for (int i = 0; i < out_edges.size(); ++i){
				bulkData_->declare_relation(new_entity, *(out_edges[i].entity()),i);
			}

			for (int i = 0; i < out_edges.size(); ++i){
				duplicate_entity(new_entity,
						*(out_edges[i].entity()),
						connectivity_temp,
						i,element);

			}
		}
		// Entity is a node
		// need to update the element connectivity stored in connectivity_temp
		else{
			for (int i = 0; i < connectivity_temp.size(); ++i){
				// connectivity_temp idexed from 0, identifier() from 1
				if (connectivity_temp[element.identifier()-1][i]->identifier() == old_entity.identifier())
					connectivity_temp[element.identifier()-1][i] = &new_entity;
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

/*
 * Create a subgraph given two vectors: a vertex list and a edge list.
 *   Subgraph stored as a boost adjacency list.
 *   Maps the subgraph to the global stk mesh graph.
 */
Subgraph::Subgraph(
		stk::mesh::BulkData* bulkData,
		std::set<entityKey>::iterator firstVertex,
		std::set<entityKey>::iterator lastVertex,
		std::set<topology::stkEdge>::iterator firstEdge,
		std::set<topology::stkEdge>::iterator lastEdge,
		int numDim)
{
	// stk mesh data
	bulkData_ = bulkData;
	numDim_ = numDim;

	// Insert vertices and create the vertex map
	std::set<entityKey>::iterator vertexIterator;
	for (vertexIterator = firstVertex;
			vertexIterator != lastVertex;
			++vertexIterator){
		// get global vertex
		stk::mesh::EntityKey globalVertex = *vertexIterator;
		// get entity rank
		EntityRank vertexRank = bulkData_->get_entity(globalVertex)->entity_rank();

		// get the new local vertex
		Vertex localVertex = boost::add_vertex(*this);

		localGlobalVertexMap.insert(std::map<Vertex,stk::mesh::EntityKey>::value_type(localVertex,globalVertex));
		globalLocalVertexMap.insert(std::map<stk::mesh::EntityKey,Vertex>::value_type(globalVertex,localVertex));

		// store entity rank to vertex property
		VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(),*this);
		boost::put(vertexPropertyMap,localVertex,vertexRank);

	}

	// Add edges to the subgraph
	std::set<topology::stkEdge>::iterator edgeIterator;
	for (edgeIterator = firstEdge;
			edgeIterator != lastEdge;
			++edgeIterator){
		// Get the edge
		topology::stkEdge globalEdge = *edgeIterator;

		// Get global source and target vertices
		entityKey globalSourceVertex = globalEdge.source;
		entityKey globalTargetVertex = globalEdge.target;

		// Get local source and target vertices
		Vertex localSourceVertex = globalLocalVertexMap.find(globalSourceVertex)->second;
		Vertex localTargetVertex = globalLocalVertexMap.find(globalTargetVertex)->second;

		Edge localEdge;
		bool inserted;

		EdgeId edge_id = globalEdge.localId;

		boost::tie(localEdge,inserted) = boost::add_edge(localSourceVertex,localTargetVertex,*this);

		assert(inserted);

		// Add edge id to edge property
		EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(),*this);
		boost::put(edgePropertyMap,localEdge,edge_id);
	}
	return;
}

/*
 * Return the global entity key given a local subgraph vertex.
 */
entityKey
Subgraph::local_to_global(Vertex localVertex){

	std::map<Vertex,stk::mesh::EntityKey>::const_iterator vertexMapIterator =
			localGlobalVertexMap.find(localVertex);

	assert(vertexMapIterator != localGlobalVertexMap.end());

	return (*vertexMapIterator).second;
}

/*
 * Return local vertex given global entity key.
 */
Vertex
Subgraph::global_to_local(entityKey globalVertexKey){

	std::map<stk::mesh::EntityKey, Vertex>::const_iterator vertexMapIterator =
			globalLocalVertexMap.find(globalVertexKey);

	assert(vertexMapIterator != globalLocalVertexMap.end());

	return (*vertexMapIterator).second;
}

/*
 * Add a vertex in the subgraph.
 *   Mirrors the change in the stk mesh
 *   Input: rank of vertex (entity) to be created
 *   Output: new vertex
 */
Vertex
Subgraph::add_vertex(EntityRank vertex_rank)
{
	// Insert the vertex into the stk mesh
	// First have to request a new entity of rank N
	std::vector<size_t> requests(numDim_+1,0); // number of entity ranks. 1 + number of dimensions
	requests[vertex_rank]=1;
	stk::mesh::EntityVector newEntity;
	bulkData_->generate_new_entities(requests,newEntity);
	stk::mesh::Entity & globalVertex = *(newEntity[0]);

	// Insert the vertex into the subgraph
	Vertex localVertex = boost::add_vertex(*this);

	// Update maps
	localGlobalVertexMap.insert(std::map<Vertex,entityKey>::value_type(localVertex,globalVertex.key()));
	globalLocalVertexMap.insert(std::map<entityKey,Vertex>::value_type(globalVertex.key(),localVertex));

	// store entity rank to the vertex property
	VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(),*this);
	boost::put(vertexPropertyMap,localVertex,vertex_rank);

	return localVertex;
}

/*
 * Remove vertex in subgraph
 *   Mirror in stk mesh
 */
void
Subgraph::remove_vertex(Vertex & vertex)
{
	// get the global entity key of vertex
	entityKey key = local_to_global(vertex);

	// look up entity from key
	stk::mesh::Entity* entity = bulkData_->get_entity(key);

	// remove the vertex and key from globalLocalVertexMap and localGlobalVertexMap
	globalLocalVertexMap.erase(key);
	localGlobalVertexMap.erase(vertex);

	// remove vertex from subgraph
	// first have to ensure that there are no edges in or out of the vertex
	boost::clear_vertex(vertex,*this);
	// remove the vertex
	boost::remove_vertex(vertex,*this);

	// destroy all relations to or from the entity
	stk::mesh::PairIterRelation relations = entity->relations();
	for (int i = 0;	i < relations.size(); ++i){
		EdgeId edgeId = relations[i].identifier();

		stk::mesh::Entity & target = *(relations[i].entity());

		bulkData_->destroy_relation(*entity,target,edgeId);
	}
	// remove the entity from stk mesh
	bool deleted = bulkData_->destroy_entity(entity);
	assert(deleted);

	return;
}

/*
 * Add edge to local graph.
 *   Mirror change in stk mesh
 *   Return true if edge inserted in local graph, else false.
 *   If false, will not insert edge into stk mesh.
 */
std::pair<Edge,bool>
Subgraph::add_edge(const EdgeId edge_id,
		const Vertex localSourceVertex,
		const Vertex localTargetVertex)
{
	// Add edge to local graph
	std::pair<Edge,bool> localEdge = boost::add_edge(localSourceVertex,localTargetVertex,*this);

	if (localEdge.second == false)
		return localEdge;

	// get global entities
	entityKey globalSourceKey = local_to_global(localSourceVertex);
	entityKey globalTargetKey = local_to_global(localTargetVertex);
	stk::mesh::Entity* globalSourceVertex = bulkData_->get_entity(globalSourceKey);
	stk::mesh::Entity* globalTargetVertex = bulkData_->get_entity(globalTargetKey);

	// Add edge to stk mesh
	bulkData_->declare_relation(*(globalSourceVertex),*(globalTargetVertex),edge_id);

	// Add edge id to edge property
	EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(),*this);
	boost::put(edgePropertyMap,localEdge.first,edge_id);

	return localEdge;
}

/*
 * Remove edge from graph
 *   Mirror change in stk mesh
 */
void
Subgraph::remove_edge(
		const Vertex & localSourceVertex,
		const Vertex & localTargetVertex){
	// Get the local id of the edge in the subgraph

	Edge edge;
	bool inserted;
	boost::tie(edge,inserted) =
	boost::edge(localSourceVertex,localTargetVertex,*this);

	assert(inserted);

	EdgeId edge_id = get_edge_id(edge);

	// remove local edge
	boost::remove_edge(localSourceVertex,localTargetVertex,*this);

	// remove relation from stk mesh
	stk::mesh::EntityKey globalSourceId = local_to_global(localSourceVertex);
	stk::mesh::EntityKey globalTargetId = local_to_global(localTargetVertex);
	stk::mesh::Entity* globalSourceVertex = bulkData_->get_entity(globalSourceId);
	stk::mesh::Entity* globalTargetVertex = bulkData_->get_entity(globalTargetId);

	bulkData_->destroy_relation(*(globalSourceVertex),*(globalTargetVertex),edge_id);

	return;
}

EntityRank &
Subgraph::get_vertex_rank(const Vertex vertex)
{
	VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);

	return boost::get(vertexPropertyMap,vertex);
}

EdgeId &
Subgraph::get_edge_id(const Edge edge)
{
	EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(), *this);

	return boost::get(edgePropertyMap,edge);
}

/*
 * The connected components boost algorithm requires an undirected graph.
 *   Subgraph creates a directed graph. Copy all vertices and edges from the
 *   subgraph into the undirected graph.
 */
void
Subgraph::undirected_graph(){
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> undirectedGraph;
	undirectedGraph g;
	VertexIterator vertex_begin;
	VertexIterator vertex_end;
	std::pair<VertexIterator,VertexIterator> vertex_bounds = vertices(*this);
	vertex_begin = vertex_bounds.first;
	vertex_end = vertex_bounds.second;

	for (VertexIterator i = vertex_begin; i != vertex_end; ++i){
		Vertex source = boost::add_vertex(g);

		// write the edges in the subgraph
		out_edge_iterator out_edge_begin;
		out_edge_iterator out_edge_end;
		std::pair<out_edge_iterator,out_edge_iterator> out_edge_bounds = out_edges(*i,*this);
		out_edge_begin = out_edge_bounds.first;
		out_edge_end = out_edge_bounds.second;

		for (out_edge_iterator j = out_edge_begin; j != out_edge_end; ++j){
			 Vertex target;
			 Edge edge = *j;
			 target = boost::target(edge,g);
			 boost::add_edge(source,target,g);
		}
	}

	boost::write_graphviz(std::cout,g);
	std::vector<int> component(boost::num_vertices(g));
	int numComponents = boost::connected_components(g, &component[0]);
	cout << "number of components: " << numComponents << "\n";

	return;
}

/*
 * Clones a boundary entity from the subgraph and separates the in-edges
 *   of the entity.
 *   Boundary entities are on boundary of the elements in the mesh. They
 *   will thus have either 1 or 2 in-edges to elements.
 *
 *   If there is only 1 in-edge, the entity is an exterior entity of the
 *   mesh and is not a candidate for fracture. If only 1 in-edge: Return error.
 *
 *   Entity must have satisfied the fracture criterion and be labeled open
 *   in is_open. If not open: Return error.
 *
 *   TODO: add functionality.
 */
void
Subgraph::clone_boundary_entity(Vertex vertex){

	return;
}

/*
 * Splits an articulation point.
 *   An articulation point is defined as a vertex which if removed
 *   yields a graph with more than 1 connected components. Creates
 *   an undirected graph and checks connected components of graph without
 *   vertex. Check if vertex is articulation point
 *
 *   Clones articulation point and splits in-edges between original and new
 *   vertices.
 *
 *   TODO: add functionality.
 */
void
Subgraph::split_articulation_point(Vertex vertex){
	return;
}

/*
 * Similar to the function in the topology class. Will output the subgraph in a
 * .dot file readable by the graphviz program dot.
 *
 * To create a figure from the file:
 *   dot -Tpng output.dot -o output.png
 */
void
Subgraph::output_to_graphviz(std::map<entityKey,bool> entity_open)
{
	// Open output file
	std::ofstream gviz_out;
	gviz_out.open ("output.dot", std::ios::out);

	cout << "Write graph to graphviz dot file\n";

	if (gviz_out.is_open()){
		// Write beginning of file
		gviz_out << "digraph mesh {\n"
				 << "node [colorscheme=paired12\n"
				 << "edge [colorscheme=paired12\n";

		VertexIterator vertex_begin;
		VertexIterator vertex_end;
		std::pair<VertexIterator,VertexIterator> vertex_bounds = vertices(*this);
		vertex_begin = vertex_bounds.first;
		vertex_end = vertex_bounds.second;


		for (VertexIterator i = vertex_begin; i != vertex_end; ++i){
			entityKey key = local_to_global(*i);
			stk::mesh::Entity & entity = *(bulkData_->get_entity(key));
			std::string label;
			std::string color;

			// Write the entity name
			switch(entity.entity_rank()){
			// nodes
			case 0:
				label = "Node";
				if (entity_open[entity.key()]==false)
					color = "6";
				else
					color = "5";
				break;
			// segments
			case 1:
				label = "Segment";
				if (entity_open[entity.key()]==false)
					color = "4";
				else
					color = "3";
				break;
			// faces
			case 2:
				label = "Face";
				if (entity_open[entity.key()]==false)
					color = "2";
				else
					color = "1";
				break;
			// volumes
			case 3:
				label = "Element";
				if (entity_open[entity.key()]==false)
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
			std::pair<out_edge_iterator,out_edge_iterator> out_edge_bounds = out_edges(*i,*this);
			out_edge_begin = out_edge_bounds.first;
			out_edge_end = out_edge_bounds.second;

			for (out_edge_iterator j = out_edge_begin; j != out_edge_end; ++j){
				Edge out_edge = *j;
				Vertex source = boost::source(out_edge,*this);
				Vertex target = boost::target(out_edge,*this);

				entityKey sourceKey = local_to_global(source);
				stk::mesh::Entity & global_source = *(bulkData_->get_entity(sourceKey));

				entityKey targetKey = local_to_global(target);
				stk::mesh::Entity & global_target = *(bulkData_->get_entity(targetKey));

				EdgeId edgeId = get_edge_id(out_edge);

				switch(edgeId){
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
				gviz_out << "  \"" << global_source.identifier() << "_" << global_source.entity_rank()
						 << "\" -> \"" << global_target.identifier() << "_" << global_target.entity_rank()
						 << "\" [color=\"" << color << "\"]" << "\n";
			}

		}

		// File end
		gviz_out << "}";
		gviz_out.close();
	}
	else
		cout << "Unable to open graphviz output file 'output.dot'\n";

	return;
}

} // namespace LCM

