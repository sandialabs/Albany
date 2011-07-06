//
// Test of mesh manipulation.
// Separate all elements of a mesh by nodal replacement
//

#if defined (ALBANY_LCM)

#include <iostream>
#include <iomanip>
#include <Teuchos_CommandLineProcessor.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/fem/CreateAdjacentEntities.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <stk/Albany_AbstractDiscretization.hpp>
#include <stk/Albany_DiscretizationFactory.hpp>
#include <stk/Albany_STKDiscretization.hpp>
#include <Albany_Utils.hpp>

//#include <Partition.h>

// Display the relations to a given entity
void disp_relation(stk::mesh::Entity & entity)
{
	cout << "Relations for entity (identifier,rank): " << entity.identifier()
			<< "," << entity.entity_rank() << "\n";
	stk::mesh::PairIterRelation relations = entity.relations();
		for (int i = 0; i < relations.size(); ++i){
			cout << "entity:\t" << relations[i].entity()->identifier() << ","
				 << relations[i].entity()->entity_rank() << "\tlocal id: "
				 << relations[i].identifier() << "\n";
		}
}

void disp_relation(stk::mesh::Entity & entity, stk::mesh::EntityRank entityRank)
{
	cout << "Relations of rank " << entityRank << " for entity (identifier,rank): "
			<< entity.identifier() << "," << entity.entity_rank() << "\n";
	stk::mesh::PairIterRelation relations = entity.relations(entityRank);
		for (int i = 0; i < relations.size(); ++i){
			cout << "entity:\t" << relations[i].entity()->identifier() << ","
				 << relations[i].entity()->entity_rank() << "\tlocal id: "
				 << relations[i].identifier() << "\n";
		}
}

// Outputs the element connectivity
void disp_connectivity(
		stk::mesh::BulkData & bulkData,
		const stk::mesh::EntityRank elementRank,
		const stk::mesh::EntityRank nodeRank)
{
	// Create a list of element entities
	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(bulkData,elementRank,element_lst);

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

	//disp_relation(*(element_lst[0]));

	//std::vector<stk::mesh::Entity*> face_lst;
	//stk::mesh::get_entities(bulkData,elementRank-1,face_lst);
	//disp_relation(*(face_lst[1]));

	return;
}

// Creates the full graph representation of the mesh.
//   Connectivity data stored in connectivty_temp
void graph_initialization(stk::mesh::BulkData & bulkData,
		std::vector<std::vector<stk::mesh::Entity*> >  & connectivity_temp,
		const stk::mesh::EntityRank elementRank,
		const stk::mesh::EntityRank nodeRank){

	stk::mesh::PartVector add_parts;
	stk::mesh::create_adjacent_entities(bulkData, add_parts);

	// Create the temporary connectivity array
	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(bulkData,elementRank,element_lst);

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
	// Remove the unneeded relationships (connections between entities with
	//   rank difference of more than 1 e.g. element and node)
	bulkData.modification_begin();

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
			bulkData.destroy_relation(element,entity,del_ids[j]);
		}
	};

	if (elementRank == 3){
		// Remove extra relations from face
		std::vector<stk::mesh::Entity*> face_lst;
		stk::mesh::get_entities(bulkData,elementRank-1,face_lst);
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
				bulkData.destroy_relation(face,entity,del_ids[j]);
			}
		}
	}

	bulkData.modification_end();
	}

	return;
}

// Removes the extra entities and relations added in graph_initialization.
//   Recreates the expected mesh connectivity through relations between
//   elements and nodes.
void graph_cleanup(stk::mesh::BulkData & bulkData,
		std::vector<std::vector<stk::mesh::Entity*> >  & connectivity_temp,
		const stk::mesh::EntityRank elementRank,
		const stk::mesh::EntityRank faceRank,
		const stk::mesh::EntityRank edgeRank,
		const stk::mesh::EntityRank nodeRank){

	std::vector<stk::mesh::Entity*> element_lst;
	stk::mesh::get_entities(bulkData,elementRank,element_lst);
	// Remove relations of faces and edges from graph
	for (int i = 0; i < element_lst.size(); ++i){
		// Remove element,face relations
		stk::mesh::Entity & element = *(element_lst[i]);
		stk::mesh::PairIterRelation faces = element.relations(faceRank);
		std::vector<stk::mesh::Entity*> del_face_relations;
		std::vector<int> del_face_ids;
		for (int j = 0; j < faces.size(); ++j){
			// Remove face,edge relations
			stk::mesh::Entity & face = *(faces[j].entity());
			del_face_relations.push_back(&face);
			del_face_ids.push_back(faces[j].identifier());

			stk::mesh::PairIterRelation edges = face.relations(edgeRank);
			std::vector<stk::mesh::Entity*> del_edge_relations;
			std::vector<int> del_edge_ids;
			for(int k = 0; k < edges.size(); ++k){
				// Remove edge,node relations
				stk::mesh::Entity & edge = *(edges[k].entity());
				del_face_relations.push_back(&edge);
				del_face_ids.push_back(edges[k].identifier());

				stk::mesh::PairIterRelation nodes = edge.relations(nodeRank);
				std::vector<stk::mesh::Entity*> del_node_relations;
				std::vector<int> del_node_ids;
				for (int m = 0; m < nodes.size(); ++m){
					stk::mesh::Entity & node = *(nodes[m].entity());
					del_node_relations.push_back(&node);
					del_node_ids.push_back(nodes[m].identifier());
				}

				for (int m = 0; m < del_node_ids.size(); ++m){
					stk::mesh::Entity & node = *(del_node_relations[m]);
					bulkData.destroy_relation(edge,node,del_node_ids[m]);
				}
			}

			for (int k = 0; k < del_edge_ids.size(); ++k){
				stk::mesh::Entity & edge = *(del_edge_relations[k]);
				bulkData.destroy_relation(face,edge,del_edge_ids[k]);
			}

		}

		for (int j = 0; j < del_face_ids.size(); ++j){
			stk::mesh::Entity & face = *(del_face_relations[j]);
			bulkData.destroy_relation(element,face,del_face_ids[j]);
		}
	}

	// Check for and remove extra relationships between element and edges
	for (int i = 0; i < element_lst.size(); ++i){
		stk::mesh::Entity & element = *(element_lst[i]);
		stk::mesh::PairIterRelation edges = element.relations(edgeRank);

		std::vector<stk::mesh::Entity*> del_relations;
		std::vector<int> del_ids;
		for (int j = 0; j < edges.size(); ++j){
			del_relations.push_back(edges[j].entity());
			del_ids.push_back(edges[j].identifier());
		}

		for (int j = 0; j < del_ids.size(); ++j)
			bulkData.destroy_relation(element,*(del_relations[j]),del_ids[j]);
	}

	// Remove faces from graph
	std::vector<stk::mesh::Entity*> face_lst;
	stk::mesh::get_entities(bulkData,faceRank,face_lst);
	for (int i = 0; i < face_lst.size(); ++i){
		bulkData.destroy_entity(face_lst[i]);
	}

	// Remove edges from graph
	std::vector<stk::mesh::Entity*> edge_lst;
	stk::mesh::get_entities(bulkData,edgeRank,edge_lst);
	for (int i = 0; i < edge_lst.size(); ++i){
		bulkData.destroy_entity(edge_lst[i]);
	}

	// Add relations from element to nodes
	for (int i = 0; i < element_lst.size(); ++i){
		stk::mesh::Entity & element = *(element_lst[i]);
		for (int j = 0; j < connectivity_temp.size(); ++j){
			stk::mesh::Entity & node = *(connectivity_temp[i][j]);
			bulkData.declare_relation(element,node,j);
		}
	}
	return;
}

// Duplicates old_entity and removes connection between entity_plus and old_entity.
//   A new entity with the same out_edges (i.e. connectivity to lower order entities) is created.
//   old_entity is the original entity before the duplication
//   in_entity is a higher order entity with an out_edge connected to old_entity
void duplicate_entity(stk::mesh::Entity & in_entity,
		stk::mesh::Entity & old_entity,
		stk::mesh::BulkData & bulkData,
		std::vector<std::vector<stk::mesh::Entity*> > & connectivity_temp,
		const stk::mesh::EntityRank elementRank,
		int local_id,
		stk::mesh::Entity & element)
{
	// Rank of the entity
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
		bulkData.generate_new_entities(requests,newEntities);

		stk::mesh::Entity & new_entity = *(newEntities[0]);

		// Remove relation between in_entity and old_entity (old_entity is no
		//   longer associated with in_entity)
		bulkData.destroy_relation(in_entity,old_entity,local_id);

		// Declare relation between in_entity and new_entity
		bulkData.declare_relation(in_entity,new_entity,local_id);

		// The connection to the element
		bulkData.destroy_relation(element,old_entity,id);
		bulkData.declare_relation(element,new_entity,id);

		// Duplicate the parameters of old_entity to new_entity
		bulkData.copy_entity_fields(old_entity,new_entity);


		if (entity_rank > 0){
			// Copy the out_edges of old_entity to new_entity
			for (int i = 0; i < out_edges.size(); ++i){
				bulkData.declare_relation(new_entity, *(out_edges[i].entity()),i);
			}

			for (int i = 0; i < out_edges.size(); ++i){
				duplicate_entity(new_entity,
						*(out_edges[i].entity()),
						bulkData,connectivity_temp,
						elementRank,i,element);

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


int main(int ac, char* av[])
{

  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor
  command_line_processor;

  command_line_processor.setDocString(
      "Test of element separation through nodal insertion.\n"
	  "Remove and replace all nodes in elements.\n");

  std::string input_file = "input.e";
  command_line_processor.setOption(
      "input",
      &input_file,
      "Input File Name");

  std::string output_file = "output.e";
  command_line_processor.setOption(
      "output",
      &output_file,
      "Output File Name");

  // Throw a warning and not error for unrecognized options
  command_line_processor.recogniseAllOptions(true);

  // Don't throw exceptions for errors
  command_line_processor.throwExceptions(false);

  // Parse command line
  Teuchos::CommandLineProcessor::EParseCommandLineReturn
  parse_return = command_line_processor.parse(ac, av);

  if (parse_return == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
    return 0;
  }

  if (parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return 1;
  }

  //
  // Read the mesh
  //
  // Copied from Partition.cc
  Teuchos::GlobalMPISession mpiSession(&ac,&av);


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

  Teuchos::RCP<Albany::AbstractDiscretization>
    discretization_ptr;

  discretization_ptr = disc_factory.createDiscretization(1, stateInfo);

  // Dimensioned: Workset, Cell, Local Node
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >
	  element_connectivity = discretization_ptr->getWsElNodeID();

  Teuchos::ArrayRCP<double>
	  coordinates = discretization_ptr->getCoordinates();

  // Need to access the bulkData and metaData classes in the mesh datastructure
  Albany::STKDiscretization &
  stk_discretization = static_cast<Albany::STKDiscretization &>(*discretization_ptr);

  Teuchos::RCP<Albany::AbstractSTKMeshStruct>
  stkMeshStruct = stk_discretization.getSTKMeshStruct();

  stk::mesh::BulkData & bulkData = *stkMeshStruct->bulkData;
  stk::mesh::fem::FEMMetaData & metaData = *stkMeshStruct->metaData;

  // The entity ranks
  const stk::mesh::EntityRank nodeRank = metaData.NODE_RANK;
  const stk::mesh::EntityRank edgeRank = metaData.EDGE_RANK;
  const stk::mesh::EntityRank faceRank = metaData.FACE_RANK;
  const stk::mesh::EntityRank elementRank = metaData.element_rank();

  // Node rank should be 0 and element rank should be equal to the dimension of the
  // system (e.g. 2 for 2D meshes and 3 for 3D meshes)
  //cout << "Node Rank: "<< nodeRank << ", Element Rank: " << elementRank << "\n";

  // Print element connectivity before the mesh topology is modified
  cout << "*************************\n"
	   << "Before element separation\n"
	   << "*************************\n";
  disp_connectivity(bulkData, elementRank, nodeRank);

  // Start the mesh update process
  //   Will fully separate the elements in the mesh by replacing element nodes
  //   Get a vector containing the element set of the mesh.
  std::vector<stk::mesh::Entity*> element_lst;
  stk::mesh::get_entities(bulkData,elementRank,element_lst);

  // Creates the graph
  std::vector<std::vector<stk::mesh::Entity*> > connectivity_temp;
  graph_initialization(bulkData, connectivity_temp, elementRank, nodeRank);

  bulkData.modification_begin();


  // Loop over the elements
  for (int i = 0; i < element_lst.size(); ++i){
	  stk::mesh::Entity & current_element = *(element_lst[i]);

	  stk::mesh::PairIterRelation face_lst = element_lst[i]->relations(elementRank - 1);

	  // Loop over faces
	  for (int j = 0; j < face_lst.size(); ++j){
		  stk::mesh::Entity & current_face = *(face_lst[j].entity());
		  duplicate_entity(current_element, current_face, bulkData,
				  connectivity_temp,elementRank,j,current_element);
	  }
  }


  // Need to remove added mesh entities before updating Albany stk discretization
  graph_cleanup(bulkData,connectivity_temp,elementRank,faceRank,edgeRank,nodeRank);

  // End mesh update
  bulkData.modification_end();

  cout << "*************************\n"
	   << "After element separation\n"
	   << "*************************\n";
  disp_connectivity(bulkData, elementRank, nodeRank);

  // Need to update the mesh to reflect changes in duplicate_entity routine.
  //   Redefine connectivity and coordinate arrays with updated values.
  //   Mesh must only have relations between elements and nodes.
  stk_discretization.updateMesh(stkMeshStruct,communicator);
  element_connectivity = discretization_ptr->getWsElNodeID();
  coordinates = stk_discretization.getCoordinates();

  // Separate the elements of the mesh to illustrate the
  //   disconnected nature of the final mesh

  // Create a vector to hold displacement values for nodes
  Teuchos::RCP<const Epetra_Map> dof_map = stk_discretization.getMap();
  Epetra_Vector displacement = Epetra_Vector(*(dof_map),true);

  // Add displacement to nodes
  stk::mesh::get_entities(bulkData,elementRank,element_lst);
  for (int i = 0; i < element_lst.size(); ++i){
	  std::vector<double> centroid(3);
	  std::vector<double> disp(3);
	  stk::mesh::PairIterRelation relations = element_lst[i]->relations(nodeRank);
	  // Get centroid of the element
	  for (int j = 0; j < relations.size(); ++j){
		  stk::mesh::Entity & node = *(relations[j].entity());
		  int id = static_cast<int>(node.identifier());
		  centroid[0] += coordinates[id*3-3];
		  centroid[1] += coordinates[id*3-2];
		  centroid[2] += coordinates[id*3-1];
	  }
	  centroid[0] /= relations.size();
	  centroid[1] /= relations.size();
	  centroid[2] /= relations.size();

	  // Determine displacement
	  for (int j = 0; j < 3; ++j){
		  if (centroid[j] < 0)
			  disp[j] = -0.5;
		  else if (centroid[j] > 0)
			  disp[j] =  0.5;
		  else
			  disp[j] = 0.0;
	  }

	  // Add displacement to nodes
	  for (int j = 0; j < relations.size(); ++j){
		  stk::mesh::Entity & node = *(relations[j].entity());
		  int id = static_cast<int>(node.identifier());
		  displacement[id*3-3] += disp[0];
		  displacement[id*3-2] += disp[1];
		  displacement[id*3-1] += disp[2];
	  }
  }

  stk_discretization.setResidualField(displacement);

  Teuchos::RCP<Epetra_Vector>
  solution_field = stk_discretization.getSolutionField();

  // Write final mesh to exodus file
  stk_discretization.outputToExodus(*solution_field);

  return 0;

}

#else // #if defined (ALBANY_LCM)

int main(int ac, char* av[])
{
  return 0;
}

#endif // #if defined (ALBANY_LCM)
