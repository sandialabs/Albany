//
// Test of mesh manipulation.
// Separate all elements of a mesh by nodal replacement
//

#include "Topology.h"

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

  LCM::topology
  topology(input_file,output_file);

  stk::mesh::BulkData&
  bulkData = *(topology.get_BulkData());

  // Node rank should be 0 and element rank should be equal to the dimension of the
  // system (e.g. 2 for 2D meshes and 3 for 3D meshes)
  //cout << "Node Rank: "<< nodeRank << ", Element Rank: " << elementRank << "\n";

  // Print element connectivity before the mesh topology is modified
  cout << "*************************\n"
	   << "Before element separation\n"
	   << "*************************\n";
  topology.disp_connectivity();

  // Start the mesh update process
  //   Will fully separate the elements in the mesh by replacing element nodes
  //   Get a vector containing the element set of the mesh.
  std::vector<stk::mesh::Entity*> element_lst;
  stk::mesh::get_entities(bulkData,topology.elementRank,element_lst);

  // Creates the graph
  std::vector<std::vector<stk::mesh::Entity*> > connectivity_temp;
  topology.graph_initialization(connectivity_temp);

  // *************
  // test
  // *************
  // Create a subgraph
  // SubGraph requires two vectors, a list of elements and a list of edges
  stk::mesh::Entity & node = *(bulkData.get_entity(0,5));
  std::set<LCM::entityKey> subgraph_entity_lst;
  std::set<LCM::topology::stkEdge,LCM::topology::EdgeLessThan> subgraph_edge_lst;
  topology.star(subgraph_entity_lst,subgraph_edge_lst,node);
  // test of connected components. want more than 1 subgraph
  stk::mesh::Entity & node2 = *(bulkData.get_entity(0,27));
  topology.star(subgraph_entity_lst,subgraph_edge_lst,node2);

  // Iterators
  std::set<LCM::entityKey>::iterator firstEntity = subgraph_entity_lst.begin();
  std::set<LCM::entityKey>::iterator lastEntity = subgraph_entity_lst.end();
  std::set<LCM::topology::stkEdge>::iterator firstEdge = subgraph_edge_lst.begin();
  std::set<LCM::topology::stkEdge>::iterator lastEdge = subgraph_edge_lst.end();

  // create subgrah
  int numDim = topology.numDim;
  LCM::Subgraph
  subgraph(&bulkData,firstEntity,lastEntity,firstEdge,lastEdge,numDim);

  // test the functions of the class
  bulkData.modification_begin();
  // add entity
  LCM::Vertex new_vertex = subgraph.add_vertex(topology.faceRank);
  LCM::entityKey new_entity_key = subgraph.local_to_global(new_vertex);
  stk::mesh::Entity & new_entity = *(bulkData.get_entity(new_entity_key));
  // check that new entity is in stk mesh
  cout << "new entity: " << new_entity.entity_rank() << "," << new_entity.identifier() << "\n";

  // add edge
  stk::mesh::Entity & element = *(bulkData.get_entity(topology.elementRank,1));
  LCM::Vertex source = subgraph.global_to_local(element.key());
  LCM::Vertex target = subgraph.global_to_local(new_entity.key());
  subgraph.add_edge(6,source,target);
  //check that new edge is in the stk mesh
  stk::mesh::PairIterRelation relations = element.relations(topology.faceRank);
  cout << "added edge\n";
  for (int i = 0; i < relations.size(); ++i)
	  cout << "target: " << relations[i].entity_rank() << "," <<
	  	  relations[i].entity()->identifier() << " ID: " << relations[i].identifier() << "\n";

  // remove the new edge
  subgraph.remove_edge(source,target);
  // check that the edge is no longer in the stk mesh
  relations = element.relations(topology.faceRank);
  cout << "removed edge\n";
  for (int i = 0; i < relations.size(); ++i)
	  cout << "target: " << relations[i].entity_rank() << "," <<
	  	  relations[i].entity()->identifier() << " ID: " << relations[i].identifier() << "\n";
  cout << "remove entity "<< new_entity.entity_rank()  << "," << new_entity.identifier() << "\n";
  subgraph.remove_vertex(new_vertex);
  bulkData.modification_end();

  // Check for failure criterion
  std::map<stk::mesh::EntityKey, bool> entity_open;
  topology.set_entities_open(entity_open);

  // Output subgraph to graphviz
  subgraph.output_to_graphviz(entity_open);
  subgraph.undirected_graph();
  // *************
  // end test
  // *************

  bulkData.modification_begin();

  // Loop over the elements
  for (int i = 0; i < element_lst.size(); ++i){
	  stk::mesh::Entity & current_element = *(element_lst[i]);
	  stk::mesh::PairIterRelation face_lst = element_lst[i]->relations(topology.elementRank - 1);

	  // Loop over faces
	  for (int j = 0; j < face_lst.size(); ++j){
		  stk::mesh::Entity & current_face = *(face_lst[j].entity());
		  topology.duplicate_entity(current_element, current_face,
				  connectivity_temp,j,current_element);
	  }
  }

  // Need to remove added mesh entities before updating Albany stk discretization
  topology.graph_cleanup(connectivity_temp);

  // End mesh update
  bulkData.modification_end();


  cout << "*************************\n"
	   << "After element separation\n"
	   << "*************************\n";
  topology.disp_connectivity();

  // Need to update the mesh to reflect changes in duplicate_entity routine.
  //   Redefine connectivity and coordinate arrays with updated values.
  //   Mesh must only have relations between elements and nodes.
  Teuchos::RCP<Albany::AbstractDiscretization> discretization_ptr = topology.get_Discretization();
  Albany::STKDiscretization & stk_discretization = static_cast<Albany::STKDiscretization &>(*discretization_ptr);

  Teuchos::RCP<Epetra_Comm>
    communicator = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
  Teuchos::RCP<Albany::AbstractSTKMeshStruct>
  stkMeshStruct = topology.get_stkMeshStruct();

  stk_discretization.updateMesh(stkMeshStruct,communicator);
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >
    element_connectivity = discretization_ptr->getWsElNodeID();
  Teuchos::ArrayRCP<double>
    coordinates = stk_discretization.getCoordinates();

  // Separate the elements of the mesh to illustrate the
  //   disconnected nature of the final mesh

  // Create a vector to hold displacement values for nodes
  Teuchos::RCP<const Epetra_Map> dof_map = stk_discretization.getMap();
  Epetra_Vector displacement = Epetra_Vector(*(dof_map),true);

  // Add displacement to nodes
  stk::mesh::get_entities(bulkData,topology.elementRank,element_lst);
  for (int i = 0; i < element_lst.size(); ++i){
	  std::vector<double> centroid(3);
	  std::vector<double> disp(3);
	  stk::mesh::PairIterRelation relations = element_lst[i]->relations(topology.nodeRank);
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
