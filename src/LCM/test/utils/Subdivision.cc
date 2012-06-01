//
// Test of barycentric subdivision.
// Reads in a mesh and applies the barycentric subdivision algorithm
// to it. Restricted to simplicial complexes.
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
      "Test of barycentric subdivision.\n"
	  "Reads in a mesh and applies the barycentric subdivision algorithm.\n"
	  "Restricted to simplicial complexes.\n");

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
	   << "Before mesh subdivision\n"
	   << "*************************\n";
  topology.disp_connectivity();

  // Start the mesh update process

  // Will fully separate the elements in the mesh by replacing element nodes
  // Get a vector containing the element set of the mesh.
  std::vector<stk::mesh::Entity*> element_lst;
  stk::mesh::get_entities(bulkData,topology.elementRank,element_lst);

  // Prepares mesh for barycentric subdivision
  // Function must be called each time before there are changes to the mesh
  topology.graph_initialization();
  topology.remove_node_relations();

  // Output graph structure for debugging
  std::string gviz_output = "before.dot";
  topology.output_to_graphviz(gviz_output);

  // test the functions of the class
  bulkData.modification_begin();

  //
  // Here starts the barycentric subdivision.
  //
  topology.barycentric_subdivision();

  cout << "*************************\n"
	   << "After element subdivision\n"
	   << "*************************\n";
  topology.disp_connectivity();

  gviz_output = "after.dot";
  topology.output_to_graphviz(gviz_output);

  // Recreates connectivity in stk mesh expected by Albany_STKDiscretization
  // Must be called each time at conclusion of mesh modification
  topology.graph_cleanup();

  // End mesh update
  bulkData.modification_end();



#if 0

  // Need to update the mesh to reflect changes in duplicate_entity routine.
  //   Redefine connectivity and coordinate arrays with updated values.
  //   Mesh must only have relations between elements and nodes.
  Teuchos::RCP<Albany::AbstractDiscretization> discretization_ptr =
		  topology.get_Discretization();
  Albany::STKDiscretization & stk_discretization =
		  static_cast<Albany::STKDiscretization &>(*discretization_ptr);

  Teuchos::ArrayRCP<double>
    coordinates = stk_discretization.getCoordinates();

  // Separate the elements of the mesh to illustrate the
  //   disconnected nature of the final mesh

  // Create a vector to hold displacement values for nodes
  Teuchos::RCP<const Epetra_Map> dof_map = stk_discretization.getMap();
  Epetra_Vector displacement = Epetra_Vector(*(dof_map),true);

  // Add displacement to nodes
  stk::mesh::get_entities(bulkData,topology.elementRank,element_lst);

  // displacement scale factor
  double alpha = 0.5;

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
		  disp[j] = alpha*centroid[j];
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
  // second arg to output is (pseudo)time
  stk_discretization.outputToExodus(*solution_field, 1.0);

#endif

  return 0;

}
