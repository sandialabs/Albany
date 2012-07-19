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
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString("Test of barycentric subdivision.\n"
      "Reads in a mesh and applies the barycentric subdivision algorithm.\n"
      "Restricted to simplicial complexes.\n");

  std::string input_file = "input.e";
  command_line_processor.setOption("input", &input_file, "Input File Name");

  std::string output_file = "output.e";
  command_line_processor.setOption("output", &output_file, "Output File Name");

  // Throw a warning and not error for unrecognized options
  command_line_processor.recogniseAllOptions(true);

  // Don't throw exceptions for errors
  command_line_processor.throwExceptions(false);

  // Parse command line
  Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return =
      command_line_processor.parse(ac, av);

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
  Teuchos::GlobalMPISession mpiSession(&ac, &av);

  LCM::topology topology(input_file, output_file);

  // Node rank should be 0 and element rank should be equal to the dimension of the
  // system (e.g. 2 for 2D meshes and 3 for 3D meshes)
  //cout << "Node Rank: "<< nodeRank << ", Element Rank: " << elementRank << "\n";

  // Print element connectivity before the mesh topology is modified
  std::cout << "***********************" << std::endl;
  std::cout << "Before mesh subdivision" << std::endl;
  std::cout << "***********************" << std::endl;

  topology.disp_connectivity();

  // Start the mesh update process

  // Prepares mesh for barycentric subdivision
  topology.remove_node_relations();

  // Output graph structure for debugging
  std::string gviz_output = "before.dot";
  topology.output_to_graphviz(gviz_output);

  //
  // Here starts the barycentric subdivision.
  //
  //-----------------------------------------------------------------------------------------------------------------------------------
  // Generate the output file
  //-----------------------------------------------------------------------------------------------------------------------------------

  Teuchos::RCP<Albany::AbstractDiscretization> discretization_ptr =
      topology.get_Discretization();
  Albany::STKDiscretization & stk_discretization =
      static_cast<Albany::STKDiscretization &>(*discretization_ptr);

  topology.barycentric_subdivision();

  std::cout << "*************************" << std::endl;
  std::cout << "After element subdivision" << std::endl;
  std::cout << "*************************" << std::endl;

  gviz_output = "after.dot";
  topology.output_to_graphviz(gviz_output);

  // Recreates connectivity in stk mesh expected by Albany_STKDiscretization
  // Must be called each time at conclusion of mesh modification
  topology.graph_cleanup();
  topology.disp_connectivity();

  Teuchos::RCP<Epetra_Vector> solution_field =
      stk_discretization.getSolutionField();

  // Write final mesh to exodus file
  // second arg to output is (pseudo)time
  stk_discretization.outputToExodus(*solution_field, 1.0);

  return 0;

}
