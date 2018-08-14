//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//
// Find connected components in a mesh by using the dual graph
//
#include <Teuchos_CommandLineProcessor.hpp>
#include <iomanip>

#include <LCMPartition.h>

int
main(int ac, char* av[])
{
  //
  // Create a command line processor and parse command line options
  //
  Teuchos::GlobalMPISession     mpiSession(&ac, &av);
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString(
      "Computation of connected components of an Exodus mesh.\n");

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
  // Read mesh
  //
  LCM::ConnectivityArray connectivity_array(input_file, output_file);

  //
  // Create dual graph
  //
  LCM::DualGraph dual_graph(connectivity_array);

  //
  // Compute connected components
  //
  std::vector<int> components;

  const int number_components = dual_graph.getConnectedComponents(components);

  // Get abstract discretization from connectivity array and convert
  // to stk discretization to use stk-specific methods.
  Albany::AbstractDiscretization& discretization =
      connectivity_array.getDiscretization();

  Albany::STKDiscretization& stk_discretization =
      static_cast<Albany::STKDiscretization&>(discretization);

  // Get MDArray which is memeopru in stk for "Partition" element variable
  Albany::MDArray stk_component =
      stk_discretization.getStateArrays().elemStateArrays[0]["Partition"];

  //
  // Output components
  //

  // Assumption: numbering of elements is contiguous.
  for (std::vector<int>::size_type element = 0; element < components.size();
       ++element) {
    const int component = components[element];

    // set component number in stk field memory
    stk_component[element] = component;
  }

  // Need solution for output call
  Teuchos::RCP<Tpetra_Vector> solution_fieldT =
      stk_discretization.getSolutionFieldT();

  // second arg to output is (pseudo)time
  //  stk_discretization.outputToExodus(*solution_field, 1.0);
  stk_discretization.writeSolutionT(*solution_fieldT, 1.0);

  const int number_elements = connectivity_array.getNumberElements();

  std::cout << std::endl;
  std::cout << "==========================================";
  std::cout << std::endl;
  std::cout << "Number of Elements        : " << number_elements << std::endl;
  std::cout << std::endl;
  std::cout << "Number of Mesh Components : " << number_components << std::endl;
  std::cout << "------------------------------------------";
  std::cout << std::endl;
  std::cout << "Element Component";
  std::cout << std::endl;
  std::cout << "------------------------------------------";
  std::cout << std::endl;

  for (std::vector<int>::size_type element = 0; element < components.size();
       ++element) {
    std::cout << std::setw(8) << element;
    std::cout << std::setw(8) << components[element] << std::endl;
  }

  std::cout << "==========================================";
  std::cout << std::endl;

  return 0;
}
