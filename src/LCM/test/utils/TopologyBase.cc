//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//
// Test of topology manipulation.
//
#include "topology/Topology.h"
#include "topology/Topology_Utils.h"

int main(int ac, char* av[])
{
  // Create a command line processor and parse command line options
  Teuchos::CommandLineProcessor
  command_line_processor;

  command_line_processor.setDocString("Test topology manipulation.\n");

  std::string
  input_file = "input.e";

  command_line_processor.setOption(
                                   "input",
                                   &input_file,
                                   "Input File Name");

  std::string
  output_file = "output.e";

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
  Teuchos::GlobalMPISession
  mpiSession(&ac,&av);

  LCM::Topology
  topology(input_file, output_file);

  topology.createBoundary();
  topology.outputBoundary();

  std::string
  gviz_filename_uu = LCM::parallelize_string("output-uu") + ".dot";
  topology.outputToGraphviz(
      gviz_filename_uu,
      LCM::Topology::UNIDIRECTIONAL_UNILEVEL);

  std::string
  gviz_filename_um = LCM::parallelize_string("output-um") + ".dot";
  topology.outputToGraphviz(
      gviz_filename_um,
      LCM::Topology::UNDIRECTIONAL_MULTILEVEL);

  std::string
  gviz_filename_bu = LCM::parallelize_string("output-bu") + ".dot";
  topology.outputToGraphviz(
      gviz_filename_bu,
      LCM::Topology::BIDIRECTIONAL_UNILEVEL);

  std::string
  gviz_filename_bm = LCM::parallelize_string("output-bm") + ".dot";
  topology.outputToGraphviz(
      gviz_filename_bm,
      LCM::Topology::BIDIRECTIONAL_MULTILEVEL);

  Teuchos::RCP<Albany::AbstractDiscretization>
  discretization_ptr = topology.getDiscretization();

  Albany::STKDiscretization &
  stk_discretization =
      static_cast<Albany::STKDiscretization &>(*discretization_ptr);

  // Need solution for output call
  Teuchos::RCP<Epetra_Vector>
  solution_field = stk_discretization.getSolutionField();

  // second arg to output is (pseudo)time
  stk_discretization.writeSolution(*solution_field, 1.0);

  return 0;
}




