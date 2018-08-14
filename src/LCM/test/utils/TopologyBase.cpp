//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//
// Test of topology manipulation.
//
#include "topology/Topology.h"
#include "topology/Topology_FractureCriterion.h"
#include "topology/Topology_Utils.h"

int
main(int ac, char* av[])
{
  // Create a command line processor and parse command line options
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString("Test topology manipulation.\n");

  std::string input_file = "input.e";

  command_line_processor.setOption("input", &input_file, "Input File Name");

  std::string output_file = "output.e";

  command_line_processor.setOption("output", &output_file, "Output File Name");

  int const num_criteria = 3;

  LCM::fracture::Criterion const criteria_values[] = {
      LCM::fracture::ONE, LCM::fracture::RANDOM, LCM::fracture::TRACTION};

  char const* criteria_names[] = {"one", "random", "traction"};

  LCM::fracture::Criterion fracture_criterion = LCM::fracture::RANDOM;

  command_line_processor.setOption(
      "fracture-criterion",
      &fracture_criterion,
      num_criteria,
      criteria_values,
      criteria_names,
      "Fracture Criterion");

  double probability = 1.0;

  command_line_processor.setOption("probability", &probability, "Probability");

  int const num_styles = 4;

  LCM::Topology::OutputType const style_values[] = {
      LCM::Topology::UNIDIRECTIONAL_UNILEVEL,
      LCM::Topology::UNIDIRECTIONAL_MULTILEVEL,
      LCM::Topology::BIDIRECTIONAL_UNILEVEL,
      LCM::Topology::BIDIRECTIONAL_MULTILEVEL};

  char const* style_names[] = {"UU", "UM", "BU", "BM"};

  LCM::Topology::OutputType plot_style = LCM::Topology::UNIDIRECTIONAL_UNILEVEL;

  command_line_processor.setOption(
      "plot-style",
      &plot_style,
      num_styles,
      style_values,
      style_names,
      "Plot Style");

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
  Teuchos::GlobalMPISession mpiSession(&ac, &av);

  LCM::Topology topology(input_file, output_file);

  Teuchos::RCP<LCM::AbstractFractureCriterion> abstract_fracture_criterion;

  switch (fracture_criterion) {
    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Unknown or unsupported fracture criterion: ";
      std::cerr << fracture_criterion;
      std::cerr << '\n';
      exit(1);
      break;

    case LCM::fracture::ONE:
      abstract_fracture_criterion =
          Teuchos::rcp(new LCM::FractureCriterionOnce(topology, probability));
      break;

    case LCM::fracture::RANDOM:
      abstract_fracture_criterion =
          Teuchos::rcp(new LCM::FractureCriterionRandom(topology, probability));
      break;
  }

#if defined(DEBUG_LCM_TOPOLOGY)
  std::string gviz_filename = LCM::parallelize_string("initial") + ".dot";
  topology.outputToGraphviz(gviz_filename);
#endif

  topology.set_fracture_criterion(abstract_fracture_criterion);

  topology.setEntitiesOpen();

  topology.set_output_type(plot_style);

  topology.splitOpenFaces();

  Teuchos::RCP<Albany::AbstractDiscretization> discretization_ptr =
      topology.get_discretization();

  Albany::STKDiscretization& stk_discretization =
      static_cast<Albany::STKDiscretization&>(*discretization_ptr);

  // Need solution for output call
  Teuchos::RCP<Tpetra_Vector> solution_fieldT =
      stk_discretization.getSolutionFieldT();

  // second arg to output is (pseudo)time
  stk_discretization.writeSolutionT(*solution_fieldT, 1.0);

  return 0;
}
