//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
// Simple mesh partitioning program
//

// Define only if Zoltan is enabled
#if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)

#include <algorithm>
#include <iomanip>
#include <Teuchos_CommandLineProcessor.hpp>

#include <LCMPartition.h>

bool TpetraBuild = false;

int main(int ac, char* av[])
{
  //
  // Initialize Zoltan
  //
  float
  version;

  Zoltan_Initialize(ac, av, &version);

  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor
  command_line_processor;

  command_line_processor.setDocString(
      "Partitioning of Exodus mesh for nonlocal regularization.\n"
      "Uses random, geometric, hypergraph or K-means variants "
      "partitioning algorithms.\n");

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

  int const
  number_schemes = 5;

  LCM::PARTITION::Scheme const
  scheme_values[] = {
      LCM::PARTITION::GEOMETRIC,
      LCM::PARTITION::HYPERGRAPH,
      LCM::PARTITION::KMEANS,
      LCM::PARTITION::SEQUENTIAL,
      LCM::PARTITION::KDTREE};

  char const *
  scheme_names[] = {
      "geometric",
      "hypergraph",
      "kmeans",
      "sequential",
      "kdtree"};

  LCM::PARTITION::Scheme
  partition_scheme = LCM::PARTITION::KDTREE;

  command_line_processor.setOption(
      "scheme",
      &partition_scheme,
      number_schemes,
      scheme_values,
      scheme_names,
      "Partition Scheme");

  double
  length_scale = 0.0017;

  command_line_processor.setOption(
      "length-scale",
      &length_scale,
      "Length Scale");

  double
  tolerance = 1.0e-4;

  command_line_processor.setOption(
      "tolerance",
      &tolerance,
      "Tolerance");

  double
  requested_cell_size = 0.0001;

  command_line_processor.setOption(
      "requested-cell-size",
      &requested_cell_size,
      "Requested cell size");

  int
  maximum_iterations = 64;

  command_line_processor.setOption(
      "maximum-iterations",
      &maximum_iterations,
      "Maximum Iterations");

  int const
  number_initializers = 3;

  LCM::PARTITION::Scheme const
  initializer_values[] = {
      LCM::PARTITION::RANDOM,
      LCM::PARTITION::GEOMETRIC,
      LCM::PARTITION::HYPERGRAPH};

  char const *
  initializer_names[] = {
      "random",
      "geometric",
      "hypergraph"};

  LCM::PARTITION::Scheme
  initializer_scheme = LCM::PARTITION::GEOMETRIC;

  command_line_processor.setOption(
      "initializer",
      &initializer_scheme,
      number_initializers,
      initializer_values,
      initializer_names,
      "Initializer Scheme");

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
  // Read mesh
  //
  LCM::ConnectivityArray
  connectivity_array(input_file, output_file);

  //
  // Set extra parameters
  //
  connectivity_array.SetTolerance(tolerance);
  connectivity_array.SetCellSize(requested_cell_size);
  connectivity_array.SetMaximumIterations(maximum_iterations);
  connectivity_array.SetInitializerScheme(initializer_scheme);

  //
  // Partition mesh
  //
  std::map<int, int> const
  partitions = connectivity_array.Partition(partition_scheme, length_scale);

  // Get abstract discretization from connectivity array and convert
  // to stk discretization to use stk-specific methods.
  Albany::AbstractDiscretization &
  discretization = connectivity_array.GetDiscretization();

  Albany::STKDiscretization &
  stk_discretization = static_cast<Albany::STKDiscretization &>(discretization);

  //
  // Output partitions
  //

  // Convert partition map to format used by Albany to set internal variables.
  // Assumption: numbering of elements is contiguous.

  int
  element = 0;

  Albany::StateArrayVec
  state_arrays = stk_discretization.getStateArrays().elemStateArrays;

  for (Albany::StateArrayVec::size_type i = 0; i < state_arrays.size(); ++i) {

    Albany::StateArray
    state_array = state_arrays[i];

    // Get MDArray which has the "Partition" element variable
    Albany::MDArray
    stk_partition = state_array["Partition"];

    for (Albany::MDArray::size_type j = 0; j < stk_partition.size(); ++j) {

      const std::map<int, int>::const_iterator
      partitions_iterator = partitions.find(element);

      if (partitions_iterator == partitions.end()) {
        std::cerr << std::endl;
        std::cerr << "Element " << element << " does not have a partition.";
        std::cerr << std::endl;
        exit(1);
      }

      int const
      partition = (*partitions_iterator).second;

      element++;
      stk_partition[j] = partition;
    }

  }

  // Need solution for output call
  Teuchos::RCP<Tpetra_Vector>
  solution_fieldT = stk_discretization.getSolutionFieldT();

  // second arg to output is (pseudo)time
  stk_discretization.writeSolutionT(*solution_fieldT, 1.0);

  // Write report
  double const
  volume = connectivity_array.GetVolume();

  double const
  length_scale_cubed = length_scale * length_scale * length_scale;

  LCM::ScalarMap const
  partition_volumes = connectivity_array.GetPartitionVolumes();

  unsigned int const
  number_partitions = partition_volumes.size();

  std::cout << std::endl;
  std::cout << "==========================================";
  std::cout << std::endl;
  std::cout << "Total Mesh Volume (V)    : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << volume << std::endl;
  std::cout << "Length Scale             : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << length_scale << std::endl;
  std::cout << "Length Scale Cubed (L^3) : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << length_scale_cubed << std::endl;
  std::cout << "V/L^3                    : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << volume / length_scale_cubed << std::endl;
  std::cout << "Number of Partitions     : " << number_partitions;
  std::cout << std::endl;
  std::cout << "------------------------------------------";
  std::cout << std::endl;
  std::cout << "Partition      Volume (Vi)          Vi/L^3";
  std::cout << std::endl;
  std::cout << "------------------------------------------";
  std::cout << std::endl;
  for (LCM::ScalarMap::const_iterator iter = partition_volumes.begin();
      iter != partition_volumes.end();
      ++iter) {
    int partition = (*iter).first;
    double volume = (*iter).second;
    std::cout << std::setw(10) << partition;
    std::cout << std::scientific << std::setw(16) << std::setprecision(8);
    std::cout << volume;
    std::cout << std::scientific << std::setw(16) << std::setprecision(8);
    std::cout << volume / length_scale_cubed << std::endl;
  }
  std::cout << "==========================================";
  std::cout << std::endl;

#if 0

  std::cout << "Number of elements       : ";
  std::cout << std::setw(14);
  std::cout << connectivity_array.GetNumberElements() << std::endl;
  std::cout << std::endl;
  std::cout << "------------------------------------------";
  std::cout << std::endl;
  std::cout << "Element        Partition";
  std::cout << std::endl;
  std::cout << "------------------------------------------";
  std::cout << std::endl;
  for (std::map<int, int>::const_iterator
      partitions_iter = partitions.begin();
      partitions_iter != partitions.end();
      ++partitions_iter) {

    int const
    element = (*partitions_iter).first;

    int const
    partition = (*partitions_iter).second;

    std::cout << std::setw(16) << element;
    std::cout << std::setw(16) << partition;
    std::cout << std::endl;
  }
  std::cout << "==========================================";
  std::cout << std::endl;


  LCM::DualGraph dual_graph(connectivity_array);
  dual_graph.Print();

#endif

  return 0;

}

#else // #if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)

// Zoltan not defined, do nothing
int main(int ac, char* av[])
{
  return 0;
}

#endif // #if defined (ALBANY_ZOLTAN)
