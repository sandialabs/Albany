//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//
// Simple mesh partitioning program
//
#include <algorithm>
#include <chrono>
#include <iomanip>

#include <LCMPartition.h>
#include <Teuchos_CommandLineProcessor.hpp>

int
main(int ac, char* av[])
{
  //
  // Initialize Zoltan
  //
  float version;

  Zoltan_Initialize(ac, av, &version);

  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString(
      "Partitioning of Exodus mesh for nonlocal regularization.\n"
      "Uses random, geometric, hypergraph or K-means variants "
      "partitioning algorithms.\n");

  std::string input_file = "input.e";

  command_line_processor.setOption("input", &input_file, "Input File Name");

  std::string output_file = "output.e";

  command_line_processor.setOption("output", &output_file, "Output File Name");

  int const number_schemes = 6;

  LCM::PARTITION::Scheme const scheme_values[] = {
      LCM::PARTITION::Scheme::RANDOM,
      LCM::PARTITION::Scheme::HYPERGRAPH,
      LCM::PARTITION::Scheme::GEOMETRIC,
      LCM::PARTITION::Scheme::KMEANS,
      LCM::PARTITION::Scheme::SEQUENTIAL,
      LCM::PARTITION::Scheme::KDTREE};

  char const* scheme_names[] = {
      "random", "hypergraph", "geometric", "kmeans", "sequential", "kdtree"};

  LCM::PARTITION::Scheme partition_scheme = LCM::PARTITION::Scheme::KDTREE;

  command_line_processor.setOption(
      "scheme",
      &partition_scheme,
      number_schemes,
      scheme_values,
      scheme_names,
      "Partition Scheme");

  double length_scale = 0.00165;

  command_line_processor.setOption(
      "length-scale", &length_scale, "Length Scale");

  double tolerance = 1.0e-4;

  command_line_processor.setOption("tolerance", &tolerance, "Tolerance");

  double requested_cell_size = 0.0001;

  command_line_processor.setOption(
      "requested-cell-size", &requested_cell_size, "Requested cell size");

  int maximum_iterations = 64;

  command_line_processor.setOption(
      "maximum-iterations", &maximum_iterations, "Maximum Iterations");

  int const number_initializers = 3;

  LCM::PARTITION::Scheme const initializer_values[] = {
      LCM::PARTITION::Scheme::RANDOM,
      LCM::PARTITION::Scheme::GEOMETRIC,
      LCM::PARTITION::Scheme::HYPERGRAPH};

  char const* initializer_names[] = {"random", "geometric", "hypergraph"};

  LCM::PARTITION::Scheme initializer_scheme = LCM::PARTITION::Scheme::GEOMETRIC;

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
  // Set extra parameters
  //
  connectivity_array.setTolerance(tolerance);
  connectivity_array.setCellSize(requested_cell_size);
  connectivity_array.setMaximumIterations(maximum_iterations);
  connectivity_array.setInitializerScheme(initializer_scheme);

  //
  // Partition mesh
  //
  auto start = std::chrono::system_clock::now();

  std::map<int, int> const partitions =
      connectivity_array.partition(partition_scheme, length_scale);

  auto end = std::chrono::system_clock::now();

  // Get abstract discretization from connectivity array and convert
  // to stk discretization to use stk-specific methods.
  Albany::AbstractDiscretization& discretization =
      connectivity_array.getDiscretization();

  Albany::STKDiscretization& stk_discretization =
      static_cast<Albany::STKDiscretization&>(discretization);

  //
  // Output partitions
  //

  // Convert partition map to format used by Albany to set internal variables.
  // Assumption: numbering of elements is contiguous.

  int element = 0;

  Albany::StateArrayVec state_arrays =
      stk_discretization.getStateArrays().elemStateArrays;

  for (Albany::StateArrayVec::size_type i = 0; i < state_arrays.size(); ++i) {
    Albany::StateArray state_array = state_arrays[i];

    // Get MDArray which has the "Partition" element variable
    Albany::MDArray stk_partition = state_array["Partition"];

    for (Albany::MDArray::size_type j = 0; j < stk_partition.size(); ++j) {
      const std::map<int, int>::const_iterator partitions_iterator =
          partitions.find(element);

      if (partitions_iterator == partitions.end()) {
        std::cerr << '\n';
        std::cerr << "Element " << element << " does not have a partition.";
        std::cerr << '\n';
        exit(1);
      }

      int const partition = (*partitions_iterator).second;

      element++;
      stk_partition[j] = partition;
    }
  }

  // Need solution for output call
  Teuchos::RCP<Tpetra_Vector> solution_fieldT =
      stk_discretization.getSolutionFieldT();

  // second arg to output is (pseudo)time
  stk_discretization.writeSolutionT(*solution_fieldT, 1.0);

  // Write report
  double const volume = connectivity_array.getVolume();

  double const length_scale_cubed = length_scale * length_scale * length_scale;

  LCM::ScalarMap const partition_volumes =
      connectivity_array.getPartitionVolumes();

  unsigned int const number_partitions = partition_volumes.size();

  std::cout << '\n';
  std::cout << "==========================================";
  std::cout << '\n';
  std::cout << "Total Mesh Volume (V)    : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << volume << '\n';
  std::cout << "Length Scale             : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << length_scale << '\n';
  std::cout << "Length Scale Cubed (L^3) : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << length_scale_cubed << '\n';
  std::cout << "V/L^3                    : ";
  std::cout << std::scientific << std::setw(14) << std::setprecision(8);
  std::cout << volume / length_scale_cubed << '\n';
  std::cout << "Number of Partitions     : " << number_partitions;
  std::cout << '\n';
  std::cout << "------------------------------------------";
  std::cout << '\n';
  std::cout << "Partition      Volume (Vi)          Vi/L^3";
  std::cout << '\n';
  std::cout << "------------------------------------------";
  std::cout << '\n';

  for (auto&& partition_volume : partition_volumes) {
    int const partition = partition_volume.first;

    double const volume = partition_volume.second;

    std::cout << std::setw(10) << partition;
    std::cout << std::scientific << std::setw(16) << std::setprecision(8);
    std::cout << volume;
    std::cout << std::scientific << std::setw(16) << std::setprecision(8);
    std::cout << volume / length_scale_cubed << '\n';
  }

#if defined(DEBUG)
  std::cout << "==========================================";
  std::cout << '\n';

  std::cout << "Number of elements       : ";
  std::cout << std::setw(14);
  std::cout << connectivity_array.GetNumberElements() << '\n';
  std::cout << '\n';
  std::cout << "------------------------------------------";
  std::cout << '\n';
  std::cout << "Element        Partition";
  std::cout << '\n';
  std::cout << "------------------------------------------";
  std::cout << '\n';

  for (auto&& partition_element : partitions) {
    int const element = partition_element.first;

    int const partition = partition_element.second;

    std::cout << std::setw(16) << element;
    std::cout << std::setw(16) << partition;
    std::cout << '\n';
  }
  std::cout << "==========================================";
  std::cout << '\n';

  LCM::DualGraph dual_graph(connectivity_array);
  dual_graph.Print();
#endif

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << std::scientific << std::setw(16) << std::setprecision(8);
  std::cout << "PARTITION TIME [s]: " << elapsed_seconds.count() << std::endl;

  return 0;
}
