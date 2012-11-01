//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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

#include <Partition.h>

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
      "Partitioning of Exodus mesh with Zoltan.\n"
      "Uses geometric, hypergraph or K-means partitioning algorithms.\n");

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

  const int
  number_schemes = 3;

  const LCM::PartitionScheme
  scheme_values[] = {LCM::GEOMETRIC, LCM::HYPERGRAPH, LCM::KMEANS};

  const char*
  scheme_names[] = {"geometric", "hypergraph", "kmeans"};

  LCM::PartitionScheme
  partition_scheme = LCM::GEOMETRIC;

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

  int
  cluster_size = 64;

  command_line_processor.setOption(
      "cluster-size",
      &cluster_size,
      "Cluster Size");

  int
  maximum_divisions = 64;

  command_line_processor.setOption(
      "maximum-divisions",
      &maximum_divisions,
      "Maximum Divisions");

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
  // Set extra parameters for K-means
  //
  connectivity_array.SetClusterSize(cluster_size);
  connectivity_array.SetMaximumDivisions(maximum_divisions);

  //
  // Partition mesh
  //
  const std::map<int, int>
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

  Albany::StateArrays
  state_arrays = stk_discretization.getStateArrays();

  for (Albany::StateArrays::size_type i = 0; i < state_arrays.size(); ++i) {

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

      const int
      partition = (*partitions_iterator).second;

      element++;
      stk_partition[j] = partition;
    }

  }

  // Need solution for output call
  Teuchos::RCP<Epetra_Vector>
  solution_field = stk_discretization.getSolutionField();

  // second arg to output is (pseudo)time
  stk_discretization.outputToExodus(*solution_field, 1.0);

  // Write report
  const double
  volume = connectivity_array.GetVolume();

  const double
  length_scale_cubed = length_scale * length_scale * length_scale;

  const LCM::ScalarMap
  partition_volumes = connectivity_array.GetPartitionVolumes();

  const unsigned int
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

    const int
    element = (*partitions_iter).first;

    const int
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
