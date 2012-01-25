//
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
      "Uses geometric or hypergraph partitioning algorithms.\n");

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
  number_schemes = 2;

  const LCM::PartitionScheme
  scheme_values[] = {LCM::GEOMETRIC, LCM::HYPERGRAPH};

  const char*
  scheme_names[] = {"geometric", "hypergraph"};

  LCM::PartitionScheme
  partition_scheme = LCM::HYPERGRAPH;

  command_line_processor.setOption(
      "scheme",
      &partition_scheme,
      number_schemes,
      scheme_values,
      scheme_names,
      "Partition Scheme");

  double
  length_scale = 0.0016;

  command_line_processor.setOption(
      "length-scale",
      &length_scale,
      "Length Scale");


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

  // Get MDArray which is memeopru in stk for "Partition" element variable
  Albany::MDArray
  stk_partition = stk_discretization.getStateArrays()[0]["Partition"];

  //
  // Output partitions
  //

  // For better color contrast in visualization programs, shuffle
  // the partition number so that it is less likely that partitions
  // with very close numbers are next to each other, leading to almost
  // the same color in output.
  const LCM::ScalarMap
  partition_volumes = connectivity_array.GetPartitionVolumes();

  const unsigned int
  number_partitions = partition_volumes.size();

  LCM::IDList
  partition_shuffle(number_partitions);

  for (LCM::IDList::size_type i = 0; i < number_partitions; ++i) {
    partition_shuffle[i] = i;
  }

  std::random_shuffle(partition_shuffle.begin(), partition_shuffle.end());

  // Convert partition map to format used by Albany to set internal variables.
  // Assumption: numbering of elements is contiguous.
  for (std::map<int, int>::const_iterator
      partitions_iter = partitions.begin();
      partitions_iter != partitions.end();
      ++partitions_iter) {

    const int
    element = (*partitions_iter).first;

    const int
    partition = (*partitions_iter).second;

    // set partition number in stk field memory
    stk_partition[element] = partition_shuffle[partition];

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

  LCM::DualGraph dual_graph(connectivity_array);
  dual_graph.Print();

  return 0;

}

#else // #if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)

// Zoltan not defined, do nothing
int main(int ac, char* av[])
{
  return 0;
}

#endif // #if defined (ALBANY_ZOLTAN)
