//
// Simple mesh partitioning program
//
#include <iomanip>
#include <Teuchos_CommandLineProcessor.hpp>

#include <Partition.h>

int main(int ac, char* av[])
{
  // Initialize Zoltan
  float version;
  Zoltan_Initialize(ac, av, &version);

  // Create a command line processor
  Teuchos::CommandLineProcessor command_line_processor;

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

  double length_scale = 0.001;
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
    // Error
    return 1;
  }

  // Read mesh
  LCM::ConnectivityArray
  ca(input_file, output_file);

  // Partition mesh
  const std::map<int, int>
  partitions = ca.PartitionHyperGraph(length_scale);

  // Convert partition map to format used by Albany to set internal variables.
  // Assumption: numbering of elements is contiguous.
  const int
  number_elements = ca.GetNumberElements();

  std::vector< std::vector< double> >
  stk_partitions(number_elements);

  for (std::map<int, int>::const_iterator
      it = partitions.begin();
      it != partitions.end();
      ++it) {

    const int element = (*it).first;
    const int partition = (*it).second;

    stk_partitions[element].push_back(static_cast<double>(partition));

  }

  // Get abstract discretization from coonectivity array and convert
  // to stk discretization to use stk specific methods.
  Albany::AbstractDiscretization &
  discretization = ca.GetDiscretization();

  Albany::STKDiscretization &
  stk_discretization = static_cast<Albany::STKDiscretization &>(discretization);

  // Need solution for output call
  stk_discretization.outputToExodus(
      (*stk_discretization.getSolutionField()),
      stk_partitions);

#if defined(DEBUG)

  // Output partition map
  std::cout << "Partition map:" << std::endl;

  for (std::map<int, int>::const_iterator
      it = partitions.begin();
      it != partitions.end();
      ++it) {

    const int
    vertex = (*it).first;

    const int
    partition = (*it).second;

    std::cout << std::setw(12) << vertex;
    std::cout << std::setw(12) << partition << std::endl;

  }

  //Teuchos::GlobalMPISession mpiSession(&ac,&av);
  std::string filename(av[1]);

  LCM::ConnectivityArray ca(filename);

  std::cout << "Connectivity Array:" << std::endl;
  std::cout << ca;
  std::cout << std::endl;
  std::cout << std::endl;

  LCM::DualGraph dg(ca);
  LCM::ZoltanHyperGraph zhg(dg);

  std::cout << "Zoltan Hypergraph:" << std::endl;
  std::cout << zhg;
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Zoltan Vertex IDs:" << std::endl;

  const std::vector<ZOLTAN_ID_TYPE>
  vertices = zhg.GetVertexIDs();

  for (std::vector<ZOLTAN_ID_TYPE>::size_type
      i = 0;
      i < vertices.size();
      ++i ) {

    std::cout << " " << vertices[i];

  }

  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Zoltan Hyperedge Pointers:" << std::endl;

  const std::vector<int>
  pointers = zhg.GetEdgePointers();

  for (std::vector<int>::size_type
      i = 0;
      i < pointers.size();
      ++i ) {

    std::cout << " " << pointers[i];

  }

  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Zoltan Hyperedge IDs:" << std::endl;

  const std::vector<ZOLTAN_ID_TYPE>
  edges = zhg.GetEdgeIDs();

  for (std::vector<ZOLTAN_ID_TYPE>::size_type
      i = 0;
      i < edges.size();
      ++i ) {

    std::cout << " " << edges[i];

  }

  std::cout << std::endl;
  std::cout << std::endl;

#endif // #if defined(DEBUG)

  return 0;

}
