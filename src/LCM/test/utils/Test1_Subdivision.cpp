//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
// Test 1 of barycentric subdivision. Input file has to be specified.
// Checks the proper operation of the barycentric_subdivision() function.
// Restricted to simplicial complexes.
//

#include "time.h"

#include "topology/Topology.h"
#include "topology/Topology_Utils.h"

typedef stk::mesh::Entity Entity;

/*
 * Returns a vector with the number of entities of the current
 * mesh. e.g. vector's element 0 returns the number of entities of rank 0.
 * The vector's element 1 returns the number of entities of rank 1, and so on.
 */
std::vector<int>
return_number_entities(LCM::Topology& topology_);

//
// Checks if the subdivision was done correctly
//
std::string
verify_subdivision(
    const std::vector<int>& former_num_entities,
    const std::vector<int>& final_num_entities);

int
main(int ac, char* av[])
{
  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString(
      "Test of barycentric subdivision.\n"
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
  LCM::Topology             topology(input_file, output_file);
  // Print element connectivity before the mesh topology is modified
  std::cout << "***********************" << std::endl;
  std::cout << "Before mesh subdivision" << std::endl;
  std::cout << "***********************" << std::endl;
  LCM::display_connectivity(topology, stk::topology::ELEMENT_RANK);

  // Request the number of entities of the input mesh
  std::vector<int> vector_initial_entities = return_number_entities(topology);

  // Start the mesh update process
  // Prepares mesh for barycentric subdivision
  topology.removeNodeRelations();

  //
  // Here starts the barycentric subdivision.
  //
  // Carry out barycentric subdivision on the mesh
  topology.barycentricSubdivision();
  std::cout << "*************************" << std::endl;
  std::cout << "After element subdivision" << std::endl;
  std::cout << "*************************" << std::endl;
  // Request the number of entities of the output mesh after subdivision
  std::vector<int> vector_final_entities = return_number_entities(topology);
  LCM::display_connectivity(topology, stk::topology::ELEMENT_RANK);

  // Checking that the final mesh after subdivision is correct
  std::cout << "*************************************" << std::endl;
  std::cout << "Checking final mesh after subdivision" << std::endl;
  std::cout << "*************************************" << std::endl;
  std::cout << verify_subdivision(
                   vector_initial_entities, vector_final_entities)
            << std::endl;

  return 0;
}

/*
 * Returns a vector with the number of entities of the current
 * mesh. e.g. vector's element 0 returns the number of entities of rank 0.
 * The vector's element 1 returns the number of entities of rank 1, and so on.
 */
std::vector<int>
return_number_entities(LCM::Topology& topology_)
{
  // Vector with output info
  std::vector<int> output_vector;
  // Push back number of nodes
  stk::mesh::BulkData&           bulkData_ = topology_.get_bulk_data();
  std::vector<stk::mesh::Entity> initial_entities_D0 =
      topology_.getEntitiesByRank(bulkData_, stk::topology::NODE_RANK);
  output_vector.push_back(initial_entities_D0.size());
  // Push back number of edges
  std::vector<stk::mesh::Entity> initial_entities_D1 =
      topology_.getEntitiesByRank(bulkData_, stk::topology::EDGE_RANK);
  output_vector.push_back(initial_entities_D1.size());
  // Push back number of faces
  std::vector<stk::mesh::Entity> initial_entities_D2 =
      topology_.getEntitiesByRank(bulkData_, stk::topology::FACE_RANK);
  output_vector.push_back(initial_entities_D2.size());
  // Push back number of elements
  std::vector<stk::mesh::Entity> initial_entities_D3 =
      topology_.getEntitiesByRank(bulkData_, stk::topology::ELEMENT_RANK);
  output_vector.push_back(initial_entities_D3.size());

  return output_vector;
}
//
// Checks if the subdivision was done correctly
//
std::string
verify_subdivision(
    const std::vector<int>& former_num_entities,
    const std::vector<int>& final_num_entities)
{
  // Verify the number of nodes
  int final_number_nodes = former_num_entities[0] + former_num_entities[1] +
                           former_num_entities[2] + former_num_entities[3];
  TEUCHOS_TEST_FOR_EXCEPTION(
      final_number_nodes != final_num_entities[0],
      std::logic_error,
      "The number of nodes after subdivision is incorrect\n");
  // Verify the number of edges
  int final_number_edges = (former_num_entities[1] * 2) +
                           (former_num_entities[2] * 6) +
                           (14 * former_num_entities[3]);
  TEUCHOS_TEST_FOR_EXCEPTION(
      final_number_edges != final_num_entities[1],
      std::logic_error,
      "The number of edges after subdivision is incorrect\n");
  // Verify the number of faces
  int final_number_faces =
      (former_num_entities[2] * 6) + (36 * former_num_entities[3]);
  TEUCHOS_TEST_FOR_EXCEPTION(
      final_number_faces != final_num_entities[2],
      std::logic_error,
      "The number of faces after subdivision is incorrect\n");
  // Verify the number of elements
  int final_number_elements = 24 * former_num_entities[3];
  TEUCHOS_TEST_FOR_EXCEPTION(
      final_number_elements != final_num_entities[3],
      std::logic_error,
      "The number of elements after subdivision is incorrect\n");
  // If all the subdivision is done correctly, the following message will be
  // displayed
  return std::string("SUBDIVISION TEST 1: PASSED");
}
