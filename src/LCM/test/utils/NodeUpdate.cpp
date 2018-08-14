//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
// Test of mesh manipulation.
// Separate all elements of a mesh by nodal replacement
//

#include "topology/Topology.h"
#include "topology/Topology_Utils.h"

int
main(int ac, char* av[])
{
  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString(
      "Test of element separation through nodal insertion.\n"
      "Remove and replace all nodes in elements.\n");

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

  LCM::Topology topology(input_file, output_file);

  stk::mesh::BulkData& bulkData = topology.get_bulk_data();

  // Node rank should be 0 and element rank should be equal to the dimension of
  // the system (e.g. 2 for 2D meshes and 3 for 3D meshes)
  // std::cout << "Node Rank: "<< nodeRank << ", Element Rank: " <<
  // getCellRank() << "\n";

  // Print element connectivity before the mesh topology is modified
  std::cout << "*************************\n"
            << "Before element separation\n"
            << "*************************\n";

  // Start the mesh update process
  // Will fully separate the elements in the mesh by replacing element nodes
  // Get a vector containing the element set of the mesh.
  std::vector<stk::mesh::Entity> element_lst;
  stk::mesh::get_entities(bulkData, stk::topology::ELEMENT_RANK, element_lst);

  // Modifies mesh for graph algorithm
  // Function must be called each time before there are changes to the mesh
  topology.removeNodeRelations();

  // Check for failure criterion
  topology.setEntitiesOpen();
  std::string gviz_output = LCM::parallelize_string("output") + ".dot";
  topology.outputToGraphviz(gviz_output);

  // test the functions of the class
  bulkData.modification_begin();

  // begin mesh fracture
  std::cout << "begin mesh fracture\n";
  topology.splitOpenFaces();

  // std::string gviz_output = "output.dot";
  // topology.output_to_graphviz(gviz_output,entity_open);

  // Recreates connectivity in stk mesh expected by Albany_STKDiscretization
  // Must be called each time at conclusion of mesh modification
  topology.restoreElementToNodeConnectivity();

  // End mesh update
  bulkData.modification_end();

  std::cout << "*************************\n"
            << "After element separation\n"
            << "*************************\n";

  // Need to update the mesh to reflect changes in duplicate_entity routine.
  //   Redefine connectivity and coordinate arrays with updated values.
  //   Mesh must only have relations between elements and nodes.
  Teuchos::RCP<Albany::AbstractDiscretization> discretization_ptr =
      topology.get_discretization();
  Albany::STKDiscretization& stk_discretization =
      static_cast<Albany::STKDiscretization&>(*discretization_ptr);

  Teuchos::ArrayRCP<double> coordinates = stk_discretization.getCoordinates();

  // Separate the elements of the mesh to illustrate the
  //   disconnected nature of the final mesh

  // Create a vector to hold displacement values for nodes
  Teuchos::RCP<const Tpetra_Map> dof_mapT = stk_discretization.getMapT();
  Teuchos::RCP<Tpetra_Vector>    displacementT =
      Teuchos::rcp(new Tpetra_Vector(dof_mapT));
  Teuchos::ArrayRCP<ST> displacementT_nonconstView =
      displacementT->get1dViewNonConst();

  // Add displacement to nodes
  stk::mesh::get_entities(bulkData, stk::topology::ELEMENT_RANK, element_lst);

  // displacement scale factor
  double alpha = 0.5;

  for (int i = 0; i < element_lst.size(); ++i) {
    std::vector<double>      centroid(3);
    std::vector<double>      disp(3);
    stk::mesh::Entity const* relations = bulkData.begin_nodes(element_lst[i]);
    int const                num_relations = bulkData.num_nodes(element_lst[i]);
    // Get centroid of the element
    for (int j = 0; j < num_relations; ++j) {
      stk::mesh::Entity node = relations[j];
      int               id   = static_cast<int>(bulkData.identifier(node));
      centroid[0] += coordinates[id * 3 - 3];
      centroid[1] += coordinates[id * 3 - 2];
      centroid[2] += coordinates[id * 3 - 1];
    }
    centroid[0] /= num_relations;
    centroid[1] /= num_relations;
    centroid[2] /= num_relations;

    // Determine displacement
    for (int j = 0; j < 3; ++j) { disp[j] = alpha * centroid[j]; }

    // Add displacement to nodes
    for (int j = 0; j < num_relations; ++j) {
      stk::mesh::Entity node = relations[j];
      int               id   = static_cast<int>(bulkData.identifier(node));
      displacementT_nonconstView[id * 3 - 3] += disp[0];
      displacementT_nonconstView[id * 3 - 2] += disp[1];
      displacementT_nonconstView[id * 3 - 1] += disp[2];
    }
  }

  stk_discretization.setResidualFieldT(*displacementT);

  Teuchos::RCP<Tpetra_Vector> solution_fieldT =
      stk_discretization.getSolutionFieldT();

  // Write final mesh to exodus file
  // second arg to output is (pseudo)time
  //  stk_discretization.outputToExodus(*solution_field, 1.0);
  stk_discretization.writeSolutionT(*solution_fieldT, 1.0);

  return 0;
}
