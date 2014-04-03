//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
// Test of mesh manipulation.
// Separate all elements of a mesh by nodal replacement
//

#include "topology/Topology.h"
#include "topology/Topology_Utils.h"

int main(int ac, char* av[])
{

  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor
    command_line_processor;

  command_line_processor.setDocString(
                                      "Test of element separation through nodal insertion.\n"
                                      "Remove and replace all nodes in elements.\n");

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
  // Copied from Partition.cc
  Teuchos::GlobalMPISession mpiSession(&ac,&av);

  LCM::Topology
    topology(input_file,output_file);

  stk::mesh::BulkData&
    bulkData = *(topology.getBulkData());

  // Node rank should be 0 and element rank should be equal to the dimension of the
  // system (e.g. 2 for 2D meshes and 3 for 3D meshes)
  //std::cout << "Node Rank: "<< nodeRank << ", Element Rank: " << getCellRank() << "\n";

  // Print element connectivity before the mesh topology is modified
  std::cout << "*************************\n"
       << "Before element separation\n"
       << "*************************\n";

  // Start the mesh update process
  // Will fully separate the elements in the mesh by replacing element nodes
  // Get a vector containing the element set of the mesh.
  std::vector<stk::mesh::Entity*> element_lst;
  stk::mesh::get_entities(bulkData,topology.getCellRank(),element_lst);

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

  //std::string gviz_output = "output.dot";
  //topology.output_to_graphviz(gviz_output,entity_open);

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
    topology.getDiscretization();
  Albany::STKDiscretization & stk_discretization =
    static_cast<Albany::STKDiscretization &>(*discretization_ptr);

  Teuchos::ArrayRCP<double>
    coordinates = stk_discretization.getCoordinates();

  // Separate the elements of the mesh to illustrate the
  //   disconnected nature of the final mesh

  // Create a vector to hold displacement values for nodes
  Teuchos::RCP<const Epetra_Map> dof_map = stk_discretization.getMap();
  Epetra_Vector displacement = Epetra_Vector(*(dof_map),true);

  // Add displacement to nodes
  stk::mesh::get_entities(bulkData,topology.getCellRank(),element_lst);

  // displacement scale factor
  double alpha = 0.5;

  for (int i = 0; i < element_lst.size(); ++i){
    std::vector<double> centroid(3);
    std::vector<double> disp(3);
    stk::mesh::PairIterRelation relations =
      element_lst[i]->relations(LCM::NODE_RANK);
    // Get centroid of the element
    for (int j = 0; j < relations.size(); ++j){
      stk::mesh::Entity & node = *(relations[j].entity());
      int id = static_cast<int>(node.identifier());
      centroid[0] += coordinates[id*3-3];
      centroid[1] += coordinates[id*3-2];
      centroid[2] += coordinates[id*3-1];
    }
    centroid[0] /= relations.size();
    centroid[1] /= relations.size();
    centroid[2] /= relations.size();

    // Determine displacement
    for (int j = 0; j < 3; ++j){
      disp[j] = alpha*centroid[j];
    }

    // Add displacement to nodes
    for (int j = 0; j < relations.size(); ++j){
      stk::mesh::Entity & node = *(relations[j].entity());
      int id = static_cast<int>(node.identifier());
      displacement[id*3-3] += disp[0];
      displacement[id*3-2] += disp[1];
      displacement[id*3-1] += disp[2];
    }
  }

  stk_discretization.setResidualField(displacement);

  Teuchos::RCP<Epetra_Vector>
    solution_field = stk_discretization.getSolutionField();

  // Write final mesh to exodus file
  // second arg to output is (pseudo)time
//  stk_discretization.outputToExodus(*solution_field, 1.0);
  stk_discretization.writeSolution(*solution_field, 1.0);

  return 0;

}
