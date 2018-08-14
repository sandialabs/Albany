//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//#include "Topology.h"
#include "topology/Topology.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
typedef stk::mesh::Entity Entity;

int
main(int ac, char* av[])
{
  std::string               input_file;
  std::string               results_file;
  typedef stk::mesh::Entity Entity;
  std::string               output_file;  // = "output.e";

  using namespace std;

  if (ac == 3) {
    std::cout << "Generating the msh file with mesh " << av[1]
              << " and from text file: " << av[2] << std::endl;
    stringstream ss1;
    stringstream ss2;
    ss1 << av[1];       // insert the char
    ss1 >> input_file;  // extract -convert char into string
    ss2 << av[2];
    ss2 >> results_file;

    // Check if third parameter from command line is a .txt file
    string s2 = ".txt";
    if (results_file.find(s2) == std::string::npos) {
      std::cout << "Missing extension .txt for the third input" << '\n';
      throw(0);
    }
  }

  else {
    std::cout << "Usage: MinsurfaceOutput Mesh.exo ResultsFile.txt"
              << std::endl;  // USAGE OF THIS FILE FROM THE COMMAND LINE
    return 1;
  }

  //
  // Read the mesh
  //
  // Copied from Partition.cc
  Teuchos::GlobalMPISession mpiSession(&ac, &av);
  LCM::Topology             topology(input_file, output_file);

  //------------------------------------------------------------------------------------------------------------------------------------
  // Obtain the results from the solver
  // BASED ON THE OUTPUT USING CLP LINEAR SOLVER (./clp mps_file -dualsimplex
  // -solu outputFileName)

  using namespace std;

  // Obtain the number of columns of the boundary operator
  std::vector<std::vector<int>> facesDirec = topology.facesDirections();
  cout << "The number of columns of the boundary operator is: "
       << facesDirec.size() * 2 << endl;

  // Read input from text file
  std::ifstream solSolver(
      results_file.c_str());  // Read the file that comes from the solver
  std::string         line, dummyLine1, dummyLine2;
  std::vector<string> vectorStrings;

  // Read the first and second line from text file.
  std::getline(solSolver, dummyLine1);
  std::getline(solSolver, dummyLine2);

  // Define the vector that holds the results from linear solver
  std::vector<int> resutsFromSolver;
  int              i;

  // Read from third line
  while (std::getline(solSolver, line)) {
    std::string        col1, col2;
    std::istringstream ss(line);
    ss >> col1 >> col2;
    std::string Finalstring = col2.substr(
        1, col2.length());          // Get the appropriate string from text file
    i = atoi(Finalstring.c_str());  // Convert string to integer
    resutsFromSolver.push_back(
        i);  // Push back the numbers from text file to resutsFromSolver vector
  }

  // Record ones (1s) at the corresponding positions specified by the solver
  std::vector<int> resultsVector(
      facesDirec.size() * 2, 0);  // Initialize vector with zeros

  for (unsigned int i = 0; i < resutsFromSolver.size(); i++) {
    resultsVector[resutsFromSolver[i] - 1] = 1;
  }

  // Call the function "MinimumSurfaceFaces" that returns the entities
  // associated with the solution given by the solver
  std::vector<stk::mesh::Entity> EntitiesMinSurface =
      topology.minimumSurfaceFaces(resultsVector);

  // Debug: return the number of repeated entities if there are any
  std::vector<stk::mesh::Entity>                 NumberRepeatedEntities;
  std::vector<stk::mesh::Entity>::const_iterator I_Entities;
  for (I_Entities = EntitiesMinSurface.begin();
       I_Entities != EntitiesMinSurface.end();
       I_Entities++) {
    if (topology.numberOfRepetitions(EntitiesMinSurface, *I_Entities) != 1) {
      cout << "Warning repeated entities" << *I_Entities << endl;
    }
  }

  // Check that the 1s in the solver file match the required number of entities
  cout << "The number of faces that belong to the minimum surface are: "
       << EntitiesMinSurface.size() << endl;

  // Create a matrix that contains the boundary nodes of each Entity
  // This is also the connectivity matrix
  std::vector<std::vector<int>>                  boundaryNodes_;
  std::vector<stk::mesh::Entity>::const_iterator I_entities;
  for (I_entities = EntitiesMinSurface.begin();
       I_entities != EntitiesMinSurface.end();
       I_entities++) {
    std::vector<stk::mesh::Entity> temp;
    // Obtain all the entities of rank 0
    temp = topology.getBoundaryEntities(*I_entities, stk::topology::NODE_RANK);
    std::vector<int>                               temp2;
    std::vector<stk::mesh::Entity>::const_iterator I_nodes;
    // Get the identifiers of the entities above
    for (I_nodes = temp.begin(); I_nodes != temp.end(); I_nodes++) {
      temp2.push_back(topology.get_bulk_data().identifier(*I_nodes));
    }
    boundaryNodes_.push_back(temp2);  // Connectivity matrix
  }

  // Compute the total area of the faces that conform the minimum surface
  // Initialize MinSurfaceAreas as a vector of zeros
  std::vector<double> MinSurfaceAreas;
  for (unsigned int i = 0; i < (EntitiesMinSurface.size()); i++) {
    MinSurfaceAreas.push_back(0);
  }

  // Iterate through the vector of Entities that build the minimum surface
  for (unsigned int i = 0; i < (EntitiesMinSurface.size()); i++) {
    // Compute the area
    std::vector<stk::mesh::Entity> Nodes = topology.getBoundaryEntities(
        EntitiesMinSurface[i], stk::topology::NODE_RANK);
    double a    = topology.getDistanceBetweenNodes(Nodes[0], Nodes[1]);
    double b    = topology.getDistanceBetweenNodes(Nodes[1], Nodes[2]);
    double c    = topology.getDistanceBetweenNodes(Nodes[2], Nodes[0]);
    double p    = (a + b + c) / 2;
    double Area = sqrt(p * (p - a) * (p - b) * (p - c));

    // Put the area into the array the right index
    MinSurfaceAreas[i] = Area;
  }

  // Add up all the areas of the faces that represent the minimum surface
  std::vector<double>::const_iterator I_areas;
  double                              MinSurfaceArea = 0;
  for (I_areas = MinSurfaceAreas.begin(); I_areas != MinSurfaceAreas.end();
       I_areas++) {
    MinSurfaceArea += *I_areas;
  }

  cout << endl;
  cout << "The area of the minimum surface computed is :" << MinSurfaceArea
       << endl;

  // Create a set based on boundaryNodes. This is to avoid having
  // repeated node identifiers
  std::set<int>                                 boundaryNodes;
  std::vector<std::vector<int>>::const_iterator I_rows;
  std::vector<int>::const_iterator              I_cols;
  for (I_rows = boundaryNodes_.begin(); I_rows != boundaryNodes_.end();
       I_rows++) {
    for (I_cols = I_rows->begin(); I_cols != I_rows->end(); I_cols++) {
      boundaryNodes.insert(
          *I_cols);  // All nodes organized in ascending order. Starting from 1
    }
  }

  // Create the msh output file
  // Extract the input file name
  string txtFile =
      ".txt";  // Define the string with the extension file of input mesh
  std::size_t Position =
      results_file.find(txtFile);  // Find the position where .txt starts
  std::string file_name_1 = results_file.substr(
      0, Position);  // sets part of the name of the output file
  std::string file_name_2 = "MinSurface_" + file_name_1 + ".msh";

  ofstream outputToVtk(file_name_2.c_str(), std::ofstream::out);
  if (outputToVtk.is_open()) {
    int dimension             = 3;
    int NumberNodesPerElement = 3;

    outputToVtk << dimension << " " << NumberNodesPerElement
                << endl;  // Dimension  NodesPerElement
    outputToVtk << boundaryNodes.size() << " " << boundaryNodes_.size()
                << endl;  // Nnodes  Nelements

    // Create a map that assigns new numbering to the nodes
    std::map<int, int>            node_map;
    int                           counter = 0;
    std::set<int>::const_iterator I_setOfNodes;

    // Output a matrix with the coordinates of the nodes
    for (I_setOfNodes = boundaryNodes.begin();
         I_setOfNodes != boundaryNodes.end();
         ++I_setOfNodes) {
      std::vector<double> nodeCoordinates =
          topology.findCoordinates(*I_setOfNodes);
      outputToVtk << nodeCoordinates[0] << " " << nodeCoordinates[1] << " "
                  << nodeCoordinates[2] << endl;  // Coordinates list OUTPUT
      node_map[*I_setOfNodes] = counter;
      counter++;
    }

    // Create the connectivity matrix
    for (unsigned int i = 0; i < boundaryNodes_.size(); ++i) {
      for (unsigned int j = 0; j < boundaryNodes_[i].size(); ++j) {
        outputToVtk << (node_map.find(boundaryNodes_[i][j])->second) + 1
                    << " ";  // List of edges     OUTPUT
      }
      outputToVtk << endl;
    }
  }  // end if
  else
    cout << "Unable to open file";

  return 0;
}
