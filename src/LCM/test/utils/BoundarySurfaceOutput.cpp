//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "topology/Topology.h"

typedef stk::mesh::Entity Entity;

// \brief Function that helps creating text files with different names
std::string
itoa(int num);

int
main(int ac, char* av[])
{
  std::string               input_file;
  std::string               normals_file;
  typedef stk::mesh::Entity Entity;
  std::string               output_file;  // = "output.e";

  using namespace std;

  if (ac == 3) {
    std::cout << "Generating .msh boundary files with mesh " << av[1]
              << " and normals from file: " << av[2] << std::endl;
    stringstream ss1;
    stringstream ss2;
    ss1 << av[1];       // insert the char
    ss1 >> input_file;  // extract -convert char into string
    ss2 << av[2];
    ss2 >> normals_file;

    // Check if third parameter from command line is a .txt file
    string s2 = ".txt";
    if (normals_file.find(s2) == std::string::npos) {
      std::cout << "Missing extension .txt for the third input" << '\n';
      throw(0);
    }
  } else {
    std::cout << "Usage:BoundarySurfaceOutput Mesh.exo Normals.txt"
              << std::endl;  // USAGE OF THIS FILE FROM THE COMMAND LINE
    return 1;
  }

  //--------------------------------------------------------------------------------------------------------
  // Read the mesh
  //--------------------------------------------------------------------------------------------------------
  // Copied from Partition.cc
  Teuchos::GlobalMPISession mpiSession(&ac, &av);
  LCM::Topology             topology(input_file, output_file);

  //--------------------------------------------------------------------------------------------------------
  // Extract the input file name
  //--------------------------------------------------------------------------------------------------------
  stringstream ss;
  std::string  file_name_;
  ss << av[1];       // insert the char
  ss >> file_name_;  // extract -convert char into string
  string exoFile =
      ".exo";  // Define the string with the extension file of input mesh
  std::size_t Position =
      file_name_.find(exoFile);  // Find the position where .exo starts
  std::string file_name2 = file_name_.substr(
      0, Position);  // sets part of the name of the output file

  //--------------------------------------------------------------------------------------------------------
  // Create the contour of the minimum surface
  //--------------------------------------------------------------------------------------------------------

  ifstream normals_(
      normals_file.c_str());  // Read the text file that contains all normals

  // numberNormals_ sets the number of rows of the matrix mNormals
  // This input text file must have the total number of normals defined in its
  // first line
  int numberNormals_;
  if (normals_.is_open()) {
    normals_ >> numberNormals_;  // Obtain the total number of input normals
  }

  // mNormal is a matrix containing all the normal vectors
  std::vector<std::vector<double>> mNormals(
      numberNormals_, std::vector<double>(3, 0));

  if (normals_.is_open()) {
    for (int i = 0; i < numberNormals_; i++) {
      normals_ >> mNormals[i][0] >> mNormals[i][1] >> mNormals[i][2];
    }
  }

  // Extract the numbers from the matrix that contains the normals
  std::vector<std::vector<int>> BoundaryVector;
  for (int i = 0; i < numberNormals_; i++) {
    std::string file_name =
        "Boundary_" + file_name2 + "_" + itoa(i + 1) +
        ".msh";  // file_name2 comes from the second input in the command line

    ofstream mshFile(file_name.c_str(), std::ofstream::out);
    if (mshFile.is_open()) {
      // Obtain the vector that defines the 1D boundary
      std::vector<std::vector<int>> BoundaryVector;

      std::vector<double> normalToPlane;
      normalToPlane.push_back(mNormals.at(i).at(0));
      normalToPlane.push_back(mNormals.at(i).at(1));
      normalToPlane.push_back(mNormals.at(i).at(2));

      // Finds coordinates of 3 vectors normal to the normal vector.
      std::vector<std::vector<double>> pointsOnPlane =
          topology.getCoordinatesOfTriangle(normalToPlane);

      // Finds the Node (stk::mesh::Entity of rank 0) that is closest to each of
      // the points on the plane
      std::vector<stk::mesh::Entity> closestNodes =
          topology.getClosestNodesOnSurface(pointsOnPlane);

      // Finds the identifiers of the nodes (entity rank 0) along the shortest
      // path connecting the three points
      std::vector<std::vector<int>> ShortestPathFinal =
          topology.shortestpath(closestNodes);

      // THE CODE WRITTEN BELOW IS TO CREATE THE .msh FILE UTILISED BY VTK
      // Create a vector with the identifiers of the nodes that build the
      // shortest path
      std::vector<int> nodesIdentifiers;
      for (unsigned int i = 0; i < ShortestPathFinal.size(); ++i) {
        for (unsigned int j = 0; j < ShortestPathFinal[i].size() - 1; ++j) {
          nodesIdentifiers.push_back((ShortestPathFinal[i][j]));
          nodesIdentifiers.push_back((ShortestPathFinal[i][j + 1]));
        }
      }

      // Create a vector without repeated numbers. Same vector as
      // nodesIdentifiers, but without repeated indices
      std::vector<int>                 setOfNodes;
      std::vector<int>::const_iterator iteratorInt;
      for (unsigned int i = 0; i < nodesIdentifiers.size(); i = i + 2) {
        setOfNodes.push_back(nodesIdentifiers[i]);
      }

      int dimension          = 3;
      int NumNodesPerElement = 2;

      mshFile << dimension << " " << NumNodesPerElement
              << endl;  // Dimension  nodesperElement
      mshFile << setOfNodes.size() << " " << ShortestPathFinal.size()
              << endl;  // Nnodes Nelements

      // Create a map that assigns new numbering to the nodes
      std::map<int, int>               node_map;
      int                              counter = 0;
      std::vector<int>::const_iterator I_setOfNodes;

      for (I_setOfNodes = setOfNodes.begin(); I_setOfNodes != setOfNodes.end();
           ++I_setOfNodes) {
        std::vector<double> nodeCoordinates =
            topology.findCoordinates(*I_setOfNodes);
        mshFile << nodeCoordinates[0] << " " << nodeCoordinates[1] << " "
                << nodeCoordinates[2] << endl;  // Coordinates list OUTPUT
        node_map[*I_setOfNodes] = counter;

        counter++;
      }

      // Create the connectivity matrix
      for (unsigned int i = 0; i < ShortestPathFinal.size(); ++i) {
        for (unsigned int j = 0; j < ShortestPathFinal[i].size(); ++j) {
          mshFile << (node_map.find(ShortestPathFinal[i][j])->second) + 1
                  << " ";  // List of edges
        }
        mshFile << endl;
      }

    }  // End of creating the .msh file, which contains the information of the
       // contour
  }  // End of for loop

  return 0;
}

// \brief Function that helps creating text files with different names
std::string
itoa(int num)
{
  std::stringstream converter;
  converter << num;
  return converter.str();
}
