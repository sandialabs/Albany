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

  if (ac == 2) {
    std::cout << "Generating MPS header file with mesh " << av[1] << std::endl;
    stringstream ss;
    ss << av[1];       // insert the char
    ss >> input_file;  // extract -convert char into string

  } else if (ac == 3) {
    std::cout << "Generating MPS rhs files with mesh " << av[1]
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
    std::cout
        << "Usage: MinsurfaceMPS Mesh.exo or MinsurfaceMPS Mesh.exo Normals.txt"
        << std::endl;  // USAGE OF THIS FILE FROM THE COMMAND LINE
    return 1;
  }

  //
  // Read the mesh
  //
  // Copied from Partition.cc
  Teuchos::GlobalMPISession mpiSession(&ac, &av);
  LCM::Topology             topology(input_file, output_file);

  //--------------------------------------------------------------------------------------------------------
  //                        mps file OUTPUT for the MinsurfaceSolver.cpp
  //--------------------------------------------------------------------------------------------------------
  // NOTE: When analyzing a single mesh, the coefficientsObjFunction, and the
  // BoundaryOperator remain the same. This is set in the first part of the MPS
  // file.The only one parameter that varies is the BoundaryVector, which is
  // defined in the second part of the MPS file
  //--------------------------------------------------------------------------------------------------------

  //--------------------------------------------------------------------------------------------------------
  // CREATE THE MPS FILE
  //--------------------------------------------------------------------------------------------------------
  stringstream ss;
  std::string  file_name_;
  ss << av[1];       // insert the char
  ss >> file_name_;  // extract -convert char into string
  string exoFile =
      ".exo";  // Define the string with the extension file of input mesh
  std::size_t Position =
      file_name_.find(exoFile);  // Find the position where .exo starts

  // First part of MPS file
  // Create the name of the text file that contains the header of the MPS file
  std::string file_name1 =
      "MPS_" + file_name_.substr(0, Position) + "_header.txt";
  std::string file_name2 =
      file_name_.substr(0, Position);  // name of the problem (important for the
                                       // LP solver that reads the mps file)

  // Compute the header of the mps file if 2 command line inputs are given
  if (ac == 2) {
    ofstream mpsFile(file_name1.c_str(), std::ofstream::out);
    if (mpsFile.is_open()) {
      mpsFile << "NAME"
              << " "
              << "MPS_" << file_name2 << "_"
              << endl;  // The linear problem name comes from the second
                        // argument in the command line
      mpsFile << "ROWS" << endl;
      mpsFile << " "
              << "N"
              << " "
              << "Z" << endl;  // Define the variable that holds the objective
                               // function. "N" means that it is unbounded

      // Define the rows and their format
      std::vector<std::vector<int>> edgesDirec = topology.edgesDirections();
      for (unsigned int i = 0; i < edgesDirec.size(); i++) {
        mpsFile << " "
                << "E"
                << " "
                << "R" << i + 1 << endl;
      }
      // Define the number of columns
      mpsFile << "COLUMNS" << endl;
      // Obtain the boundary operator and the areas of all faces
      std::vector<std::vector<double>> outputForMpsFile =
          topology.outputForMpsFile();
      for (unsigned int i = 0; i < outputForMpsFile.size(); i += 4) {
        // The columns of the boundary operator need to be saved in order X1,
        // X2, X3,...etc. Also their corresponding areas
        mpsFile << " "
                << "X" << outputForMpsFile[i + 1][0] << " "
                << "Z"
                << " " << setprecision(numeric_limits<long double>::digits10)
                << outputForMpsFile[i][0] << endl;
        mpsFile << " "
                << "X" << outputForMpsFile[i + 1][0] << " "
                << "R" << outputForMpsFile[i + 1][1] << " "
                << outputForMpsFile[i + 1][2] << endl;
        mpsFile << " "
                << "X" << outputForMpsFile[i + 2][0] << " "
                << "R" << outputForMpsFile[i + 2][1] << " "
                << outputForMpsFile[i + 2][2] << endl;
        mpsFile << " "
                << "X" << outputForMpsFile[i + 3][0] << " "
                << "R" << outputForMpsFile[i + 3][1] << " "
                << outputForMpsFile[i + 3][2] << endl;
        mpsFile << " "
                << "X" << outputForMpsFile[i + 1][3] << " "
                << "Z"
                << " " << setprecision(numeric_limits<long double>::digits10)
                << outputForMpsFile[i][0] << endl;
        mpsFile << " "
                << "X" << outputForMpsFile[i + 1][3] << " "
                << "R" << outputForMpsFile[i + 1][4] << " "
                << outputForMpsFile[i + 1][5] << endl;
        mpsFile << " "
                << "X" << outputForMpsFile[i + 2][3] << " "
                << "R" << outputForMpsFile[i + 2][4] << " "
                << outputForMpsFile[i + 2][5] << endl;
        mpsFile << " "
                << "X" << outputForMpsFile[i + 3][3] << " "
                << "R" << outputForMpsFile[i + 3][4] << " "
                << outputForMpsFile[i + 3][5] << endl;
      }
    }
  }

  // Second part of the MPS file: Define the r vector into the mps file
  // Compute the normals below when the number of inputs is 3
  else {
    ifstream normals_(
        normals_file.c_str());  // Read the file that contains all normals
    int numberNormals_;  // Variable to define the number of rows of the matrix
                         // mNormals
    if (normals_.is_open()) {
      normals_ >> numberNormals_;  // Obtain the total number of input normals
    }

    // mNormal is a matrix containing all the normal vectors
    std::vector<std::vector<double>> mNormals(
        numberNormals_, std::vector<double>(3, 0));  // numberNormals_

    if (normals_.is_open()) {
      for (int i = 0; i < numberNormals_; i++) {
        normals_ >> mNormals[i][0] >> mNormals[i][1] >> mNormals[i][2];
      }
    }

    // Extract the numbers from the matrix that contains the normals
    std::vector<std::vector<int>> BoundaryVector;
    for (int i = 0; i < numberNormals_; i++) {
      std::string file_name =
          "MPS_" + file_name2 + "_rhs_" + itoa(i + 1) +
          ".txt";  // file_name2 comes from the second input in the command line

      ofstream mpsFile(file_name.c_str(), std::ofstream::out);
      if (mpsFile.is_open()) {
        mpsFile << "RHS" << endl;

        // Obtain the vector that defines the 1D boundary
        std::vector<std::vector<int>> BoundaryVector;

        std::vector<double> normalToPlane;
        normalToPlane.push_back(mNormals.at(i).at(0));
        normalToPlane.push_back(mNormals.at(i).at(1));
        normalToPlane.push_back(mNormals.at(i).at(2));

        // Finds coordinates of 3 vectors normal to the normal vector.
        std::vector<std::vector<double>> pointsOnPlane =
            topology.getCoordinatesOfTriangle(normalToPlane);

        // Finds the Node (stk::mesh::Entity of rank 0) that is closest to each
        // of the points on the plane
        std::vector<stk::mesh::Entity> closestNodes =
            topology.getClosestNodesOnSurface(pointsOnPlane);

        // Finds the identifiers of the nodes (entity rank 0) along the shortest
        // path connecting the three points
        std::vector<std::vector<int>> ShortestPathFinal =
            topology.shortestpath(closestNodes);
        BoundaryVector = topology.boundaryVector(
            ShortestPathFinal);  //.boundaryVectorOuterSurface
        for (unsigned int i = 0; i < BoundaryVector.size(); i++) {
          mpsFile << " "
                  << "RHS1"
                  << " "
                  << "R" << BoundaryVector[i][0] << " " << BoundaryVector[i][1]
                  << endl;
        }

        // Define the bounds of the variables X1, X2, ...etc
        mpsFile << "BOUNDS" << endl;
        mpsFile << "ENDATA" << endl;
      }  // End of second part of mps file
    }    // End of for loop
  }
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
