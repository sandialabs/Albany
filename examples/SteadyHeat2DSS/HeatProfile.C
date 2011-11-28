#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

/*
 * This code calculates a solution of the heat equation on a two dimensional domain 0 <= x <= 1
 * and 0 <= y <= 1, with thermal cond k = 1. However, we have BC's of T(1, y) = 1 and 
 * dT/dx (0, y) = q (constant), so the solution is really one dimensional (in x):
 * T(x, y) = c_1 + c_2 x
 *
 * With the BC's, we find dT/dx = c_2 = q, and T(1) = c_1 + q x
 * c_1 is thus = 1 - q, so the general solution is T(x, y) = q(1 - x) + 1
 *
 */

using namespace std;

int main(){

  double avg = 0;
  double two_norm = 0;
  double max = -1000;
  double value;
  double q = 5;

  // Read the ncdump file to get the x locations of the nodes in the mesh.

  FILE *ifp = fopen("quadQuadSS.ncdump", "r");
  char word[BUFSIZ];
  int num_nodes;

  if(ifp == NULL) perror("Cannot open ncdump input file");

  // Gobble the file, one word at a time

  while( fscanf(ifp, "%s", word) == 1){

    // Look for the word num_nodes

    if(strcmp(word, "num_nodes") == 0){

      fscanf(ifp, "%*s%d", &num_nodes);

      break;

    }
  }

  double *x = new double[num_nodes];

  while( fscanf(ifp, "%s", word) == 1){

    // Look for the word coord

    if(strcmp(word, "coord") == 0){

      fscanf(ifp, "%*s"); // gobble the equal sign

      for(int i = 0; i < num_nodes; i++) // grab the x coordinate values

        if(fscanf(ifp, "%lf,%*f,%*f,", &x[i]) != 1){
          cout << "Error" << endl;
          return -1;
        }

      break;

    }
  }

  fclose(ifp);

	ofstream out;

	out.open("reference_solution.dat", ios::out);

	out << "%%MatrixMarket matrix array real general" << endl;

	out << "% Steady 2D Heat Equation, with side sets" << endl;

	out << num_nodes << " 1" << endl; // Write M and N values

	for(int i = 0; i < num_nodes; i++){

    value = q * (1.0 - x[i]) + 1.0;
		out << value << endl;

    avg += value;
    two_norm += value * value;
    if(value > max) max = value;

  }

	out.close();

  avg /= (double)num_nodes;
  two_norm = sqrt(two_norm);

  cout << "Solution Average = " << avg << endl;
  cout << "Solution Two Norm = " << two_norm << endl;
  cout << "Solution Max Value = " << max << endl;

  return 0;

}

	
