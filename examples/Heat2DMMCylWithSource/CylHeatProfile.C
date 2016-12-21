#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

/*
 * This code calculates a solution of the heat equation as described in MMHeatExample.tex
 *
 */

double sqr(double x){ return x * x;}

using namespace std;

int main(){

  double max = -1000;
  double value;
  double q = 20095; // W / m^3

  // problem constants

  double r1 = 0.0046482;
  double r2 = 0.0051562;
  double r3 = 0.01;

  double k1 = 4.9844;
  double k2 = 17.033;
  double k3 = 0.0004;

  double T3 = 313.15;

  double T1 = (q * sqr(r1) / 2.0) * (log(r2/r1) / k2 +
          log(r3/r2) / k3) + T3;

  double T2 = (q * sqr(r1) / 2.0) * (log(r3/r2) / k3) + T3;

  // Read the ncdump file to get the x locations of the nodes in the mesh.

  FILE *ifp = fopen("fuel_volume.ncdump", "r");
  char word[BUFSIZ];
  int num_nodes;

  if(ifp == NULL) perror("Cannot open ncdump input file");

  // Gobble the file, one word at a time

  while( fscanf(ifp, "%s", word) == 1){

    // Look for the word num_nodes

    if(strcmp(word, "num_nodes") == 0){

      if(fscanf(ifp, "%*s%d", &num_nodes) != 1){
        cout << "Error" << endl;
        return -1;
      }

      break;

    }
  }

  double *x = new double[num_nodes];
  double *y = new double[num_nodes];

  while( fscanf(ifp, "%s", word) == 1){

    // Look for the word coord

    if(strcmp(word, "coord") == 0){

      if(fscanf(ifp, "%*s") != 0) { // gobble the equal sign
        cout << "Error" << endl;
        return -1;
      }

      for(int i = 0; i < num_nodes; i++) // grab the x coordinate values

        if(fscanf(ifp, "%lf,", &x[i]) != 1){
          cout << "Error" << endl;
          return -1;
        }

      for(int i = 0; i < num_nodes - 1; i++) // grab the y coordinate values (all but the last one)

        if(fscanf(ifp, "%lf,", &y[i]) != 1){ // comma follows
          cout << "Error" << endl;
          return -1;
        }

      if(fscanf(ifp, "%lf", &y[num_nodes - 1]) != 1){ // Get the last one
        cout << "Error" << endl;
        return -1;
      }

      break;

    }
  }

  fclose(ifp);

	ofstream out;

	out.open("reference_solution.dat", ios::out);
  out.precision(10);

	out << "%%MatrixMarket matrix array real general" << endl;

	out << "% Steady 2D Heat Equation, multimaterial, cylindrical geometry" << endl;

	out << num_nodes << " 1" << endl; // Write M and N values

  double rad;

	for(int i = 0; i < num_nodes; i++){

    rad = sqrt(sqr(x[i]) + sqr(y[i]));

  // Three regions, fuel, clad, and cask

    if(rad <= r1){

      value = (q / (4.0 * k1))*(sqr(r1) - sqr(rad)) + (q * sqr(r1) / 2.0) * (log(r2/r1) / k2 +
          log(r3/r2) / k3) + T3;

    }
    else if(rad <= r2){

      value = (T1 -T2) / log(r1/r2) * log(rad / r2) + T2;

    }
    else {

      value = (T2 -T3) / log(r2/r3) * log(rad / r3) + T3;

    }

		out << value << endl;

    if(value > max) max = value;


  }

	out.close();

  cout << "Solution Max Value = " << max << endl;

// Write out a gnuplot file to look at the temperature profile in 1D

  int num_gplot_nodes = 100;
	ofstream output;

	// Write the Gnuplot driver file

	output.open("HeatProfile.plt", std::ios::out);
  output.precision(10);

	output <<

	       "set ylabel \"Temperature (K)\"				" << endl <<
	       "set ytics nomirror						" << endl <<
	       "										" << endl <<
	       "set xtics nomirror						" << endl <<
	       "set xlabel \"Radius (m)\"						" << endl <<
	       "										" << endl <<
	       "plot 'HeatProfile.dat' with lines title \"Temp Profile\" " << endl;

	output.close();
	output.open("HeatProfile.dat", std::ios::out);

	for(int i = 0; i <= num_gplot_nodes; i++){

    rad = i * r3 / (double)num_gplot_nodes;

  // Three regions, fuel, clad, and cask

    if(rad <= r1){

      value = (q / (4.0 * k1))*(sqr(r1) - sqr(rad)) + (q * sqr(r1) / 2.0) * (log(r2/r1) / k2 +
          log(r3/r2) / k3) + T3;

    }
    else if(rad <= r2){

      value = (T1 -T2) / log(r1/r2) * log(rad / r2) + T2;

    }
    else {

      value = (T2 -T3) / log(r2/r3) * log(rad / r3) + T3;

    }

		output << rad << " " << value << endl;

  }

	output.close();

  return 0;

}

	
