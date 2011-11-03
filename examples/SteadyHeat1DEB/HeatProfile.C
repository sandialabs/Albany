#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

int main(){

	int num_nodes = 101;

	double minx = 0.0;
	double maxx = 1.0;

	double dx = (maxx - minx) / static_cast<double>(num_nodes - 1);

  double avg = 0;
  double two_norm = 0;
  double max = -1000;
  double value;

	ofstream out;

	out.open("reference_solution.dat", ios::out);

	out << "%%MatrixMarket matrix array real general" << endl;

	out << "% Steady 1D Heat Equation, with element blocks" << endl;

	out << num_nodes << " 1" << endl; // Write M and N values

    // Soln is 1.0 @ x=0 and 0.0 @ x=1
    // MatA (Plastic) goes from x=0 to x=0.5 and has k=1.0
    // MatB (Metal) goes from x=0.5 to x=1.0 and has k = 10
    // These values are set in materials.xml 

    // q = (T1 - T3) / ((L1/k1) + (L2/k2)) = 1.818181 ...

	for(int i = 0; i < num_nodes / 2; i++){

    // T = 1.0 - Lq/k
		value = 1.0 - (i * dx)*1.818181818181;
		out << value << endl;

    avg += value;
    two_norm += value * value;
    if(value > max) max = value;

  }

	for(int i = num_nodes / 2; i < num_nodes; i++){

    // T = Lq/k (L from R bound) = (1.0 - L)q/k
		value = (1.0 - i * dx)*1.818181818181/10.0;
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

	
