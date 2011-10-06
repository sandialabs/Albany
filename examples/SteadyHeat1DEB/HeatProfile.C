#include <iostream>
#include <fstream>

using namespace std;

main(){

	int num_nodes = 101;

	double minx = 0.0;
	double maxx = 1.0;

	double dx = (maxx - minx) / static_cast<double>(num_nodes - 1);

	ofstream out;

	out.open("reference_solution.dat", ios::out);

	out << "%%MatrixMarket matrix array real general" << endl;

	out << "% Steady 1D Heat Equation, with element blocks" << endl;

	out << num_nodes << " 1" << endl; // Write M and N values

    // Soln is 1.0 @ x=0 and 0.0 @ x=1, and linear in between

	for(int i = 0; i < num_nodes; i++)

		out << 1.0 - i * dx << endl;

	out.close();

}

	
