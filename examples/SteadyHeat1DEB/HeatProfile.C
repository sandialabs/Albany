#include <iostream>
#include <fstream>

using namespace std;

int main(){

	int num_nodes = 101;

	double minx = 0.0;
	double maxx = 1.0;

	double dx = (maxx - minx) / static_cast<double>(num_nodes - 1);

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

	for(int i = 0; i < num_nodes / 2; i++)

    // T = 1.0 - Lq/k
		out << 1.0 - (i * dx)*1.818181818181 << endl;

	for(int i = num_nodes / 2; i < num_nodes; i++)

    // T = Lq/k (L from R bound) = (1.0 - L)q/k
		out << (1.0 - i * dx)*1.818181818181/10.0 << endl;

	out.close();

  return 0;

}

	
