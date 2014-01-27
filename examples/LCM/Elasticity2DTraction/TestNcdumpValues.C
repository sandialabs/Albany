#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

/*
 * This code compares the output of the Albany ncdump file with a known solution
 *
 * The test problem is a benchmark in linear elasticity from:
 * E. Stein, P. Wriggers, A. Rieger, and M. Schmidt, "Benchmarks," in Error-Controlled Adaptive
 * Finite Elements in Solid Mechanics, E. Stein, ed., John Wiley and Sons, New York, 2002, pp. 385-404.
 * 
 * This particular solution compares with the benchmark problem shown in section 6 of:
 * Zhiqiang Cai and Gerhard Starke, "Least-Squares Methods for Linear Elasticity," SIAM J. Numer. Anal., 42(2),
 * pp. 826-842, (2005).
 * 
 * We specifically compare with the 511 element case shown in Table 6.1, that gives \sigma_{22} at (1.0, 0.0)
 * of 13.7213
 *
 * This test problem is different; it employs a quad mesh with 435 elements with a bit of refinement down by this
 * area. The point (1.0, 0.0) is contained in element 47, as determined using paraview. So, this code finds the
 * solution for \sigma_{22} (held in the ncdump file in the variable vals_elem_var4eb1), for element 47.
 *
 * Questions to Glen Hansen gahanse@sandia.gov
 *
 */

using namespace std;

int main(){

  double target = 13.7213; // Cai reference above
  double tolerance = 1.0e-03;

  // Read the ncdump file to get the \sigma_{22} value in element 47

  FILE *ifp = fopen("hole_out.ncdump", "r");
  char word[BUFSIZ];
  double val;

  if(ifp == NULL) perror("Cannot open ncdump input file");

  // Gobble the file, one word at a time

  while( fscanf(ifp, "%s", word) == 1){

    // Look for the word "vals_elem_var4eb1"

    if(strcmp(word, "vals_elem_var4eb1") == 0){

      // gobble the "="

      fscanf(ifp, "%s", word);

      for(int i = 0; i < 47; i++) // get the value in element 47

        fscanf(ifp, "%lf,", &val);

        // test it

	      fclose(ifp);

        if(fabs(target - val) / target >= tolerance){ // test failed

          std::cout << "Error: stress_22 value outside of tolerance: \n"
            << " target = " << target << " calculated = " << val << "\ntolerance = " 
            << tolerance << " error = " << fabs(target - val) / target  << std::endl;

            return -1;

        }

        return 0;

    }
  }

  // shouldnt be here

	fclose(ifp);

  return -2;

}

	
