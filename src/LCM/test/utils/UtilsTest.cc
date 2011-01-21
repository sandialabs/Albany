//
// Test for LCM utilities
//

#include "Tensor.h"

int main(int ac, char* av[])
{

  //
  // Intializations and parsing of command line
  //
  bool verbose = false;
  bool debug = false;
  bool matlab = false;
  bool InvalidCmdLineArgs = false;

  for (int i = 1; i < ac; ++i) {
    if (av[i][0] == '-') {
      switch (av[i][1]) {
      case 'v':
        if(verbose == false) {
          verbose = true;
        }
        else {
          InvalidCmdLineArgs = true;
        }
        break;
      case 'd':
        if (debug == false) {
          debug = true;
        }
        else {
          InvalidCmdLineArgs = true;
        }
        break;
      case 'm':
        if(matlab == false) {
          matlab = true;
        }
        else {
          InvalidCmdLineArgs = true;
        }
        break;
      default:
        InvalidCmdLineArgs = true;
        break;
      }
    }
  }

  if(InvalidCmdLineArgs == true || (ac > 4)) {
    std::cout << "Invalid command line arguments detected. Use the following flags:" << std::endl
      << "\t -v enables verbose mode (reports number of failed/successful tests)" << std::endl
      << "\t -d enables debug mode (same as verbose with output of each test)" << std::endl
      << "\t -m enables matlab-style output; only has an effect if debug mode is enabled" << std::endl;
    return 1;
  }

  int PassedTestCount = 0;
  const int TotalTests = 2;
  bool passed = false;

  //
  // The tests
  //

  LCM::Vector<double> u(1.0, 0.0, 0.0);
  LCM::Vector<double> v(0.0, 1.0, 0.0);
  LCM::Vector<double> w(1.0, 1.0, 0.0);

  passed = (u + v == w);

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
      std::cout << "Tensor: " << PassedTestCount << " of " << TotalTests << " tests were successful.";
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << u << std::endl;
      std::cout << v << std::endl;
      std::cout << w << std::endl;
  }

  LCM::Tensor4<double> A(1.0);
  LCM::Tensor4<double> B(2.0);
  LCM::Tensor4<double> C(3.0);

  passed = (C == A + B);

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
      std::cout << "Tensor: " << PassedTestCount << " of " << TotalTests << " tests were successful.";
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << A << std::endl;
      std::cout << B << std::endl;
      std::cout << C << std::endl;
  }

  return 0;
}
