//
// Test for LCM utilities
//

#include "Tensor.h"

typedef double ScalarT;

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

  if (InvalidCmdLineArgs == true || (ac > 4)) {
    std::cout << "Invalid command line arguments. Use:" << std::endl;
    std::cout << "\t -v verbose (number of failed/passed tests)" << std::endl;
    std::cout << "\t -d debug (same as verbose)" << std::endl;
    std::cout << "\t -m matlab-style output" << std::endl;
    return 1;
  }

  //
  // The tests
  //
  int PassedTestCount = 0;
  const int TotalTests = 9;
  bool passed = false;

  //
  // Test 1
  //
  LCM::Vector<ScalarT> u(1.0, 0.0, 0.0);
  LCM::Vector<ScalarT> v(0.0, 1.0, 0.0);
  LCM::Vector<ScalarT> w(1.0, 1.0, 0.0);

  passed = (u + v == w);

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
      std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  //
  // Test 2
  //
  u(0) = 1.0;
  u(1) = 2.0;
  u(2) = 3.0;

  v = u - u;

  passed = (LCM::norm(v) == 0.0);

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
      std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  //
  // Test 3
  //
  v(0) = -2.0;
  v(1) = -4.0;
  v(2) = -6.0;

  w = 4.0 * u + 2.0 * v;

  passed = (LCM::norm(w) == 0.0);

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << u << std::endl;
      std::cout << v << std::endl;
      std::cout << w << std::endl;
  }

  //
  // Test 4
  //
  LCM::Tensor<ScalarT> A(1.0);
  LCM::Tensor<ScalarT> B(2.0);
  LCM::Tensor<ScalarT> C(3.0);

  passed = (C == A + B);

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  //
  // Test 5
  //
  A = LCM::eye<ScalarT>();
  A = 2.0 * A;
  A(1,0) = A(0,1) = 1.0;
  A(2,1) = A(1,2) = 1.0;

  B = LCM::inverse(A);

  C = A * B;

  passed = (LCM::norm(C - LCM::eye<ScalarT>()) <=
        std::numeric_limits<ScalarT>::epsilon());

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << A << std::endl;
      std::cout << B << std::endl;
      std::cout << C << std::endl;
  }

  //
  // Test 6
  //
  ScalarT I1 = LCM::I1(A);
  ScalarT I2 = LCM::I2(A);
  ScalarT I3 = LCM::I3(A);

  u(0) = I1 - 6;
  u(1) = I2 - 10;
  u(2) = I3 - 4;

  passed = (LCM::norm(u) <= std::numeric_limits<ScalarT>::epsilon());

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << I1 << std::endl;
      std::cout << I2 << std::endl;
      std::cout << I3 << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
  }

  //
  // Test 7
  //
  A = LCM::eye<ScalarT>();
  B = LCM::log(A);

  passed = (LCM::norm(B) <= std::numeric_limits<ScalarT>::epsilon());

  C = LCM::exp(B);

  C -= A;

  passed = passed && (LCM::norm(C) <= std::numeric_limits<ScalarT>::epsilon());

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << A << std::endl;
      std::cout << B << std::endl;
      std::cout << C << std::endl;
  }

  //
  // Test 8
  //
  A = LCM::eye<ScalarT>();
  A(0,1) = 0.1;
  A(1,0) = 0.1;
  LCM::Tensor<ScalarT> eVec;
  LCM::Tensor<ScalarT> eVal;
  boost::tie(eVec,eVal) = LCM::eig_spd(A);

  passed = (std::abs(eVal(0,0) - 1.0) <= std::numeric_limits<ScalarT>::epsilon());
  passed = passed && (std::abs(eVal(1,1) - 0.9) <= std::numeric_limits<ScalarT>::epsilon());
  passed = passed && (std::abs(eVal(2,2) - 1.1) <= std::numeric_limits<ScalarT>::epsilon());

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << A << std::endl;
      std::cout << eVec << std::endl;
      std::cout << eVal << std::endl;
  }

  //
  // Test 9
  //
  LCM::Tensor<ScalarT> V0(1.1, 0.2, 0.0,
			  0.2, 1.0, 0.0,
			  0.0, 0.0, 1.2);
  LCM::Tensor<ScalarT> R0(sqrt(2)/2, -sqrt(2)/2, 0.0,
			  sqrt(2)/2,  sqrt(2)/2, 0.0,
			  0.0,        0.0,       1.0);

  LCM::Tensor<ScalarT> F = V0*R0;
  LCM::Tensor<ScalarT> V;
  LCM::Tensor<ScalarT> R;
  boost::tie(V,R) = LCM::polar_left(F);

  passed = (LCM::norm(V-V0) <= 10*std::numeric_limits<ScalarT>::epsilon());
  passed = passed && (LCM::norm(R-R0) <= std::numeric_limits<ScalarT>::epsilon());

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << F << std::endl;
      std::cout << V0 << std::endl;
      std::cout << V << std::endl;
      std::cout << R0 << std::endl;
      std::cout << R << std::endl;
      std::cout << LCM::norm(V-V0) << std::endl;
      std::cout << LCM::norm(R-R0) << std::endl;
  }

  return 0;
}
