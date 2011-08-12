//
// Test for LCM utilities
//

#include "Tensor.h"

typedef double ScalarT;

int main(int ac, char* av[])
{

  //
  // Initializations and parsing of command line
  //
  bool verbose = false;
  bool debug = true;
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
  const int TotalTests = 14;
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
  }

  //
  // Test 10
  //
  R = LCM::identity<ScalarT>();

  LCM::Tensor<ScalarT> r  = log_rotation(R);
  LCM::Tensor<ScalarT> r0 = log_rotation(R0);

  passed = LCM::norm(r - LCM::zero<ScalarT>()) <= std::numeric_limits<ScalarT>::epsilon();
  passed = passed && std::abs(r0(0,1) + 0.785398163397448) <=
		   10*std::numeric_limits<ScalarT>::epsilon();
  passed = passed && std::abs(r0(0,1) + r0(1,0)) <= 10*std::numeric_limits<ScalarT>::epsilon();

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << R0 << std::endl;
      std::cout << r0 << std::endl;
  }

  //
  // Test 11
  //
  F = 3.0*LCM::identity<ScalarT>();
  LCM::Tensor<ScalarT> logV;
  LCM::Tensor<ScalarT> logR;

  boost::tie(V,R,logV) = polar_left_logV(F);
  logR = log_rotation(R);

  LCM::Tensor<ScalarT> f = LCM::bch(logV,logR);

  passed = std::abs(f(0,0) - std::log(3.0)) <= std::numeric_limits<ScalarT>::epsilon();

  if (passed == true) {
    PassedTestCount++;
  }

  if(verbose || debug) {
    std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
      std::cout << std::endl;
  }

  if(matlab) {
      std::cout << f << std::endl;
      std::cout << logV << std::endl;
      std::cout << logR << std::endl;
  }


//
// Test 12
//
// Test the log mapping of element from SO(3) group. In this test,
//a reference solution is  calculated by hand. This solution is compared
//with the solution computed via function log_rotation.
// rotation angle equal to pi with respect to z axis in this case.
// added by WaiChing Sun 8/4/2011

  //Create a 0.3 rad rotation use Z axis. Store it at R.
  double theta = std::acos(-1.0); // rotation angle

  R(0,0) =  cos(theta);
  R(1,1) =  cos(theta);
  R(0,1) =  sin(theta);
  R(1,0) = -sin(theta);
  R(2,2) =  1.0;

  // Compute log R via Rodrigues' rotation formulas
  logR = log_rotation(R); // reuse logR and reassign component via log_rotation

  // Enter solution computed by hand
  LCM::Tensor<ScalarT> Rref(0.0); // create 3-by-3 zero tensor
  Rref(0,1) = std::acos(-1.0);
  Rref(1,0) = -std::acos(-1.0);



  // Compute those two values (if correct, passed == 1)
  passed =(norm(logR - Rref) <= 100*std::numeric_limits<ScalarT>::epsilon()) ||
		  ( norm(logR + Rref) <= 100*std::numeric_limits<ScalarT>::epsilon());
  if (passed == true) {
      PassedTestCount++;
  } else
  {
	  std::cout << "Error in logarithmic mapping for SO(3) in Test 12" << std::endl;
	  std::cout << norm(logR - Rref) << std::endl;
  }

    if(verbose || debug) {
      std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
        std::cout << std::endl;
    }

    if(matlab) {
        std::cout << logR << std::endl;
    }

//
// Test 13
//
// Test the exp mapping from so(3) to SO(3). In this test,
// a reference solution is  calculated by hand. This solution is compared
// with the solution computed via function log_rotation.
// rotation angle equal to pi with respect to z axis in this case.
// added by WaiChing Sun 8/8/2011.
   u(0) = std::acos(-1.0)/std::sqrt(2.0); //generate basis vector
   u(1) = u(0);
   u(2) = 0.0;

   LCM::Tensor<ScalarT> R1(0.0); // create 3-by-3 zero tensor
   LCM::Tensor<ScalarT> logR2(0.0); // create 3-by-3 zero tensor
   logR2(0,2) =  u(1);
   logR2(1,2) = -u(0);
   logR2(2,0) = -u(1);
   logR2(2,1) =  u(0);
   logR2(0,1) = -u(2);
   logR2(1,0) =  u(2);

   R1 = exp_skew_symmetry(logR2); // perform log mapping
   Rref = LCM::zero<ScalarT>();
   Rref(0,1) = 1.0;
   Rref(1,0) = 1.0;
   Rref(2,2) = -1.0;
   // Compare exp(log(R)) with R
   passed = norm(Rref- R1) <= 100*std::numeric_limits<ScalarT>::epsilon();

   if (passed == true) {
       PassedTestCount++;
   } else {
	   std::cout << "Error in logarithmic mapping for SO(3) in Test 13" << std::endl;
	   std::cout << norm(R1 - Rref) << std::endl;
   }

   if(verbose || debug) {
		std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
		  std::cout << std::endl;
  }


//
// Test 14
//
// Test exp and log mapping between SO(3) and so(3) groups. We check whether
// exp(log(R)) - R = 0 using the log(R) computed in test 12.
// added by WaiChing Sun 8/4/2011

    // Compare exp(log(R)) with R
    passed = norm(exp_skew_symmetry(logR) - R) <= 100*std::numeric_limits<ScalarT>::epsilon();

    if (passed == true) {
          PassedTestCount++;
    } else
    {
        std::cout << "Error in exponential mapping for so(3) in Test 14" << std::endl;
    	std::cout << norm(exp_skew_symmetry(logR) - R) << std::endl;
    }
    if(verbose || debug) {
          std::cout << "Tensor: passed " << PassedTestCount << " of " << TotalTests;
            std::cout << std::endl;
    }



  return PassedTestCount == TotalTests ? 0 : 1;

}
