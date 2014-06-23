//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <cmath>
#include <ctime>
#include <cstdlib>

#include "AAdapt_AnalyticFunction.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Exceptions.hpp"

#include "Aeras_ShallowWaterConstants.hpp"

const double pi = 3.141592653589793;


// Factory method to build functions based on a string
Teuchos::RCP<AAdapt::AnalyticFunction> AAdapt::createAnalyticFunction(
  std::string name, int neq, int numDim,
  Teuchos::Array<double> data) { 
  Teuchos::RCP<AAdapt::AnalyticFunction> F;

  if(name == "Constant")
    F = Teuchos::rcp(new AAdapt::ConstantFunction(neq, numDim, data));

  else if(name == "1D Gauss-Sin")
    F = Teuchos::rcp(new AAdapt::GaussSin(neq, numDim, data));

  else if(name == "1D Gauss-Cos")
    F = Teuchos::rcp(new AAdapt::GaussCos(neq, numDim, data));

  else if(name == "Linear Y")
    F = Teuchos::rcp(new AAdapt::LinearY(neq, numDim, data));

  else if(name == "Gaussian Pressure")
    F = Teuchos::rcp(new AAdapt::GaussianPress(neq, numDim, data));

  else if(name == "Sin-Cos")
    F = Teuchos::rcp(new AAdapt::SinCos(neq, numDim, data));

  else if(name == "Taylor-Green Vortex")
    F = Teuchos::rcp(new AAdapt::TaylorGreenVortex(neq, numDim, data));

  else if(name == "1D Acoustic Wave")
    F = Teuchos::rcp(new AAdapt::AcousticWave(neq, numDim, data));

  else if(name == "Aeras Schar Density")
    F = Teuchos::rcp(new AAdapt::AerasScharDensity(neq, numDim, data));

  else if(name == "Aeras X Scalar Advection")
    F = Teuchos::rcp(new AAdapt::AerasXScalarAdvection(neq, numDim, data));

  else if(name == "Aeras XZ Hydrostatic")
    F = Teuchos::rcp(new AAdapt::AerasXZHydrostatic(neq, numDim, data));

  else if(name == "Aeras Hydrostatic")
    F = Teuchos::rcp(new AAdapt::AerasHydrostatic(neq, numDim, data));

  else if(name == "Aeras Heaviside")
    F = Teuchos::rcp(new AAdapt::AerasHeaviside(neq, numDim, data));

  else if(name == "Aeras CosineBell")
      F = Teuchos::rcp(new AAdapt::AerasCosineBell(neq, numDim, data));

  else if(name == "Aeras ZonalFlow") //this used to be called TestCase2.  Irina has renamed it so it can be used for test case 5 too.
      F = Teuchos::rcp(new AAdapt::AerasZonalFlow(neq, numDim, data));

  else if(name == "Aeras PlanarCosineBell")
        F = Teuchos::rcp(new AAdapt::AerasPlanarCosineBell(neq, numDim, data));

  else if(name == "Aeras RossbyHaurwitzWave")
      F = Teuchos::rcp(new AAdapt::AerasRossbyHaurwitzWave(neq, numDim, data));

  else if(name == "Aeras TC5Init")
    F = Teuchos::rcp(new AAdapt::AerasTC5Init(neq, numDim, data));

  else
    TEUCHOS_TEST_FOR_EXCEPTION(name != "Valid Initial Condition Function",
        std::logic_error,
        "Unrecognized initial condition function name: " << name);

  return F;
}


//*****************************************************************************
AAdapt::ConstantFunction::ConstantFunction(int neq_, int numDim_,
    Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((data.size() != neq),
                             std::logic_error,
                             "Error! Invalid specification of initial condition: incorrect length of Function Data for Constant Function; neq = " << neq << ", data.size() = " << data.size() <<  std::endl) ;
}
void AAdapt::ConstantFunction::compute(double* x, const double* X) {
  if(data.size() > 0)
    for(int i = 0; i < neq; i++)
      x[i] = data[i];
}

//*****************************************************************************
// Private convenience function
long AAdapt::seedgen(int worksetID) {
  long seconds, s, seed, pid;

  pid = getpid();
  s = time(&seconds);    /* get CPU seconds since 01/01/1970 */

  // Use worksetID to give more randomness between calls

  seed = abs(((s * 181) * ((pid - 83) * 359) * worksetID) % 104729);
  return seed;
}

AAdapt::ConstantFunctionPerturbed::ConstantFunctionPerturbed(int neq_, int numDim_,
    int worksetID,
    Teuchos::Array<double> data_,  Teuchos::Array<double> pert_mag_)
  : numDim(numDim_), neq(neq_), data(data_), pert_mag(pert_mag_) {

  TEUCHOS_TEST_FOR_EXCEPTION((data.size() != neq || pert_mag.size() != neq),
                             std::logic_error,
                             "Error! Invalid specification of initial condition: incorrect length of " <<
                             "Function Data for Constant Function Perturbed; neq = " << neq <<
                             ", data.size() = " << data.size()
                             << ", pert_mag.size() = " << pert_mag.size()
                             <<  std::endl) ;

  //  srand( time(NULL) ); // seed the random number gen
  srand(seedgen(worksetID)); // seed the random number gen

}

void AAdapt::ConstantFunctionPerturbed::compute(double* x, const double* X) {
  for(int i = 0; i < neq; i++)
    x[i] = data[i] + udrand(-pert_mag[i], pert_mag[i]);
}

// Private convenience function
double AAdapt::ConstantFunctionPerturbed::udrand(double lo, double hi) {
  static const double base = 1.0 / (RAND_MAX + 1.0);
  double deviate = std::rand() * base;
  return lo + deviate * (hi - lo);
}
//*****************************************************************************
AAdapt::ConstantFunctionGaussianPerturbed::ConstantFunctionGaussianPerturbed(int neq_, int numDim_,
    int worksetID,
    Teuchos::Array<double> data_,  Teuchos::Array<double> pert_mag_)
  : numDim(numDim_),
    neq(neq_),
    data(data_),
    pert_mag(pert_mag_),
    //      rng(boost::random::random_device()()), // seed the rng
    rng(seedgen(worksetID)), // seed the rng
    nd(neq_),
    var_nor(neq_) {


  TEUCHOS_TEST_FOR_EXCEPTION((data.size() != neq || pert_mag.size() != neq),
                             std::logic_error,
                             "Error! Invalid specification of initial condition: incorrect length of " <<
                             "Function Data for Constant Function Gaussian Perturbed; neq = " << neq <<
                             ", data.size() = " << data.size()
                             << ", pert_mag.size() = " << pert_mag.size()
                             <<  std::endl) ;

  if(data.size() > 0 && pert_mag.size() > 0)
    for(int i = 0; i < neq; i++)
      if(pert_mag[i] > std::numeric_limits<double>::epsilon()) {

        nd[i] = Teuchos::rcp(new boost::normal_distribution<double>(data[i], pert_mag[i]));
        var_nor[i] = Teuchos::rcp(new
                                  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(rng, *nd[i]));

      }

}

void AAdapt::ConstantFunctionGaussianPerturbed::compute(double* x, const double* X) {

  for(int i = 0; i < neq; i++)
    if(var_nor[i] != Teuchos::null)
      x[i] = (*var_nor[i])();

    else
      x[i] = data[i];

}


//*****************************************************************************
AAdapt::GaussSin::GaussSin(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq != 1) || (numDim != 1) || (data.size() != 1),
                             std::logic_error,
                             "Error! Invalid call of GaussSin with " << neq
                             << " " << numDim << "  " << data.size() << std::endl);
}
void AAdapt::GaussSin::compute(double* x, const double* X) {
  x[0] =     sin(pi * X[0]) + 0.5 * data[0] * X[0] * (1.0 - X[0]);
}

//*****************************************************************************
AAdapt::GaussCos::GaussCos(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq != 1) || (numDim != 1) || (data.size() != 1),
                             std::logic_error,
                             "Error! Invalid call of GaussCos with " << neq
                             << " " << numDim << "  " << data.size() << std::endl);
}
void AAdapt::GaussCos::compute(double* x, const double* X) {
  x[0] = 1 + cos(2 * pi * X[0]) + 0.5 * data[0] * X[0] * (1.0 - X[0]);
}
//*****************************************************************************
AAdapt::LinearY::LinearY(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq < 2) || (numDim < 2) || (data.size() != 1),
                             std::logic_error,
                             "Error! Invalid call of LinearY with " << neq
                             << " " << numDim << "  " << data.size() << std::endl);
}
void AAdapt::LinearY::compute(double* x, const double* X) {
  x[0] = 0.0;
  x[1] = data[0] * X[0];

  if(numDim > 2) x[2] = 0.0;
}
//*****************************************************************************
AAdapt::GaussianPress::GaussianPress(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq < 3) || (numDim < 2) || (data.size() != 4),
                             std::logic_error,
                             "Error! Invalid call of GaussianPress with " << neq
                             << " " << numDim << "  " << data.size() << std::endl);
}
void AAdapt::GaussianPress::compute(double* x, const double* X) {
  for(int i = 0; i < neq - 1; i++) {
    x[i] = 0.0;
  }

  x[neq - 1] = data[0] * exp(-data[1] * ((X[0] - data[2]) * (X[0] - data[2]) + (X[1] - data[3]) * (X[1] - data[3])));
}
//*****************************************************************************
AAdapt::SinCos::SinCos(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq < 3) || (numDim < 2),
                             std::logic_error,
                             "Error! Invalid call of SinCos with " << neq
                             << " " << numDim << "  " << data.size() << std::endl);
}
void AAdapt::SinCos::compute(double* x, const double* X) {
  x[0] = sin(2.0 * pi * X[0]) * cos(2.0 * pi * X[1]);
  x[1] = cos(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]);
  x[2] = sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]);
}
//*****************************************************************************
AAdapt::TaylorGreenVortex::TaylorGreenVortex(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq < 3) || (numDim != 2),
                             std::logic_error,
                             "Error! Invalid call of TaylorGreenVortex with " << neq
                             << " " << numDim << "  " << data.size() << std::endl);
}
void AAdapt::TaylorGreenVortex::compute(double* x, const double* X) {
  x[0] = 1.0; //initial density
  x[1] = -cos(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]); //initial u-velocity
  x[2] = sin(2.0 * pi * X[0]) * cos(2.0 * pi * X[1]); //initial v-velocity
  x[3] = cos(2.0 * pi * X[0]) + cos(2.0 * pi * X[1]); //initial temperature
}
//*****************************************************************************
AAdapt::AcousticWave::AcousticWave(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq > 3) || (numDim > 2) || (data.size() != 3),
                             std::logic_error,
                             "Error! Invalid call of AcousticWave with " << neq
                             << " " << numDim << "  " << data.size() << std::endl);
}
void AAdapt::AcousticWave::compute(double* x, const double* X) {
  const double U0 = data[0];
  const double n = data[1];
  const double L = data[2];
  x[0] = U0 * cos(n * X[0] / L);

  for(int i = 1; i < numDim; i++)
    x[i] = 0.0;
}

//*****************************************************************************
AAdapt::AerasScharDensity::AerasScharDensity(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim > 2),
                             std::logic_error,
                             "Error! Invalid call of Aeras Schar Density with " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasScharDensity::compute(double* x, const double* X) {
  //const double U0 = data[0];
  const double myPi = Aeras::ShallowWaterConstants::self().pi;

  double r = sqrt ( std::pow((X[0] - 100.0)/25.0 ,2) +  std::pow((X[1] - 9.0)/3.0,2));
  if (r <= 1.0) x[0] = std::pow(cos(myPi*r / 2.0),2);
  else          x[0] = 0.0;
}
//*****************************************************************************
AAdapt::AerasXScalarAdvection::AerasXScalarAdvection(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim > 1),
                             std::logic_error,
                             "Error! Invalid call of Aeras X Scalar Advection " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasXScalarAdvection::compute(double* x, const double* X) {
  for (int i=0; i<neq; ++i) {
    x[i] = data[0];
  }
}
//*****************************************************************************
AAdapt::AerasXZHydrostatic::AerasXZHydrostatic(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim > 1),
                             std::logic_error,
                             "Error! Invalid call of Aeras XZ Hydrostatic Model " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasXZHydrostatic::compute(double* x, const double* X) {
  //Flattened data layout
  //x[0]                                = SP
  //x[1]             ... x[1*numLevels] = u
  //x[1*numLevels+1] ... x[2*numLevels] = T
  //x[2*numLevesl+1] ... x[3*numLevels] = q0
  //x[3*numLevesl+1] ... x[4*numLevels] = q1
  //x[4*numLevesl+1] ... x[5*numLevels] = q2
  const int numLevels  = (int) data[0];
  const int numTracers = (int) data[1];
  double SP0     = data[2];
  double U0      = data[3];
  double T0      = data[4];
  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[5+nt];
  }

  int offset = 0;
  //Surface Pressure
  x[offset++] = SP0;
  
  //Velx
  for (int i=0; i<numLevels; ++i) {
     x[offset++] = U0;// + i;
     x[offset++] = T0;
  }

  //Tracers
  for (int nt=0; nt<numTracers; ++nt) {
    for (int i=0; i<numLevels; ++i) {
      x[offset++] = q0[nt];
    }
  }

}
//*****************************************************************************
AAdapt::AerasHydrostatic::AerasHydrostatic(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim != 3),
                             std::logic_error,
                             "Error! Invalid call of Aeras XZ Hydrostatic Model " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasHydrostatic::compute(double* x, const double* X) {
  //Flattened data layout
  //x[0]                                = SP
  //x[1]             ... x[1*numLevels] = u
  //x[1*numLevels+1] ... x[2*numLevels] = T
  //x[2*numLevesl+1] ... x[3*numLevels] = q0
  //x[3*numLevesl+1] ... x[4*numLevels] = q1
  //x[4*numLevesl+1] ... x[5*numLevels] = q2
  int numLevels  = (int) data[0];
  int numTracers = (int) data[1];
  double SP0     = data[2];
  double U0      = data[3];
  double T0      = data[4];
  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[5+nt];
  }

  int offset = 0;
  //Surface Pressure
  x[offset++] = SP0;
  
  //Velx
  for (int i=0; i<numLevels; ++i) x[offset++] = U0;// + i;
  for (int i=0; i<numLevels; ++i) x[offset++] = T0;

  //Tracers
  for (int nt=0; nt<numTracers; ++nt) {
    for (int i=0; i<numLevels; ++i) {
      x[offset++] = q0[nt];
    }
  }

}
//*****************************************************************************
AAdapt::AerasHeaviside::AerasHeaviside(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((neq != 3) || (numDim != 2),
                             std::logic_error,
                             "Error! Invalid call of Aeras Heaviside with " << neq
                             << " " << numDim <<  std::endl);
}
void AAdapt::AerasHeaviside::compute(double* x, const double* X) {
  //const double U0 = data[0];
  if (X[0] <= 0.5) x[0] = 1.1;
  else             x[0] = 1.0;
  x[1]=0.0;
  x[2]=0.0;
}
//*****************************************************************************
AAdapt::AerasCosineBell::AerasCosineBell(int neq_, int spatialDim_, Teuchos::Array<double> data_)
  : spatialDim(spatialDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || spatialDim!=3 || data.size()!=1) ,
                             std::logic_error,
                             "Error! Invalid call of Aeras CosineBell with " << neq
                             << " " << spatialDim <<  " "<< data.size()<< std::endl);


}
void AAdapt::AerasCosineBell::compute(double* solution, const double* X) {

  const double a = Aeras::ShallowWaterConstants::self().earthRadius;
  const double cosAlpha = std::cos(data[0]);
  const double sinAlpha = std::sin(data[0]);

  const double myPi = Aeras::ShallowWaterConstants::self().pi;
  const double u0 = 2*myPi*a/(12.*24.*3600.);
  const double lambda_c = 1.5*myPi;
  const double theta_c = 0;
  const double sinTheta_c = std::sin(theta_c);
  const double cosTheta_c = std::cos(theta_c);

  const double x = X[0];
  const double y = X[1];
  const double z = X[2];

  const double theta  = std::asin(z);

  double lambda = std::atan2(y,x);

  const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
  if (std::abs(std::abs(theta)-myPi/2.) < DIST_THRESHOLD) lambda = 0;
  else if (lambda < 0) lambda += 2*myPi;

  const double sinTheta = std::sin(theta);
  const double cosTheta = std::cos(theta);

  const double sinLambda = std::sin(lambda);
  const double cosLambda = std::cos(lambda);


  const double u = u0*(cosTheta*cosAlpha + sinTheta*cosLambda*sinAlpha);
  const double v = -u0*(sinLambda*sinAlpha);

  const double R = a/3.;
  const double h0 = 1000.;

  const double r = a*std::acos(sinTheta_c*sinTheta + cosTheta_c*cosTheta*std::cos(lambda - lambda_c));

  const double h = r < R ? 0.5*h0*(1 + std::cos(myPi*r/R)) : 0;

  solution[0] = h;
  solution[1] = u;
  solution[2] = v;
}
//*****************************************************************************
//IK, 2/5/14: added to data array h0*g, which corresponds to data[2] 
AAdapt::AerasZonalFlow::AerasZonalFlow(int neq_, int spatialDim_, Teuchos::Array<double> data_)
  : spatialDim(spatialDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || spatialDim!=3 || data.size()!=1) ,
                             std::logic_error,
                             "Error! Invalid call of Aeras ZonalFlow with " << neq
                             << " " << spatialDim <<  " "<< data.size()<< std::endl);

}
void AAdapt::AerasZonalFlow::compute(double* solution, const double* X) {

  const double myPi = Aeras::ShallowWaterConstants::self().pi;
  const double gravity = Aeras::ShallowWaterConstants::self().gravity;

  const double Omega = 2.0*myPi/(24.*3600.);
  const double a = Aeras::ShallowWaterConstants::self().earthRadius;

  const double u0 = 2.*myPi*a/(12*24*3600.);  // magnitude of wind
  const double h0g = data[0];

  const double alpha = 0; /* must match value in ShallowWaterResidDef
                             don't know how to get data from input into this class and that one. */

  const double cosAlpha = std::cos(alpha);
  const double sinAlpha = std::sin(alpha);

  const double x = X[0];  //assume that the mesh has unit radius
  const double y = X[1];
  const double z = X[2];

  const double theta  = std::asin(z);

  double lambda = std::atan2(y,x);

  static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
  if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0;
  else if (lambda < 0) lambda += 2*myPi;

  const double sinTheta = std::sin(theta);
  const double cosTheta = std::cos(theta);

  const double sinLambda = std::sin(lambda);
  const double cosLambda = std::cos(lambda);

  const double u = u0*(cosTheta*cosAlpha + sinTheta*cosLambda*sinAlpha);
  const double v = -u0*(sinLambda*sinAlpha);

  const double h0     = h0g/gravity;

  const double h = h0 - 1.0/gravity * (a*Omega*u0 + u0*u0/2.0)*(-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha)*
      (-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha);

  solution[0] = h;
  solution[1] = u;
  solution[2] = v;
}

AAdapt::AerasTC5Init::AerasTC5Init(int neq_, int spatialDim_, Teuchos::Array<double> data_)
  : spatialDim(spatialDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || spatialDim!=3 || data.size()!=0) ,
                             std::logic_error,
                             "Error! Invalid call of Aeras ZonalFlow with " << neq
                             << " " << spatialDim <<  " "<< data.size()<< std::endl);


}
void AAdapt::AerasTC5Init::compute(double* solution, const double* X) {

  const double gravity = Aeras::ShallowWaterConstants::self().gravity;
  const double myPi = Aeras::ShallowWaterConstants::self().pi;

  const double Omega = 2.0*myPi/(24.*3600.);
  const double u0 = 20.;  // magnitude of wind

  const double alpha = 0; /* must match value in ShallowWaterResidDef
                             don't know how to get data from input into this class and that one. */

  const double cosAlpha = std::cos(alpha);  //alpha
  const double sinAlpha = std::sin(alpha);

  const double x = X[0];  //assume that the mesh has unit radius
  const double y = X[1];
  const double z = X[2];

  const double theta  = std::asin(z);

  double lambda = std::atan2(y,x);

  static const double DIST_THRESHOLD = 1.0e-9;
  if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0;
  else if (lambda < 0) lambda += 2*myPi;

  const double sinTheta = std::sin(theta);
  const double cosTheta = std::cos(theta);

  const double sinLambda = std::sin(lambda);
  const double cosLambda = std::cos(lambda);


  const double u = u0*(cosTheta*cosAlpha + sinTheta*cosLambda*sinAlpha);
  const double v = -u0*(sinLambda*sinAlpha);

  const double a      = Aeras::ShallowWaterConstants::self().earthRadius;
  const double h0     =  5960.;

  const double R = myPi/9.0;
  const double lambdac = 1.5*myPi;
  const double thetac = myPi/6.0;
  const double hs0 = 2000; //meters are units
  const double radius2 = (lambda-lambdac)*(lambda-lambdac) + (theta-thetac)*(theta-thetac);
      //r^2 = min(R^2, (lambda-lambdac)^2 + (theta-thetac)^2);
  double r;
  if (radius2 > R*R) r = R;
  else r = sqrt(radius2);
  //hs = hs0*(1-r/R) for test case 5
  const double mountainHeight  = hs0*(1.0-r/R);

  const double h = h0 - 1.0/gravity * (a*Omega*u0 + u0*u0/2.0)*(-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha)*
      (-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha) - mountainHeight;

  solution[0] = h;
  solution[1] = u;
  solution[2] = v;
}

//*****************************************************************************
AAdapt::AerasPlanarCosineBell::AerasPlanarCosineBell(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || numDim!=2 || data.size()!=3)  ,
                             std::logic_error,
                             "Error! Invalid call of Aeras PlanarCosineBell with " << neq
                             << " " << numDim <<  " "<< data.size()<< std::endl);
}
void AAdapt::AerasPlanarCosineBell::compute(double* solution, const double* X) {
  const double u0 = data[0];  // magnitude of wind
  const double h0 = data[1];
  const double R = data[2];

  const double myPi = Aeras::ShallowWaterConstants::self().pi;

  const double x = X[0];
  const double y = X[1];
  const double z = X[2];

  const double u = u0;
  const double v = 0;


  const double r = std::sqrt( (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5));

  const double h = r < R ? 1 + 0.5*h0*(1 + std::cos(myPi*r/R)) : 1;

  solution[0] = h;
  solution[1] = u;
  solution[2] = v;

}
//*****************************************************************************
AAdapt::AerasRossbyHaurwitzWave::AerasRossbyHaurwitzWave(int neq_, int spatialDim_, Teuchos::Array<double> data_)
  : spatialDim(spatialDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || spatialDim!=3 || data.size()!=1) ,
                             std::logic_error,
                             "Error! Invalid call of Aeras RossbyHaurwitzWave with " << neq
                             << " " << spatialDim <<  " "<< data.size()<< std::endl);
}
void AAdapt::AerasRossbyHaurwitzWave::compute(double* solution, const double* X) {

  // Problem constants

  const double gravity = Aeras::ShallowWaterConstants::self().gravity;
  const double myPi = Aeras::ShallowWaterConstants::self().pi;

  const double Omega = 2.0*myPi/(24.*3600.);

  const double a     = Aeras::ShallowWaterConstants::self().earthRadius;
  const double aSq   = a * a;     // Radius of earth squared

  // User-supplied data
  const double omega = 7.848e-6;
  const double K     =  omega;
  const int    R     = data[0];
  const double h0    = 8000.0;

  // Computed constants
  const double KSq = K * K;
  const int    RSq = R * R;

  // Coordinates
  const double x = X[0];
  const double y = X[1];
  const double z = X[2];
  const double theta  = std::asin(z);
  double lambda = std::atan2(y,x);

  static const double DIST_THRESHOLD = 1.0e-9;
  if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0;
  else if (lambda < 0) lambda += 2*myPi;

  // Trigonometric forms of latitude and longitude
  const double sinRLambda = std::sin(R*lambda);
  const double cosRLambda = std::cos(R*lambda);
  const double sinTheta   = z;
  const double cosTheta   = std::cos(theta);
  const double cosSqTheta = cosTheta * cosTheta;

  // Initial velocities
  const double u = a * omega * cosTheta + a * K * std::pow(cosTheta,R-1) *
    (R * sinTheta*sinTheta - cosSqTheta) * cosRLambda;
  const double v = -a*K*R * std::pow(cosTheta,R-1) * sinTheta * sinRLambda;

  // Latitudinal constants for computing h
  const double A = 0.5 * omega * (2*Omega + omega) * cosSqTheta + 0.25 * KSq *
                   std::pow(cosTheta,2*R) * ((R+1) * cosSqTheta +
                   (2*RSq - R - 2) - 2 * RSq * std::pow(cosTheta,-2));

  const double B = (2 * (Omega + omega) * K * std::pow(cosTheta,R) *
                    ((RSq + 2*R + 2) - (R+1) * (R+1) * cosSqTheta)) /
                   ((R+1)*(R+2));

  const double C = 0.25 * KSq * std::pow(cosTheta,2*R) *
                   ((R+1) * cosSqTheta - (R+2));

  // Height field
  const double h = h0 +
                   (aSq*A + aSq*B*cosRLambda + aSq*C*std::cos(2*R*lambda)) / gravity;

  // Assign the solution
  solution[0] = h;
  solution[1] = u;
  solution[2] = v;
}
