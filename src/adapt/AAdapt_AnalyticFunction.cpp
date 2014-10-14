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

  else if(name == "Aeras XZ Hydrostatic Gaussian Ball")
    F = Teuchos::rcp(new AAdapt::AerasXZHydrostaticGaussianBall(neq, numDim, data));

  else if(name == "Aeras XZ Hydrostatic Gaussian Ball In Shear")
    F = Teuchos::rcp(new AAdapt::AerasXZHydrostaticGaussianBallInShear(neq, numDim, data));

  else if(name == "Aeras XZ Hydrostatic Gaussian Velocity Bubble")
    F = Teuchos::rcp(new AAdapt::AerasXZHydrostaticGaussianVelocityBubble(neq, numDim, data));

  else if(name == "Aeras XZ Hydrostatic Cloud")
    F = Teuchos::rcp(new AAdapt::AerasXZHydrostaticCloud(neq, numDim, data));

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
    
  else if(name == "Aeras TC3Init")
    F = Teuchos::rcp(new AAdapt::AerasTC3Init(neq, numDim, data));
    
  else if(name == "Aeras TCGalewskyInit")
      F = Teuchos::rcp(new AAdapt::AerasTCGalewskyInit(neq, numDim, data));
 
  else if(name == "Aeras TC4Init")
    F = Teuchos::rcp(new AAdapt::AerasTC4Init(neq, numDim, data));
  
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
  const int numLevels  = (int) data[0];
  const int numTracers = (int) data[1];
  const double SP0     = data[2];
  const double U0      = data[3];
  const double T0      = data[4];
  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[5+nt];
  }

  int offset = 0;
  //Surface Pressure
  x[offset++] = SP0;
  
  //Velx
  for (int i=0; i<numLevels; ++i) {
     x[offset++] = U0;
     x[offset++] = T0;
  }

  //Tracers
  for (int i=0; i<numLevels; ++i) {
    for (int nt=0; nt<numTracers; ++nt) {
      x[offset++] = q0[nt];
    }
  }

}

//*****************************************************************************
AAdapt::AerasXZHydrostaticGaussianBall::AerasXZHydrostaticGaussianBall(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim > 1),
                             std::logic_error,
                             "Error! Invalid call of Aeras XZ Hydrostatic Gaussian Ball Model " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasXZHydrostaticGaussianBall::compute(double* x, const double* X) {
  const int numLevels  = (int) data[0];
  const int numTracers = (int) data[1];
  const double SP0     =       data[2];
  const double U0      =       data[3];
  const double T0      =       data[4];
  const double amp     =       data[5];
  const double x0      =       data[6];
  const double z0      =       data[7];
  const double sig_x   =       data[8];
  const double sig_z   =       data[9];
  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[10+nt];
  }

  int offset = 0;
  //Surface Pressure
  x[offset++] = SP0;
  
  //Velx
  for (int i=0; i<numLevels; ++i) {
     x[offset++] = U0;
     x[offset++] = T0;
  }

  //Tracers
  for (int i=0; i<numLevels; ++i) {
    for (int nt=0; nt<numTracers; ++nt) {
      x[offset++] = q0[nt] + amp*std::exp( -( ((i-z0)*(i-z0)/(sig_z*sig_z)) + ((X[0]-x0)*(X[0]-x0)/(sig_x*sig_x)) ) )  ;
    }
  }

}

//*****************************************************************************
AAdapt::AerasXZHydrostaticGaussianBallInShear::AerasXZHydrostaticGaussianBallInShear(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim > 1),
                             std::logic_error,
                             "Error! Invalid call of Aeras XZ Hydrostatic Gaussian Ball Model " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasXZHydrostaticGaussianBallInShear::compute(double* x, const double* X) {
  const int numLevels  = (int) data[0];
  const int numTracers = (int) data[1];
  const double SP0     =       data[2];
  const double U0      =       data[3];
  const double deltaU  =       data[4];
  const double T0      =       data[5];
  const double amp     =       data[6];
  const double x0      =       data[7];
  const double z0      =       data[8];
  const double sig_x   =       data[9];
  const double sig_z   =       data[10];
  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[11+nt];
  }

  int offset = 0;
  //Surface Pressure
  x[offset++] = SP0;
  
  //Velx
  for (int i=0; i<numLevels; ++i) {
     x[offset++] = U0 + i/deltaU;
     x[offset++] = T0;
  }

  //Tracers
  for (int i=0; i<numLevels; ++i) {
    for (int nt=0; nt<numTracers; ++nt) {
      x[offset++] = q0[nt] + amp*std::exp( -( ((i-z0)*(i-z0)/(sig_z*sig_z)) + ((X[0]-x0)*(X[0]-x0)/(sig_x*sig_x)) ) )  ;
    }
  }

}

#include "Aeras_Eta.hpp"
struct DoubleType   { typedef double  ScalarT; typedef double MeshScalarT; };

//*****************************************************************************
AAdapt::AerasXZHydrostaticGaussianVelocityBubble::AerasXZHydrostaticGaussianVelocityBubble(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim > 1),
                             std::logic_error,
                             "Error! Invalid call of Aeras XZ Hydrostatic Gaussian Ball Model " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasXZHydrostaticGaussianVelocityBubble::compute(double* x, const double* X) {
  const int numLevels  = (int) data[0];
  const int numTracers = (int) data[1];
  const double P0      =       data[2];
  const double U0      =       data[3];
  const double T0      =       data[4];
  const double amp     =       data[5];
  const double x0      =       data[6];
  const double z0      =       data[7];
  const double sig_x   =       data[8];
  const double sig_z   =       data[9];
  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[10+nt];
  }

  const double Ptop = 101.325;
  const double Ps   = P0;
  const double Ttop = 235;
  const double Ts   = T0;
  const double DT   = 4.8e5;
  const double R    = 287;
  const double g    = 9.80616;
  const double Gamma= 0.005;  
  const double Dtop = 0.00385;
  const double Dbot = 1.225;
  const double EtaT = 0.2;
  const int    N    = numLevels-1;

  const Aeras::Eta<DoubleType> &EP = Aeras::Eta<DoubleType>::self(Ptop,P0,numLevels);
  const Aeras::Eta<DoubleType> ET(Ttop,T0,numLevels);


  std::vector<double> Pressure(numLevels);
  std::vector<double> Temperature(numLevels);
  std::vector<double> Pi(numLevels);
  std::vector<double> D(numLevels);
  for (int i=0; i<numLevels; ++i) Pressure[i]    = EP.A(i)*EP.p0() + EP.B(i)*Ps;

  for (int i=0; i<numLevels; ++i) {
    const double Eta =  EP.eta(i);
    Temperature[i] =  T0 * std::pow(Eta,R*Gamma/g);
    if (Eta < EtaT)Temperature[i] += DT*std::pow(EtaT-Eta, 5);
  }
  
  for (int i=0; i<numLevels; ++i) {
    const double pp   = i<N ? 0.5*(Pressure[i] + Pressure[i+1]) : Ps;
    const double pm   = i   ? 0.5*(Pressure[i] + Pressure[i-1]) : EP.ptop();
    Pi[i] = (pp - pm) /EP.delta(i);
  }
  for (int i=0; i<numLevels; ++i) {
    D[i] = Pressure[i]/(R*Temperature[i]);
  }

  int offset = 0;
  //Surface Pressure
  x[offset++] = P0;
  
  //Velx
  for (int i=0; i<numLevels; ++i) {
    x[offset++] = U0 + amp*std::exp( -( ((i-z0)*(i-z0)/(sig_z*sig_z)) + ((X[0]-x0)*(X[0]-x0)/(sig_x*sig_x)) ) )  ;
    x[offset++] = Temperature[i];
  }

  //Tracers
  for (int i=0; i<numLevels; ++i) {
    for (int nt=0; nt<numTracers; ++nt) {
      x[offset++] = q0[nt];
    }
  }

}

//*****************************************************************************
AAdapt::AerasXZHydrostaticCloud::AerasXZHydrostaticCloud(int neq_, int numDim_, Teuchos::Array<double> data_)
  : numDim(numDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION((numDim > 1),
                             std::logic_error,
                             "Error! Invalid call of Aeras XZ Hydrostatic Cloud Model " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasXZHydrostaticCloud::compute(double* x, const double* X) {
  const int numLevels  = (int) data[0];
  const int numTracers = (int) data[1];
  const double SP0     =       data[2];
  const double U0      =       data[3];
  //const double T0      =       data[4];
  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[4+nt];
  }

  const double Ptop = 101.325;
  const double P0   = 101325;
  const double Ps   = 101325;
  const double Ttop = 235;
  const double T0   = 288;
  const double Ts   = T0;
  const double DT   = 4.8e5;
  const double R    = 287;
  const double g    = 9.80616;
  const double Gamma= 0.005;  
  const double Dtop = 0.00385;
  const double Dbot = 1.225;
  const double EtaT = 0.2;
  const int    N    = numLevels-1;

  const Aeras::Eta<DoubleType> &EP = Aeras::Eta<DoubleType>::self(Ptop,P0,numLevels);

  std::vector<double> Pressure(numLevels);
  std::vector<double> Pi(numLevels);
  std::vector<double> Temperature(numLevels);
  std::vector<double> D(numLevels);
  std::vector<double> qvs(numLevels);
  std::vector<double> qv(numLevels);
  std::vector<double> qc(numLevels);
  std::vector<double> qr(numLevels);

  for (int i=0; i<numLevels; ++i) Pressure[i] = EP.A(i)*EP.p0() + EP.B(i)*Ps;
  
  for (int i=0; i<numLevels; ++i) {
    const double Eta =  EP.eta(i);
    Temperature[i] =  T0 * std::pow(Eta,R*Gamma/g);
    if (Eta < EtaT)Temperature[i] += DT*std::pow(EtaT-Eta, 5);
  }

  for (int i=0; i<numLevels; ++i) {
    const double pp   = i<numLevels-1 ? 0.5*(Pressure[i] + Pressure[i+1]) : Ps;
    const double pm   = i             ? 0.5*(Pressure[i] + Pressure[i-1]) : EP.ptop();
    Pi[i] = (pp - pm) / EP.delta(i);
  }

  for (int i=0; i<numLevels; ++i) {
    D[i] = Pressure[i]/(R*Temperature[i]);
  }

  //Compute saturation mixture ratio qvs
  double eps = 0.622;
  for (int i=0; i<numLevels; ++i) {
    double TC = Temperature[i] - 273.13;
    double Pmb = Pressure[i] / 100.0;                          // (1bar/100000Pa)*(1000mbar/1bar)
    qvs[i] = eps * 6.112 * exp( (17.67*TC)/(TC+243.5) ) / Pmb;
  }

  //Initialize qv, qc, qr
  for (int i=0; i<numLevels; ++i) {
    qv[i] = 0.1*qvs[i];
    qc[i] = q0[1];
    qr[i] = q0[2];
  } 

  static bool first_time=true;
  if (first_time) {
    for (int i=0; i<numLevels; ++i) std::cout<<__FILE__
    <<" Pressure["<<i<<"]="<<Pressure[i]
    <<" Temperature["<<i<<"]="<<Temperature[i]
    <<" Density["<<i<<"]="<<D[i]
    <<" qvs["<<i<<"]="<<qvs[i]
    <<" Pi["<<i<<"]="<<Pi[i]<<std::endl;
  }
  first_time = false;

  int offset = 0;
  //Surface Pressure
  x[offset++] = SP0;
  
  for (int i=0; i<numLevels; ++i) {
    //Velx
    x[offset++] = U0;
    //Temperature
    x[offset++] = Temperature[i];
  }

  //Vapor
  for (int i=0; i<numLevels; ++i) {
    x[offset++] = Pi[i]*qv[i];
  }

  //cloud water
  for (int i=0; i<numLevels; ++i) {
    x[offset++] = Pi[i]*qc[i];
  }

  //rain
  for (int i=0; i<numLevels; ++i) {
    x[offset++] = Pi[i]*qr[i];
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
void AAdapt::AerasHydrostatic::compute(double* solution, const double* X) {

  const int numLevels  = (int) data[0];
  const int numTracers = (int) data[1];
  const double SP0     =       data[2];
  const double U0      =       data[3];
  const double U1      =       data[4];
  const double T0      =       data[5];

  std::vector<double> q0(numTracers);
  for (int nt = 0; nt<numTracers; ++nt) {
    q0[nt] = data[6 + nt];
  }

  const double x = X[0];
  const double y = X[1];
  const double z = X[2];


  const double myPi  = pi;
  const double alpha = myPi/4;
  const double cosAlpha = std::cos(alpha);
  const double sinAlpha = std::sin(alpha);

  const double theta  = std::asin(z);
  double lambda = std::atan2(y,x);

  static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
  if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0;
  else if (lambda < 0) lambda += 2*myPi;

  const double sinTheta = std::sin(theta);
  const double cosTheta = std::cos(theta);

  const double sinLambda = std::sin(lambda);
  const double cosLambda = std::cos(lambda);

  const double u =  U1*(cosTheta*cosAlpha + sinTheta*cosLambda*sinAlpha);
  const double v = -U1*(sinLambda*sinAlpha);


  int offset = 0;
  //Surface Pressure
  solution[offset++] = SP0;
  
  for (int i=0; i<numLevels; ++i) {
    //Velx
    solution[offset++] = u; // U0*(1-z*z);
    solution[offset++] = v; // U1*(1-x*x);
    //Temperature
    solution[offset++] = T0;
  }


  //Tracers
  for (int i=0; i<numLevels; ++i) {
    for (int nt=0; nt<numTracers; ++nt) {
      const double w = nt%3 ? ((nt%3 == 1) ? y : z) : x;
      solution[offset++] = w*q0[nt];
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
//TC2
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

    const double alpha = 0.0;//1.047; /* must match value in ShallowWaterResidDef
                             //don't know how to get data from input into this class and that one. */

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



//**********************************************************
//TC5
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
//TC Galewsky

AAdapt::AerasTCGalewskyInit::AerasTCGalewskyInit(int neq_, int spatialDim_, Teuchos::Array<double> data_)
: spatialDim(spatialDim_), neq(neq_), data(data_) {
    TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || spatialDim!=3 || data.size()!=0) ,
                               std::logic_error,
                               "Error! Invalid call of Aeras TCGalewskyInit with " << neq
                               << " " << spatialDim <<  " "<< data.size()<< std::endl);
    myPi = Aeras::ShallowWaterConstants::self().pi;
    gravity = Aeras::ShallowWaterConstants::self().gravity;
    earthRadius = Aeras::ShallowWaterConstants::self().earthRadius;
    Omega = 2.0*myPi/(24.*3600.);
    
    phi0 = myPi/7.;
    phi1 = myPi/2. - phi0;
    en = exp(-4./(phi1-phi0)/(phi1-phi0));
    phi2 = myPi/4.;
  
    umax = 80.;
    h0 = 10158.2951; //note that this const was grabbed from homme
    //because otherwise global integration is needed
  
    al = 1./3.;
    beta = 1./15.;
    hhat = 120.;
  
}

double AAdapt::AerasTCGalewskyInit::ucomponent(const double lat){
    
    if( (lat <= phi0) || (lat >=phi1)){
        return 0.0;
    }else{
        return umax/en*exp(1./((lat - phi0)*(lat - phi1)));
    }
}

double AAdapt::AerasTCGalewskyInit::hperturb(const double lon, const double lat){
    
    double lon2=lon;
    if (lon2 >= myPi){
        lon2=lon - 2.*myPi;
    }
    return (hhat*cos(lat)*exp(-lon2*lon2/al/al)*exp( -((phi2-lat)/beta)*((phi2-lat)/beta) ));

}

void AAdapt::AerasTCGalewskyInit::compute(double* solution, const double* X) {
  
    const double a = earthRadius;
    
    const double x = X[0];  //assume that the mesh has unit radius
    const double y = X[1];
    const double z = X[2];
    
    //begin note 1
    const double theta  = std::asin(z); // this is a repeated code
            // should be in SW class again?
    
    double lambda = std::atan2(y,x);
    
    static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
    if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0;
    else if (lambda < 0) lambda += 2*myPi;
    //end note 1
    
    double h = h0;
  
    //integrating height numerically
    const int integration_steps = 1000;//make this a const in class
    const double deltat = (theta+myPi/2.0)/integration_steps;
    for(int i=0; i<integration_steps; i++){
        
        double midpoint1 = -myPi/2.0 + (i-1)*deltat;
        double midpoint2 = -myPi/2.0 + i*deltat;
        
        double loc_u = ucomponent(midpoint1);
        
        h -= a/gravity*deltat*(2.*Omega*sin(midpoint1)+loc_u*tan(midpoint1)/a)*loc_u/2.;
        
        loc_u = ucomponent(midpoint2);
        
        h -= a/gravity*deltat*(2.*Omega*sin(midpoint2)+loc_u*tan(midpoint2)/a)*loc_u/2.;
        
    }
    
    //now add the perturbation
    
    h += hperturb(lambda, theta);
    
    double u, v;

    u = ucomponent(theta);
    v = 0.0;
    
    solution[0] = h;
    solution[1] = u;
    solution[2] = v;

}


//*****************************************************************************
//TC4


AAdapt::AerasTC4Init::AerasTC4Init(int neq_, int spatialDim_, Teuchos::Array<double> data_)
: spatialDim(spatialDim_), neq(neq_), data(data_) {
  TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || spatialDim!=3 || data.size()!=0) ,
                             std::logic_error,
                             "Error! Invalid call of Aeras TC4Init with " << neq
                             << " " << spatialDim <<  " "<< data.size()<< std::endl);
  myPi = Aeras::ShallowWaterConstants::self().pi;
  earthRadius = Aeras::ShallowWaterConstants::self().earthRadius;
  gravity = Aeras::ShallowWaterConstants::self().gravity;
  
  Omega = 2.0*myPi/(24.*3600.); //this should be sitting in SW class
  
  rlon0 = 0.;
  rlat0 = myPi/4.;
  npwr = 14.;
  
  su0 = 20.;
  phi0 = 1.0e5;

}

void AAdapt::AerasTC4Init::compute(double* solution, const double* X) {
  
  //like SW constants
  double a = earthRadius;
  
  const double x = X[0];  //assume that the mesh has unit radius
  const double y = X[1];
  const double z = X[2];
  
  //begin note 1
  const double theta  = std::asin(z); // this is a repeated code
  // should be in SW class again?
  
  double lambda = std::atan2(y,x);
  
  static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
  if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0;
  else if (lambda < 0) lambda += 2*myPi;
  //end note 1
  //end of repeated code
  
  
  double tol = 1.e-10;
  
  alfa = -0.03*(phi0/(2.*Omega*sin(myPi/4.)));
  sigma = (2.*a/1.0e6)*(2.*a/1.0e6);
  
  
  double ai = 1./a;
  double a2i = ai*ai;
  
  double snj = sin(theta);
  double csj = cos(theta)*cos(theta);
  double srcsj = cos(theta);
  double tmpry = tan(theta);
  double tmpry2 = tmpry*tmpry;
  double den = 1./cos(theta);
  double aacsji = 1./(a*a*csj);
  double corr = 2.*Omega*snj;
  
  
  double ucon = bubfnc(theta);
  double bigubr = ucon*srcsj; ///
  double dbub = dbubf(theta); ///
  
  double c = sin(rlat0)*snj + cos(rlat0)*srcsj*cos(lambda - rlon0);
  
  //if-statements about ~fabs(c+1) is due to singularities ~1/(c+1)
  //in derivatives. However, they are overtaken by the presence of
  //multipliers ~exp(-1/(c+1)).
  double psib = 0.;
  if(fabs(c+1.)>tol)
     psib = alfa*exp(-sigma*((1.-c)/(1.+c)));
  
  double dcdm = sin(rlat0)-cos(lambda-rlon0)*cos(rlat0)*tmpry;
  double dcdl = -cos(rlat0)*srcsj*sin(lambda-rlon0);
  double d2cdm = -cos(rlat0)*cos(lambda-rlon0)*(1.+tmpry2)/srcsj;
  double d2cdl = -cos(rlat0)*srcsj*cos(lambda-rlon0);
  
  double tmp1 = 0.;
  if(fabs(c+1.)>tol)
    tmp1 = 2.*sigma*psib/((1.+c)*(1.+c));
  double tmp2 = 0.;
  if(fabs(c+1.)>tol)
    tmp2 = (sigma - (1.0+c))/((1.+c)*(1.+c));
  double dkdm = tmp1*dcdm;
  double dkdl = tmp1*dcdl;
  
  double d2kdm  = tmp1*(d2cdm + 2.0*(dcdm*dcdm)*tmp2);
  double d2kdl  = tmp1*(d2cdl + 2.0*(dcdl*dcdl)*tmp2);
  
  double u, v, h;
  
  u = bigubr*den - srcsj*ai*dkdm;
  v = (dkdl*ai)*den;
  h = phicon(theta)+corr*psib/gravity;
  
  solution[0] = h;
  solution[1] = u;
  solution[2] = v;
  
}


double AAdapt::AerasTC4Init::dbubf(const double lat){
  double rmu = sin(lat);
  double coslat = cos(lat);
  return 2.*su0*std::pow(2.*rmu*coslat,npwr-1.)
           *(npwr-(2.*npwr+1)*rmu*rmu);
}

double AAdapt::AerasTC4Init::bubfnc(const double lat){
  return su0*std::pow((2.*sin(lat)*cos(lat)), npwr);
}

double AAdapt::AerasTC4Init::phicon(const double lat){
  
  double a = earthRadius;
  
  const int integration_steps = 1000;
  
  double h = 0.;
  
  const double deltat = (lat+myPi/2.0)/integration_steps;
  for(int i=0; i<integration_steps; i++){
    
    double midpoint1 = -myPi/2.0 + (i-1)*deltat;
    double midpoint2 = -myPi/2.0 + i*deltat;
    
    double loc_u = bubfnc(midpoint1);
    
    h -= a*deltat*(2*Omega*sin(midpoint1)+loc_u*tan(midpoint1)/a)*loc_u/2.;
    
    loc_u = bubfnc(midpoint2);
    
    h -= a*deltat*(2*Omega*sin(midpoint2)+loc_u*tan(midpoint2)/a)*loc_u/2.;
    
  }

  h = (phi0 + h)/gravity;
  
  return h;
  
}


//*****************************************************************************
//almost identical to TC2, AerasZonalFlow

AAdapt::AerasTC3Init::AerasTC3Init(int neq_, int spatialDim_, Teuchos::Array<double> data_)
: spatialDim(spatialDim_), neq(neq_), data(data_) {
    TEUCHOS_TEST_FOR_EXCEPTION( (neq!=3 || spatialDim!=3 || data.size()!=1) ,
                               std::logic_error,
                               "Error! Invalid call of Aeras TC3Init with " << neq
                               << " " << spatialDim <<  " "<< data.size()<< std::endl);
    myPi = Aeras::ShallowWaterConstants::self().pi;

    earthRadius = Aeras::ShallowWaterConstants::self().earthRadius;
    
    testDuration = 12*24*3600.;
    
    u0 = 2.*myPi*earthRadius/testDuration;  // magnitude of wind
    
    Omega = 2.0*myPi/(24.*3600.);
  
    gravity = Aeras::ShallowWaterConstants::self().gravity;
  
    h0g = 2.94e04;
  
    xe = 0.3;
  
    thetae = myPi/2.0;
  
    thetab = -myPi/6.0;
}

double AAdapt::AerasTC3Init::ucomponent(const double lon){
  
    double xx = xe*(lon - thetab)/(thetae-thetab);
    
    double res = u0*exp(4.0/xe)*bx(xx)*bx(xe-xx);
    
    return res;
}

double AAdapt::AerasTC3Init::bx(const double x){
    double res;
    (x > 0) ? (res = exp(-1./x)): (res = 0.);
    return res;
}

void AAdapt::AerasTC3Init::rotate(const double lon, const double lat,
                                  const double alpha, double& rotlon, double& rotlat){

    if(alpha == 0.0){
        rotlon = lon;
        rotlat = lat;
    }else{
        double test = sin(lat)*cos(alpha) - cos(lat)*cos(lon)*sin(alpha);
        
        if(test > 1.0){
            rotlat = myPi/2.;
        }else{
            if (test < -1.0) {
                rotlat = -myPi/2.;
            }else{
                rotlat = asin(test);
            }
        }
      
        test = cos(rotlat);
        if (test == 0.0) {
            rotlon = 0.0;
        }else{
            test = sin(lon)*cos(lat)/test;
            if (test > 1.0) {
                rotlon = myPi/2.;
            }else{
                if (test < -1.0) {
                    rotlon = -myPi/2.0;
                }else{
                    rotlon = asin(test);
                }
            }
        }
        
        test = cos(alpha)*cos(lon)*cos(lat) + sin(alpha)*sin(lat);
        if (test < 0.0) {
            rotlon = myPi - rotlon;
        }
    }

}


void AAdapt::AerasTC3Init::compute(double* solution, const double* X) {
  
    const double a = earthRadius;

    const double alpha = data[0];//1.047; /* must match value in ShallowWaterResidDef
                             //don't know how to get data from input into this class and that one. */
  
    const double x = X[0];  //assume that the mesh has unit radius
    const double y = X[1];
    const double z = X[2];
  
    //repeated code
    const double theta  = std::asin(z);
    
    double lambda = std::atan2(y,x);
    
    static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
    if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0;
    else if (lambda < 0) lambda += 2*myPi;
    //end of repeated code
  
    double h = h0g/gravity;
    
    double rotlambda, rottheta;
    
    rotate(lambda, theta, alpha, rotlambda, rottheta);
    
    //integrating height numerically
    const int integration_steps = 1000;
    const double deltat = (rottheta+myPi/2.0)/integration_steps;
    for(int i=0; i<integration_steps; i++){
        double midpoint1 = -myPi/2.0 + (i-1)*deltat;
        double midpoint2 = -myPi/2.0 + i*deltat;
        double loc_u = ucomponent(midpoint1);
        h -= a/gravity*deltat*(2*Omega*sin(midpoint1)+loc_u*tan(midpoint1)/a)*loc_u/2.;
        loc_u = ucomponent(midpoint2);
        h -= a/gravity*deltat*(2*Omega*sin(midpoint2)+loc_u*tan(midpoint2)/a)*loc_u/2.;
    }
    
    double u, v;
    
    if (alpha == 0.0) {
        u = ucomponent(theta);
        v = 0.0;
    }else{
        u = ucomponent(rottheta)*(cos(alpha)*sin(rotlambda)*sin(lambda)
                                +cos(lambda)*cos(rotlambda));
        v = ucomponent(rottheta)*(cos(alpha)*cos(lambda)*sin(rotlambda)*sin(theta)
                                -cos(rotlambda)*sin(lambda)*sin(theta)
                                -sin(alpha)*sin(rotlambda)*cos(theta));
    }
    
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
//TC6
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
