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
  TEUCHOS_TEST_FOR_EXCEPTION((neq > 1) || (numDim > 2),
                             std::logic_error,
                             "Error! Invalid call of Aeras Schar Density with " << neq
                             << " " << numDim << std::endl);
}
void AAdapt::AerasScharDensity::compute(double* x, const double* X) {
  //const double U0 = data[0];
  double r = sqrt ( std::pow((X[0] - 100.0)/25.0 ,2) +  std::pow((X[1] - 9.0)/3.0,2));
  if (r <= 1.0) x[0] = std::pow(cos(pi*r / 2.0),2);
  else          x[0] = 0.0;
}
//*****************************************************************************
