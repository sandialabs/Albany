//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <cmath>
#include <ctime>
#include <cstdlib>

#include "Albany_AnalyticFunction.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Exceptions.hpp"

const double pi=3.141592653589793;


// Factory method to build functions based on a string
Teuchos::RCP<Albany::AnalyticFunction> Albany::createAnalyticFunction(
   std::string name, int neq, int numDim,
   Teuchos::Array<double> data)
{
  Teuchos::RCP<Albany::AnalyticFunction> F;

  if (name=="Constant")
    F = Teuchos::rcp(new Albany::ConstantFunction(neq, numDim, data));
  else if (name=="1D Gauss-Sin")
    F = Teuchos::rcp(new Albany::GaussSin(neq, numDim, data));
  else if (name=="1D Gauss-Cos")
    F = Teuchos::rcp(new Albany::GaussCos(neq, numDim, data));
  else if (name=="Linear Y")
    F = Teuchos::rcp(new Albany::LinearY(neq, numDim, data));
  else if (name=="Gaussian Pressure")
    F = Teuchos::rcp(new Albany::GaussianPress(neq, numDim, data));
  else
    TEUCHOS_TEST_FOR_EXCEPTION(name != "Valid Initial Condition Function",
                       std::logic_error,
                       "Unrecognized initial condition function name: " << name);
  return F;
}


//*****************************************************************************
Albany::ConstantFunction::ConstantFunction(int neq_, int numDim_,
   Teuchos::Array<double> data_) : numDim(numDim_), neq(neq_), data(data_)
{
   TEUCHOS_TEST_FOR_EXCEPTION((data.size()!=neq),
                             std::logic_error,
                             "Error! Invalid specification of initial condition: incorrect length of Function Data for Constant Function; neq = " << neq << ", data.size() = " << data.size() <<  std::endl) ;
}
void Albany::ConstantFunction::compute(double* x, const double *X) 
{
  if (data.size()>0)
    for (int i=0; i<neq; i++) 
      x[i]=data[i];
}

//*****************************************************************************
// Private convenience function
long Albany::seedgen(int worksetID)
{
  long seconds, s, seed, pid;

    pid = getpid();
    s = time ( &seconds ); /* get CPU seconds since 01/01/1970 */

    // Use worksetID to give more randomness between calls

    seed = abs(((s*181)*((pid-83)*359)*worksetID)%104729); 
    return seed;
}

Albany::ConstantFunctionPerturbed::ConstantFunctionPerturbed(int neq_, int numDim_,
   int worksetID, 
   Teuchos::Array<double> data_,  Teuchos::Array<double> pert_mag_) 
    : numDim(numDim_), neq(neq_), data(data_), pert_mag(pert_mag_)
{

   TEUCHOS_TEST_FOR_EXCEPTION((data.size()!=neq || pert_mag.size()!=neq),
                             std::logic_error,
                             "Error! Invalid specification of initial condition: incorrect length of " <<
                             "Function Data for Constant Function Perturbed; neq = " << neq << 
                             ", data.size() = " << data.size() 
                             << ", pert_mag.size() = " << pert_mag.size()
                             <<  std::endl) ;

//  srand( time(NULL) ); // seed the random number gen
  srand(seedgen(worksetID)); // seed the random number gen

}

void Albany::ConstantFunctionPerturbed::compute(double* x, const double *X) 
{
    for (int i=0; i<neq; i++) 
      x[i]=data[i] + udrand(-pert_mag[i], pert_mag[i]);
}

// Private convenience function
double Albany::ConstantFunctionPerturbed::udrand ( double lo, double hi )
{
  static const double base = 1.0 / ( RAND_MAX + 1.0 );
  double deviate = std::rand() * base;
  return lo + deviate * ( hi - lo );
}
//*****************************************************************************
Albany::ConstantFunctionGaussianPerturbed::ConstantFunctionGaussianPerturbed(int neq_, int numDim_,
   int worksetID,
   Teuchos::Array<double> data_,  Teuchos::Array<double> pert_mag_) 
    : numDim(numDim_), 
      neq(neq_), 
      data(data_), 
      pert_mag(pert_mag_),
//      rng(boost::random::random_device()()), // seed the rng
      rng(seedgen(worksetID)), // seed the rng
      nd(neq_),
      var_nor(neq_)
{


   TEUCHOS_TEST_FOR_EXCEPTION((data.size()!=neq || pert_mag.size()!=neq),
                             std::logic_error,
                             "Error! Invalid specification of initial condition: incorrect length of " <<
                             "Function Data for Constant Function Gaussian Perturbed; neq = " << neq << 
                             ", data.size() = " << data.size() 
                             << ", pert_mag.size() = " << pert_mag.size()
                             <<  std::endl) ;

  if (data.size()>0 && pert_mag.size()>0)
    for(int i = 0; i < neq; i++)
      if(pert_mag[i] > std::numeric_limits<double>::epsilon()){

        nd[i] = Teuchos::rcp(new boost::normal_distribution<double>(data[i], pert_mag[i]));
        var_nor[i] = Teuchos::rcp(new 
          boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(rng, *nd[i]));

      }

}

void Albany::ConstantFunctionGaussianPerturbed::compute(double* x, const double *X) 
{

    for (int i=0; i<neq; i++) 
      if(var_nor[i] != Teuchos::null)
        x[i] = (*var_nor[i])();
      else
        x[i] = data[i];

}


//*****************************************************************************
Albany::GaussSin::GaussSin(int neq_, int numDim_, Teuchos::Array<double> data_)
 : numDim(numDim_), neq(neq_), data(data_)
{
  TEUCHOS_TEST_FOR_EXCEPTION((neq!=1) || (numDim!=1) || (data.size()!=1),
			     std::logic_error,
			     "Error! Invalid call of GaussSin with " <<neq
			     <<" "<< numDim <<"  "<< data.size() << std::endl);
}
void Albany::GaussSin::compute(double* x, const double *X) 
{
  x[0] =     sin(pi * X[0]) + 0.5*data[0]*X[0]*(1.0-X[0]);
}

//*****************************************************************************
Albany::GaussCos::GaussCos(int neq_, int numDim_, Teuchos::Array<double> data_)
 : numDim(numDim_), neq(neq_), data(data_)
{
  TEUCHOS_TEST_FOR_EXCEPTION((neq!=1) || (numDim!=1) || (data.size()!=1),
			     std::logic_error,
			     "Error! Invalid call of GaussCos with " <<neq
			     <<" "<< numDim <<"  "<< data.size() << std::endl);
}
void Albany::GaussCos::compute(double* x, const double *X) 
{
  x[0] = 1 + cos(2*pi * X[0]) + 0.5*data[0]*X[0]*(1.0-X[0]);
}
//*****************************************************************************
Albany::LinearY::LinearY(int neq_, int numDim_, Teuchos::Array<double> data_)
 : numDim(numDim_), neq(neq_), data(data_)
{
  TEUCHOS_TEST_FOR_EXCEPTION((neq<2) || (numDim<2) || (data.size()!=1),
			     std::logic_error,
			     "Error! Invalid call of LinearY with " <<neq
			     <<" "<< numDim <<"  "<< data.size() << std::endl);
}
void Albany::LinearY::compute(double* x, const double *X) 
{
  x[0] = 0.0;
  x[1] = data[0] * X[0];
  if (numDim>2) x[2]=0.0;
}
//*****************************************************************************
Albany::GaussianPress::GaussianPress(int neq_, int numDim_, Teuchos::Array<double> data_)
 : numDim(numDim_), neq(neq_), data(data_)
{
  //for LinComprNS problems; 
  //Sets an initial condition as a Gaussian pressure pulse 
  std::cout << "setting gaussian pressure IC!" << std::endl; 
  TEUCHOS_TEST_FOR_EXCEPTION((neq<3) || (numDim<2) || (data.size()!=4),
			     std::logic_error,
			     "Error! Invalid call of GaussianPress with " <<neq
			     <<" "<< numDim <<"  "<< data.size() << std::endl);
}
void Albany::GaussianPress::compute(double* x, const double *X) 
{
  x[0] = 0.0;
  x[1] = 0.0;
  x[2] = data[0]*exp(-data[1]*((X[0] - data[2])*(X[0] - data[2]) + (X[1] - data[3])*(X[1] - data[3]))); 
}
//*****************************************************************************
