/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <cmath>

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
  else
    TEST_FOR_EXCEPTION(name != "Valid Initial Condition Function",
                       std::logic_error,
                       "Unrecognized initial condition function name: " << name);
  return F;
};


//*****************************************************************************
Albany::ConstantFunction::ConstantFunction(int neq_, int numDim_,
   Teuchos::Array<double> data_) : neq(neq_), numDim(numDim_), data(data_)
{
  if (data.size()>0) val=data[0];
  else val = 0.0;
}
void Albany::ConstantFunction::compute(double* x, const double *X) 
{
  for (int i=0; i<neq; i++) x[i]=val;
}

//*****************************************************************************
Albany::GaussSin::GaussSin(int neq_, int numDim_, Teuchos::Array<double> data_)
 : neq(neq_), numDim(numDim_), data(data_)
{
  TEST_FOR_EXCEPTION((neq!=1) || (numDim!=1) || (data.size()!=1),
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
 : neq(neq_), numDim(numDim_), data(data_)
{
  TEST_FOR_EXCEPTION((neq!=1) || (numDim!=1) || (data.size()!=1),
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
 : neq(neq_), numDim(numDim_), data(data_)
{
  TEST_FOR_EXCEPTION((neq<2) || (numDim<2) || (data.size()!=1),
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
