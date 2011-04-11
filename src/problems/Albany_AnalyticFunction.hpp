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


#ifndef ALBANY_ANALYTICFUNCTION_HPP
#define ALBANY_ANALYTICFUNCTION_HPP

#include <string>

#include "Teuchos_Array.hpp"

namespace Albany {

// Base class for initial condition functions
class AnalyticFunction {
  public:
    virtual ~AnalyticFunction(){}
    virtual void compute(double* x, const double *X) = 0;
};

// Factory method to build functions based on a string name
Teuchos::RCP<AnalyticFunction> createAnalyticFunction(
   std::string name, int neq, int numDim,
   Teuchos::Array<double> data);

// Below is a library of intial condition functions

class ConstantFunction : public AnalyticFunction {
  public:
    ConstantFunction(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double *X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;  
    double val;
};

class GaussSin : public AnalyticFunction {
  public:
    GaussSin(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double *X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;  
};

class GaussCos : public AnalyticFunction {
  public:
    GaussCos(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double *X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;  
};

class LinearY : public AnalyticFunction {
  public:
    LinearY(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double *X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;  
};


}

#endif
