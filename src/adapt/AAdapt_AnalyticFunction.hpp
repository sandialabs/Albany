//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_ANALYTICFUNCTION_HPP
#define AADAPT_ANALYTICFUNCTION_HPP

#include "Albany_config.h"

#include <string>

// Random and Gaussian number distribution
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include "Teuchos_Array.hpp"
#ifdef ALBANY_PAMGEN
#include "RTC_FunctionRTC.hh"
#endif

namespace AAdapt {

// generate seed convenience function
long seedgen(int worksetID);

// Base class for initial condition functions
class AnalyticFunction {
  public:
    virtual ~AnalyticFunction() {}
    virtual void compute(double* x, const double* X) = 0;
};

// Factory method to build functions based on a string name
Teuchos::RCP<AnalyticFunction> createAnalyticFunction(
  std::string name, int neq, int numDim,
  Teuchos::Array<double> data);

// Below is a library of intial condition functions

class ConstantFunction : public AnalyticFunction {
  public:
    ConstantFunction(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class StepX : public AnalyticFunction {
  public:
    StepX(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class TemperatureStep : public AnalyticFunction {
  public:
    TemperatureStep(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class DispConstTemperatureStep : public AnalyticFunction {
  public:
    DispConstTemperatureStep(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class TemperatureLinear : public AnalyticFunction {
  public:
    TemperatureLinear(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector Y
    int neq;    // size of solution vector
    Teuchos::Array<double> data;
};

class DispConstTemperatureLinear : public AnalyticFunction {
  public:
    DispConstTemperatureLinear(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector Y
    int neq;    // size of solution vector
    Teuchos::Array<double> data;
};

class ConstantFunctionPerturbed : public AnalyticFunction {
  public:
    ConstantFunctionPerturbed(int neq_, int numDim_, int worksetID,
                              Teuchos::Array<double> const_data_, Teuchos::Array<double> pert_mag_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
    Teuchos::Array<double> pert_mag;

    // random number generator convenience function
    double udrand(double lo, double hi);

};

class ConstantFunctionGaussianPerturbed : public AnalyticFunction {
  public:
    ConstantFunctionGaussianPerturbed(int neq_, int numDim_, int worksetID,
                                      Teuchos::Array<double> const_data_, Teuchos::Array<double> pert_mag_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
    Teuchos::Array<double> pert_mag;

    boost::mt19937 rng;
    Teuchos::Array<Teuchos::RCP<boost::normal_distribution<double> > > nd;
    Teuchos::Array < Teuchos::RCP < boost::variate_generator < boost::mt19937&,
            boost::normal_distribution<double> > > > var_nor;

};

class GaussSin : public AnalyticFunction {
  public:
    GaussSin(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class GaussCos : public AnalyticFunction {
  public:
    GaussCos(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class LinearY : public AnalyticFunction {
  public:
    LinearY(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class Linear : public AnalyticFunction {
  public:
    Linear(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class ConstantBox : public AnalyticFunction {
  public:
  ConstantBox(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AboutZ : public AnalyticFunction {
  public:
    AboutZ(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class RadialZ : public AnalyticFunction {
  public:
    RadialZ(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AboutLinearZ : public AnalyticFunction {
  public:
    AboutLinearZ(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class GaussianZ : public AnalyticFunction {
  public:
  GaussianZ(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class Circle : public AnalyticFunction {
  public:
    Circle(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class GaussianPress : public AnalyticFunction {
  public:
    GaussianPress(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class SinCos : public AnalyticFunction {
  public:
    SinCos(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class SinScalar : public AnalyticFunction {
  public:
    SinScalar(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class TaylorGreenVortex : public AnalyticFunction {
  public:
    TaylorGreenVortex(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AcousticWave : public AnalyticFunction {
  public:
    AcousticWave(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};


class ExpressionParser : public AnalyticFunction {
  public:
    ExpressionParser(int neq_, int spatialDim_, std::string expressionX_, std::string expressionY_, std::string expressionZ_);
    void compute(double* x, const double* X);
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    std::string expressionX;
    std::string expressionY;
    std::string expressionZ;
#ifdef ALBANY_PAMGEN
    PG_RuntimeCompiler::Function rtcFunctionX;
    PG_RuntimeCompiler::Function rtcFunctionY;
    PG_RuntimeCompiler::Function rtcFunctionZ;
#endif
};

#ifdef ALBANY_STK_EXPR_EVAL
class ExpressionParserAllDOFs : public AnalyticFunction
{
 public:
  ExpressionParserAllDOFs(int neq_, int dim_, Teuchos::Array<std::string>& expr);
  void
  compute(double* unknowns, double const* coords);

 private:
  int                         dim;  // size of coordinate vector X
  int                         neq;  // size of solution vector x
  Teuchos::Array<std::string> expr;
};
#endif
}

#endif
