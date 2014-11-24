//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_ANALYTICFUNCTION_HPP
#define AADAPT_ANALYTICFUNCTION_HPP

#include <string>

// Random and Gaussian number distribution
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include "Teuchos_Array.hpp"

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

class AerasScharDensity : public AnalyticFunction {
  public:
    AerasScharDensity(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXScalarAdvection : public AnalyticFunction {
  public:
    AerasXScalarAdvection(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXZHydrostatic : public AnalyticFunction {
  public:
    AerasXZHydrostatic(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    const int numDim; // size of coordinate vector X
    const int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXZHydrostaticGaussianBall : public AnalyticFunction {
  public:
    AerasXZHydrostaticGaussianBall(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    const int numDim; // size of coordinate vector X
    const int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXZHydrostaticGaussianBallInShear : public AnalyticFunction {
  public:
    AerasXZHydrostaticGaussianBallInShear(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    const int numDim; // size of coordinate vector X
    const int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXZHydrostaticGaussianVelocityBubble : public AnalyticFunction {
  public:
    AerasXZHydrostaticGaussianVelocityBubble(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    const int numDim; // size of coordinate vector X
    const int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasXZHydrostaticCloud : public AnalyticFunction {
  public:
    AerasXZHydrostaticCloud(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    const int numDim; // size of coordinate vector X
    const int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasHydrostatic : public AnalyticFunction {
  public:
    AerasHydrostatic(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasHeaviside : public AnalyticFunction {
  public:
    AerasHeaviside(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasCosineBell : public AnalyticFunction {
  public:
    AerasCosineBell(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasZonalFlow : public AnalyticFunction {
  public:
     AerasZonalFlow(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x

    Teuchos::Array<double> data;
};
class AerasTC5Init : public AnalyticFunction {
  public:
     AerasTC5Init(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);

  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    
    Teuchos::Array<double> data;
};
  
//------------------------------------------------------------------------------
    
class AerasTC3Init : public AnalyticFunction {
    public:
        AerasTC3Init(int neq_, int spatialDim_, Teuchos::Array<double> data_);
        void compute(double* x, const double* X);
    
    private:
        int spatialDim; // size of coordinate vector X
        int neq;        // size of solution vector x
    
        Teuchos::Array<double> data;
    
    private:
        double bx(const double x); //an indicator function
    
        double ucomponent(const double lon); //unrotated u-comp in TC3
  
        //obtains rotated lon lat for TC3.
        void rotate(const double lon, const double lat, const double alpha, double& rotlon, double& rotlat);
    
        double earthRadius; //Earth radius
        double testDuration; // =12 days, in seconds
        double myPi; // a local copy of pi
        double u0; //a u-comp velocity multiplier, based on Williamson1992
        double Omega;
        double gravity;
        double h0g;
  
        double xe;
        double thetae;
        double thetab;
};
  
//------------------------------------------------------------------------------
  
class AerasTCGalewskyInit : public AnalyticFunction {
    public:
        AerasTCGalewskyInit(int neq_, int spatialDim_, Teuchos::Array<double> data_);
        void compute(double* x, const double* X);
        
    private:
        int spatialDim; // size of coordinate vector X
        int neq;    // size of solution vector x
        
        Teuchos::Array<double> data;
        
    private:
    
        double ucomponent(const double lon); //
        double hperturb(const double lon, const double lat); //
    
        double earthRadius; //Earth radius
        double gravity; // a loc copy of gravity
        double myPi; // a local copy of pi
        double Omega;
    
        double phi0;
        double phi1;
        double umax;
        double en;

        double h0;
        double phi2;
        double al;
        double beta;
        double hhat;
};
  
//-------------------------------------------------------------------
class AerasTC4Init : public AnalyticFunction {
  public:
    AerasTC4Init(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
    
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    
    Teuchos::Array<double> data;
    
  private:
    
    double earthRadius; //Earth radius
    double myPi; // a local copy of pi
    double Omega;
    double gravity;
  
    double su0;
    double phi0;
    double rlon0;
    double rlat0;
  
    double alfa; //spelling is correct
    double sigma;
    double npwr;
  
    double phicon(const double lat);
    double bubfnc(const double lat);
    double dbubf(const double lat);
  
};

//----------------------------------------------------------------------------
  
class AerasPlanarCosineBell : public AnalyticFunction {
  public:
    AerasPlanarCosineBell(int neq_, int numDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int numDim; // size of coordinate vector X
    int neq;    // size of solution vector x
    Teuchos::Array<double> data;
};

class AerasRossbyHaurwitzWave : public AnalyticFunction {
  public:
    AerasRossbyHaurwitzWave(int neq_, int spatialDim_, Teuchos::Array<double> data_);
    void compute(double* x, const double* X);
  private:
    int spatialDim; // size of coordinate vector X
    int neq;    // size of solution vector x

    Teuchos::Array<double> data;
};

}

#endif
