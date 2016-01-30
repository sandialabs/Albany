//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_Optimizer_HPP
#define ATO_Optimizer_HPP

#include "Albany_StateManager.hpp"
#include "Teuchos_ParameterList.hpp"
#include "ATO_TopoTools.hpp"
#include "ATO_Types.hpp"

#include <string>
#include <vector>


#ifdef ATO_USES_NLOPT
#include "nlopt.h"
#endif //ATO_USES_NLOPT

namespace ATO {

class Solver;
class OptInterface;
class ConvergenceTest;

class Optimizer 
/** \brief Optimizer wrapper

    This class provides a very basic container for topological optimization algorithms.

*/
{
 public:
  Optimizer(const Teuchos::ParameterList& optimizerParams);
  virtual ~Optimizer(){};

  virtual void Optimize()=0;
  virtual void Initialize()=0;
  virtual void SetInterface(Solver*);
  virtual void SetCommunicator(const Teuchos::RCP<const Epetra_Comm>& _comm) {comm = _comm;}
 protected:

  double computeDiffNorm(const double* v1, const double* v2, int n, bool printResult=false);
  double computeNorm(const double* v2, int n);
  OptInterface* solverInterface;
  Teuchos::RCP<const Epetra_Comm> comm;

  Teuchos::RCP<ConvergenceTest> convergenceChecker;
  Teuchos::RCP<Topology> topology;

  int    _nIterations;

  std::string _measureType;

};

class Optimizer_OC : public Optimizer {
 public:
  Optimizer_OC(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_OC();
  virtual void Optimize();
  void Initialize();
 protected:
  void computeUpdatedTopology();

  double* p;
  double* p_last;
  double* dmdp;

  double f;
  double f_last;
  double* dfdp;

  double g;
  double g_last;
  double* dgdp;


  int numOptDofs;

  std::string secondaryConstraintGradient;

  double _measureConvTol;
  double _measureAccpTol;
  double _measureMaxIter;
  double _initLambda;
  double _moveLimit;
  double _stabExponent;
  double _measureConstraint;
  double _minMeasure;
  double _maxMeasure;
  double _optMeasure;
  bool   _useNewtonSearch;

};

class Optimizer_OCG : public Optimizer_OC {
  public:
    Optimizer_OCG(const Teuchos::ParameterList& optimizerParams) : Optimizer_OC(optimizerParams){}
    virtual void Optimize();
};


#ifdef ATO_USES_NLOPT
class Optimizer_NLopt : public Optimizer {
 public:
  Optimizer_NLopt(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_NLopt();
  void Optimize();
  void Initialize();
 private:

  enum ResponseType {Measure, Aggregate};

  ResponseType objectiveType, primaryConstraintType;

  double objectiveValue, objectiveValue_last;
  double constraintValue, constraintValue_last;

  double *p, *p_last;
  double f, g, measure;
  double* dfdp;
  double* dgdp;
  double* dmdp;
  int numOptDofs;

  double _minDensity;
  double _measureConstraint;
  double _measureConvTol;
  double _optMeasure;
  double _optConvTol;

  std::string _optMethod;
  double _conConvTol;

  nlopt_opt opt;

  bool isChanged(const double* x);
  double* x_ref;

  double evaluate_backend( unsigned int n, const double* x, double* grad );
  static double evaluate( unsigned int n, const double* x, double* grad, void* data);

  double constraint_backend( unsigned int n, const double* x, double* grad );
  static double constraint( unsigned int n, const double* x, double* grad, void* data);
};
#endif //ATO_USES_NLOPT


class OptimizerFactory {
 public:
  Teuchos::RCP<Optimizer> create(const Teuchos::ParameterList& optimizerParams);
};

class ConvergenceTest {
  public:
    ConvergenceTest(const Teuchos::ParameterList& convParams);

    bool isConverged( double delta_f, double delta_p, int iter, int myPID = -1);
    void initNorm(double f, double p);
    int getMaxIterations(){return maxIterations;}

  private:

    int maxIterations;
    int minIterations;

    enum ComboType {AND, OR};

    ComboType comboType;

    class ConTest {
      public: 
        ConTest(double val) : conValue(val){};
        virtual bool passed(double delta_f, double delta_p, bool write) = 0;
        virtual void initNorm(double f0, double p0){}
      protected:
        double conValue;
    };
    class AbsDeltaP : public ConTest {
      public:
        AbsDeltaP(double val) : ConTest(val){}
        virtual bool passed(double delta_f, double delta_p, bool write);
    };
    class RelDeltaP : public ConTest {
      public:
        RelDeltaP(double val) : ConTest(val),p0(0.0){};
        virtual bool passed(double delta_f, double delta_p, bool write);
        virtual void initNorm(double f, double p){p0 = p;}
      private:
        double p0;
    };
    class AbsDeltaF : public ConTest {
      public:
        AbsDeltaF(double val) : ConTest(val){}
        virtual bool passed(double delta_f, double delta_p, bool write);
    };
    class RelDeltaF : public ConTest {
      public:
        RelDeltaF(double val) : ConTest(val),f0(0.0){};
        virtual bool passed(double delta_f, double delta_p, bool write);
        virtual void initNorm(double f, double p){f0 = f;}
      private:
        double f0;
    };
    class AbsRunningDF : public ConTest {
      public:
        AbsRunningDF(double val) : ConTest(val),runningDF(0.0), nave(10){}
        virtual bool passed(double delta_f, double delta_p, bool write);
      private:
        std::vector<double> dF;
        double runningDF;
        int nave;
    };
    class RelRunningDF : public ConTest {
      public:
        RelRunningDF(double val) : ConTest(val),f0(0.0),runningDF(0.0),nave(10){};
        virtual bool passed(double delta_f, double delta_p, bool write);
        virtual void initNorm(double f, double p){f0 = f;}
      private:
        double f0;
        std::vector<double> dF;
        double runningDF;
        int nave;
    };

    Teuchos::Array<Teuchos::RCP<ConTest> > conTests;
};

}
#endif
