//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_Optimizer_HPP
#define ATO_Optimizer_HPP

#include "Albany_StateManager.hpp"

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"

#ifdef ATO_USES_NLOPT
#include "nlopt.h"
#endif //ATO_USES_NLOPT

namespace ATO {

class Solver;
class OptInterface;

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
  OptInterface* solverInterface;
  Teuchos::RCP<const Epetra_Comm> comm;

  double _optConvTol;
  int    _optMaxIter;

};


class Optimizer_OC : public Optimizer {
 public:
  Optimizer_OC(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_OC();
  void Optimize();
  void Initialize();
 private:
  void computeUpdatedTopology();

  double* p;
  double* p_last;
  double f;
  double* dfdp;
  int numOptDofs;

  double _volConvTol;
  double _volMaxIter;
  double _minDensity;
  double _initLambda;
  double _moveLimit;
  double _stabExponent;
  double _volConstraint;
  double _optVolume;

};

#ifdef ATO_USES_NLOPT
class Optimizer_NLopt : public Optimizer {
 public:
  Optimizer_NLopt(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_NLopt();
  void Optimize();
  void Initialize();
 private:

  double* p;
  double* p_last;
  double f;
  double* dfdp;
  int numOptDofs;

  double _minDensity;
  double _volConstraint;
  double _volConvTol;
  double _optVolume;

  std::string _optMethod;

  nlopt_opt opt;

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


}
#endif
