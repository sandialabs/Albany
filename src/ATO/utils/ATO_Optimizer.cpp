//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Optimizer.hpp"
#include "Teuchos_TestForException.hpp"
#include "ATO_Solver.hpp"
#ifdef ATO_USES_NLOPT
#include <nlopt.h>
#endif //ATO_USES_NLOPT
#include <algorithm>

namespace ATO {


/**********************************************************************/
Teuchos::RCP<Optimizer> 
OptimizerFactory::create(const Teuchos::ParameterList& optimizerParams)
/**********************************************************************/
{
  std::string optPackage = optimizerParams.get<std::string>("Package");
  if( optPackage == "OC"  )  return Teuchos::rcp(new Optimizer_OC(optimizerParams));
#ifdef ATO_USES_NLOPT
  else
  if( optPackage == "NLopt"  )  return Teuchos::rcp(new Optimizer_NLopt(optimizerParams));
#endif //ATO_USES_NLOPT
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Optimization package: " << optPackage << " Unknown!" << std::endl 
      << "Valid options are\n"
      << "/t OC ... optimiality criterion\n"
#ifdef ATO_USES_NLOPT
      << "/t NLopt ... NLOPT library\n" 
#endif //ATO_USES_NLOPT
      << std::endl);
}

/**********************************************************************/
Optimizer::Optimizer(const Teuchos::ParameterList& optimizerParams)
/**********************************************************************/
{ 
  _optConvTol = optimizerParams.get<double>("Optimization Convergence Tolerance");
  _optMaxIter = optimizerParams.get<int>("Optimization Maximum Iterations");

  solverInterface = NULL;
}

/**********************************************************************/
Optimizer_OC::Optimizer_OC(const Teuchos::ParameterList& optimizerParams) :
Optimizer(optimizerParams)
/**********************************************************************/
{ 
  p = NULL;
  p_last = NULL;
  dfdp = NULL;

  _volConvTol    = optimizerParams.get<double>("Volume Enforcement Convergence Tolerance");
  _volMaxIter    = optimizerParams.get<int>("Volume Enforcement Maximum Iterations");
  _minDensity    = optimizerParams.get<double>("Minimum Density");
  _initLambda    = optimizerParams.get<double>("Volume Multiplier Initial Guess");
  _volConstraint = optimizerParams.get<double>("Volume Fraction Constraint");
  _moveLimit     = optimizerParams.get<double>("Move Limiter");
  _stabExponent  = optimizerParams.get<double>("Stabilization Parameter");
}

#ifdef ATO_USES_NLOPT
/**********************************************************************/
Optimizer_NLopt::Optimizer_NLopt(const Teuchos::ParameterList& optimizerParams) :
Optimizer(optimizerParams)
/**********************************************************************/
{ 
  p = NULL;
  p_last = NULL;

  opt = NULL;

  _minDensity    = optimizerParams.get<double>("Minimum Density");
  _volConstraint = optimizerParams.get<double>("Volume Fraction Constraint");
  _optMethod     = optimizerParams.get<std::string>("Method");
  _volConvTol = optimizerParams.get<double>("Volume Enforcement Convergence Tolerance");
}
#endif //ATO_USES_NLOPT

/**********************************************************************/
void
Optimizer::SetInterface(Solver* mySolverInterface)
/**********************************************************************/
{
  solverInterface = dynamic_cast<OptInterface*>(mySolverInterface);
  TEUCHOS_TEST_FOR_EXCEPTION(
    solverInterface == NULL, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error! Dynamic cast of Solver* to OptInterface* failed." << std::endl);
}

/******************************************************************************/
Optimizer_OC::~Optimizer_OC()
/******************************************************************************/
{
  if( p      ) delete [] p;
  if( p_last ) delete [] p_last;
  if( dfdp   ) delete [] dfdp;
}

#ifdef ATO_USES_NLOPT
/******************************************************************************/
Optimizer_NLopt::~Optimizer_NLopt()
/******************************************************************************/
{
  if( p      ) delete [] p;
  if( p_last ) delete [] p_last;
}
#endif //ATO_USES_NLOPT


/******************************************************************************/
double
Optimizer::computeDiffNorm(const double* p, const double* p_last, int n, bool printResult)
/******************************************************************************/
{
  double norm = 0.0;
  for(int i=0; i<n; i++){
    norm += pow(p[i]-p_last[i],2);
  }
  double gnorm = 0.0;
  comm->SumAll(&norm, &gnorm, 1);
  gnorm = (gnorm > 0.0) ? sqrt(gnorm) : 0.0;
  if(printResult && comm->MyPID()==0){
    std::cout << "************************************************************************" << std::endl;
    std::cout << "  Optimizer:  computed diffnorm is: " << gnorm << std::endl;
    std::cout << "************************************************************************" << std::endl;
  }
  return gnorm;
}

/******************************************************************************/
void
Optimizer_OC::Initialize()
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION (
    solverInterface == NULL, Teuchos::Exceptions::InvalidParameter, 
    std::endl << "Error! Optimizer requires valid Solver Interface" << std::endl);

  numOptDofs = solverInterface->GetNumOptDofs();

  p      = new double[numOptDofs];
  p_last = new double[numOptDofs];
  dfdp   = new double[numOptDofs];

  std::fill_n(p,      numOptDofs, _volConstraint);
  std::fill_n(p_last, numOptDofs, 0.0);
  std::fill_n(dfdp,   numOptDofs, 0.0);

  solverInterface->ComputeVolume(_optVolume);
}

/******************************************************************************/
void
Optimizer_OC::Optimize()
/******************************************************************************/
{

  int iter=0;
  bool optimization_converged = false;

  while(!optimization_converged && iter < _optMaxIter) {

    solverInterface->ComputeObjective(p, f, dfdp);

    computeUpdatedTopology();

    // check for convergence
    double delta_p = computeDiffNorm(p, p_last, numOptDofs, /*result to cout*/ true);
    if( delta_p < _optConvTol ) optimization_converged = true;

    iter++;
  }

  return;
}



/******************************************************************************/
void
Optimizer_OC::computeUpdatedTopology()
/******************************************************************************/
{

  // find multiplier that enforces volume constraint
  const double maxDensity = 1.0;
  double vmid, v1=0.0;
  double v2=_initLambda;
  int niters=0;

  for(int i=0; i<numOptDofs; i++)
    p_last[i] = p[i];

  double vol = 0.0;
  do {
    TEUCHOS_TEST_FOR_EXCEPTION(
      niters > _volMaxIter, Teuchos::Exceptions::InvalidParameter, 
      std::endl << "Enforcement of volume constraint failed:  Exceeded max iterations" 
      << std::endl);

    vol = 0.0;
    vmid = (v2+v1)/2.0;

    // update topology
    for(int i=0; i<numOptDofs; i++) {
      double be = -dfdp[i]/vmid;
      double p_old = p_last[i];
      double p_new = p_old*pow(be,_stabExponent);
      // limit change
      double dval = p_new - p_old;
      if( fabs(dval) > _moveLimit) p_new = p_old+fabs(dval)/dval*_moveLimit;
      // enforce limits
      if( p_new < _minDensity ) p_new = _minDensity;
      if( p_new > maxDensity ) p_new = maxDensity;
      p[i] = p_new;
    }

    // compute new volume
    solverInterface->ComputeVolume(p, vol);
    if( (vol - _volConstraint*_optVolume) > 0.0 ) v1 = vmid;
    else v2 = vmid;
    niters++;
  } while ( fabs(vol - _volConstraint*_optVolume) > _volConvTol );
}

#ifdef ATO_USES_NLOPT
/******************************************************************************/
void
Optimizer_NLopt::Initialize()
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION (
    solverInterface == NULL, Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error! Optimizer requires valid Solver Interface" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (
    (comm->NumProc() != 1) && (comm->MyPID()==0), 
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error! NLopt package doesn't work in parallel.  Use OC package." << std::endl);
  TEUCHOS_TEST_FOR_EXCEPT ( (comm->NumProc() != 1) );
 
  

  numOptDofs = solverInterface->GetNumOptDofs();
  
  if( _optMethod == "MMA" )
    opt = nlopt_create(NLOPT_LD_MMA, numOptDofs);
  else
  if( _optMethod == "CCSA" )
    opt = nlopt_create(NLOPT_LD_CCSAQ, numOptDofs);
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Optimization method: " << _optMethod << " Unknown!" << std::endl 
      << "Valid options are (MMA)" << std::endl);

  
  // set bounds
  nlopt_set_lower_bounds1(opt, _minDensity);
  nlopt_set_upper_bounds1(opt, 1.0);

  // set objective function
  nlopt_set_min_objective(opt, this->evaluate, this);
  
  // set stop criteria
  nlopt_set_xtol_rel(opt, _optConvTol);
  nlopt_set_maxeval(opt, _optMaxIter);

  // set volume constraint
  nlopt_add_inequality_constraint(opt, this->constraint, this, _volConvTol);

  p      = new double[numOptDofs];
  p_last = new double[numOptDofs];

  std::fill_n(p,      numOptDofs, _volConstraint);
  std::fill_n(p_last, numOptDofs, 0.0);

  solverInterface->ComputeVolume(_optVolume);
}

#define ATO_XTOL_REACHED 104

/******************************************************************************/
void
Optimizer_NLopt::Optimize()
/******************************************************************************/
{

  double minf;
  int errorcode = nlopt_optimize(opt, p, &minf);

  if( errorcode == NLOPT_FORCED_STOP && comm->MyPID()==0 ){
    int forcestop_errorcode = nlopt_get_force_stop(opt);
    std::cout << "************************************************************************" << std::endl;
    std::cout << "  Optimizer converged.  Objective value = " << minf << std::endl;
    std::cout << "    Convergence code: " << forcestop_errorcode << std::endl;
    std::cout << "************************************************************************" << std::endl;
  } else
  if( errorcode < 0 ){
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Optimization failed with errorcode " << errorcode << std::endl);
  }

  return;
}

/******************************************************************************/
double 
Optimizer_NLopt::evaluate_backend( unsigned int n, const double* x, double* grad )
/******************************************************************************/
{
  double delta_p = computeDiffNorm(x, p_last, numOptDofs, /*result to cout*/ true);
  if( delta_p < _optConvTol ){
    nlopt_set_force_stop(opt, ATO_XTOL_REACHED);
    nlopt_force_stop(opt);
  }

  double f;
  solverInterface->ComputeObjective(x, f, grad);

  std::memcpy((void*)p_last, (void*)x, numOptDofs*sizeof(double));

  if(comm->MyPID()==0){
    std::cout << "************************************************************************" << std::endl;
    std::cout << "  Optimizer:  objective value is: " << f << std::endl;
    std::cout << "************************************************************************" << std::endl;
  }

  return f;
}

/******************************************************************************/
double 
Optimizer_NLopt::evaluate( unsigned int n, const double* x,
                           double* grad, void* data)
/******************************************************************************/
{
  Optimizer_NLopt* NLopt = reinterpret_cast<Optimizer_NLopt*>(data);
  return NLopt->evaluate_backend(n, x, grad);
}

/******************************************************************************/
double 
Optimizer_NLopt::constraint_backend( unsigned int n, const double* x, double* grad )
/******************************************************************************/
{
  double vol;
  solverInterface->ComputeVolume(x, vol, grad);

  if(comm->MyPID()==0){
    std::cout << "************************************************************************" << std::endl;
    std::cout << "  Optimizer:  computed volume is: " << vol << std::endl;
    std::cout << "************************************************************************" << std::endl;
  }
  return vol-_volConstraint*_optVolume;
}

/******************************************************************************/
double 
Optimizer_NLopt::constraint( unsigned int n, const double* x,
                             double* grad, void* data)
/******************************************************************************/
{
  Optimizer_NLopt* NLopt = reinterpret_cast<Optimizer_NLopt*>(data);
  return NLopt->constraint_backend(n, x, grad);
}
#endif //ATO_USES_NLOPT


}

