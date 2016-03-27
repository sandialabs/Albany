//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Pareto_Optimizer.hpp"
#include "Teuchos_TestForException.hpp"
#include "ATO_Solver.hpp"

namespace ATO {

/**********************************************************************/
Optimizer_Pareto::Optimizer_Pareto(const Teuchos::ParameterList& optimizerParams) :
Optimizer(optimizerParams)
/**********************************************************************/
{ 
  TEUCHOS_TEST_FOR_EXCEPTION (
    !(optimizerParams.isType<double>("Volume Fraction")) &&
    !(optimizerParams.isType<double>("Final Volume Fraction") &&
      optimizerParams.isType<double>("Initial Volume Fraction") &&
      optimizerParams.isType<int>("Volume Fraction Steps")),
      Teuchos::Exceptions::InvalidParameter, std::endl
      << "Error! Pareto Optimizer: " << std::endl 
      << " Must specify either 'Volume Fraction' or 'Initial Volume Fraction', " << std::endl 
      << "'Final Volume Fraction', and 'Volume Fraction Steps'" << std::endl);

  if(optimizerParams.isType<double>("Volume Fraction")){
    _volFrac = optimizerParams.get<double>("Volume Fraction");
    _volFracLow  = 0.0;
    _volFracHigh = 0.0;
    _nVolFracSteps = 0;
  } else {
    _volFrac = 0.0;
    _volFracLow    = optimizerParams.get<double>("Final Volume Fraction");
    _volFracHigh   = optimizerParams.get<double>("Initial Volume Fraction");
    _nVolFracSteps = optimizerParams.get<int>("Volume Fraction Steps");
  }

  _volConvTol    = optimizerParams.get<double>("Volume Enforcement Convergence Tolerance");
  _volMaxIter    = optimizerParams.get<int>("Volume Enforcement Maximum Iterations");



}

/******************************************************************************/
Optimizer_Pareto::~Optimizer_Pareto()
/******************************************************************************/
{
}

/******************************************************************************/
void
Optimizer_Pareto::Initialize()
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION (
    solverInterface == NULL, Teuchos::Exceptions::InvalidParameter,
    std::endl << "Error! Optimizer requires valid Solver Interface" << std::endl);

  numOptDofs = solverInterface->GetNumOptDofs();

  p      = new double[numOptDofs];
  p_last = new double[numOptDofs];
  dfdp   = new double[numOptDofs];

  // pareto optimization starts with the design volume full
  std::fill_n(p,      numOptDofs, 1.0);
  std::fill_n(p_last, numOptDofs, 1.0);
  std::fill_n(dfdp,   numOptDofs, 0.0);

  solverInterface->ComputeVolume(_optVolume);

}

/******************************************************************************/
void
Optimizer_Pareto::Optimize()
/******************************************************************************/
{

  double volFrac, dVolFrac;
  if( _volFrac ){
    volFrac = _volFrac;
    dVolFrac = 0.0;
  } else {
    volFrac = _volFracHigh;
    dVolFrac = (_volFracLow - _volFracHigh)/_nVolFracSteps;
  }

  solverInterface->ComputeObjective(p, f, dfdp);
  computeUpdatedTopology(volFrac);

  double pnorm = computeNorm(p, numOptDofs);
  convergenceChecker->initNorm(f, pnorm);

  for(int iv=0; iv<=_nVolFracSteps; iv++){
  
    int iter=0;
    bool optimization_converged = false;
    while(!optimization_converged){
  
      f_last = f;
      solverInterface->ComputeObjective(p, f, dfdp);
      computeUpdatedTopology(volFrac);
    
      if(comm->MyPID()==0.0){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "** Optimization Status Check *******************************************" << std::endl;
        std::cout << "Status: Objective = " << f << std::endl;
      }
  
      double delta_f = f-f_last;
      double delta_p = computeDiffNorm(p, p_last, numOptDofs, /*result to cout*/ false);
  
      optimization_converged = convergenceChecker->isConverged(delta_f, delta_p, iter, comm->MyPID());

      iter++;
    }

    volFrac += dVolFrac;

  }

}

/******************************************************************************/
void
Optimizer_Pareto::computeUpdatedTopology(double volFrac)
/******************************************************************************/
{
  double vol = 0.0;

  double localTauMax = dfdp[0];
  double localTauMin = dfdp[0];

  for(int i=1; i<numOptDofs; i++){
    if( dfdp[i] > localTauMax ) localTauMax = dfdp[i];
    if( dfdp[i] < localTauMin ) localTauMin = dfdp[i];
  }

  double tau_1, tau_2, tauMid;
  comm->MinAll(&localTauMin, &tau_1, 1);
  comm->MaxAll(&localTauMax, &tau_2, 1);

  const double minDensity = topology->getBounds()[0];

  for(int i=0; i<numOptDofs; i++)
    p_last[i] = p[i];

  int niters = 0;
  do {
//    TEUCHOS_TEST_FOR_EXCEPTION(
//      niters > _volMaxIter, Teuchos::Exceptions::InvalidParameter,
//      std::endl << "Enforcement of volume constraint failed:  Exceeded max iterations"
//      << std::endl);

    vol = 0.0;
    tauMid = (tau_2+tau_1)/2.0;

    solverInterface->ComputeVolume(p, dfdp, vol, tauMid, minDensity);

    if( vol > volFrac*_optVolume ) 
      tau_2 = tauMid;
    else 
      tau_1 = tauMid;

    niters++;

    if( niters > _volMaxIter ){
      if(comm->MyPID() == 0 ){
        std::cout << std::endl 
        << "Enforcement of volume constraint failed:  Exceeded max iterations" << std::endl
        << "Actual = " << vol << std::endl
        << "Target = " << volFrac*_optVolume << std::endl;
      }
      break;
    }


  } while ( fabs(vol - volFrac*_optVolume) > _volConvTol );

}

}
