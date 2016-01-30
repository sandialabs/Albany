//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Optimizer.hpp"
#include "ATO_Pareto_Optimizer.hpp"
#include "Teuchos_TestForException.hpp"
#include "ATO_Solver.hpp"

#ifdef ATO_USES_NLOPT
#include <nlopt.h>
#endif //ATO_USES_NLOPT

#ifdef ATO_USES_DOTK
#include "ATO_DOTk_Optimizer.hpp"
#endif //ATO_USES_DOTK

#include <algorithm>


namespace ATO {

/**********************************************************************/
Teuchos::RCP<Optimizer> 
OptimizerFactory::create(const Teuchos::ParameterList& optimizerParams)
/**********************************************************************/
{
  std::string optPackage = optimizerParams.get<std::string>("Package");

  if( optPackage == "OC"  )  return Teuchos::rcp(new Optimizer_OC(optimizerParams));

  else
  if( optPackage == "OCG"  )  return Teuchos::rcp(new Optimizer_OCG(optimizerParams));

  else
  if( optPackage == "Pareto"  )  return Teuchos::rcp(new Optimizer_Pareto(optimizerParams));

#ifdef ATO_USES_NLOPT
  else
  if( optPackage == "NLopt"  ) return Teuchos::rcp(new Optimizer_NLopt(optimizerParams));

#endif //ATO_USES_NLOPT

#ifdef ATO_USES_DOTK
  else
  if( optPackage == "DOTk"  )  return Teuchos::rcp(new Optimizer_DOTk(optimizerParams));
#endif //ATO_USES_DOTK
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Optimization package: " << optPackage << " Unknown!" << std::endl 
      << "Valid options are\n"
      << "/t OC ... optimality criterion\n"
      << "/t Pareto ... pareto optimization\n"

#ifdef ATO_USES_NLOPT
      << "/t NLopt ... NLOPT library\n" 
#endif //ATO_USES_NLOPT

#ifdef ATO_USES_DOTK
      << "/t DOTk ... Design Optimization Toolkit library\n" 
#endif //ATO_USES_DOTK
      << std::endl);

}

/**********************************************************************/
Optimizer::Optimizer(const Teuchos::ParameterList& optimizerParams)
/**********************************************************************/
{ 

  solverInterface = NULL;
  comm = Teuchos::null;

  if( optimizerParams.isType<Teuchos::ParameterList>("Convergence Tests") ){
    const Teuchos::ParameterList& 
      convParams = optimizerParams.get<Teuchos::ParameterList>("Convergence Tests");
    convergenceChecker = Teuchos::rcp(new ConvergenceTest(convParams));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, 
      std::endl << "Optimization convergence:  'Convergence Tests' ParameterList is required" << std::endl);
  }

  if( optimizerParams.isType<Teuchos::ParameterList>("Measure Enforcement") ){
    const Teuchos::ParameterList& 
      measureParams = optimizerParams.get<Teuchos::ParameterList>("Measure Enforcement");

    _measureType = measureParams.get<std::string>("Measure");
  } else
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error! Missing 'Measure Enforcement' ParameterList." << std::endl);
  

}

/**********************************************************************/
Optimizer_OC::Optimizer_OC(const Teuchos::ParameterList& optimizerParams) :
Optimizer(optimizerParams)
/**********************************************************************/
{ 
  p = NULL;
  p_last = NULL;
  f = 0.0;
  f_last = 0.0;
  g = 0.0;
  g_last = 0.0;
  dfdp = NULL;
  dgdp = NULL;
  dmdp = NULL;

  _moveLimit     = optimizerParams.get<double>("Move Limiter");
  _stabExponent  = optimizerParams.get<double>("Stabilization Parameter");

  if( optimizerParams.isType<Teuchos::ParameterList>("Measure Enforcement") ){
    const Teuchos::ParameterList& 
      measureParams = optimizerParams.get<Teuchos::ParameterList>("Measure Enforcement");

    _measureConvTol    = measureParams.get<double>("Convergence Tolerance");
    _measureConstraint = measureParams.get<double>("Target");
    _measureMaxIter    = measureParams.get<int>("Maximum Iterations");

    if(measureParams.isType<double>("Minimum"))
      _minMeasure        = measureParams.get<double>("Minimum");
    else _minMeasure     = 0.1;
    if(measureParams.isType<double>("Maximum"))
      _maxMeasure        = measureParams.get<double>("Maximum");
    else _maxMeasure     = 1.0;
    if( measureParams.isType<double>("Acceptable Tolerance") )
      _measureAccpTol    = measureParams.get<double>("Acceptable Tolerance");
    else _measureAccpTol = _measureConvTol;
    if( measureParams.isType<bool>("Use Newton Search") )
      _useNewtonSearch   = measureParams.get<bool>("Use Newton Search");
    else _useNewtonSearch = true;

  } else
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error! Missing 'Measure Enforcement' ParameterList." << std::endl);
  

  if( optimizerParams.isType<Teuchos::ParameterList>("Constraint Enforcement") ){
    const Teuchos::ParameterList& 
      conParams = optimizerParams.get<Teuchos::ParameterList>("Constraint Enforcement");
    secondaryConstraintGradient = conParams.get<std::string>("Constraint Gradient");
  } else
    secondaryConstraintGradient = "None";

}

#ifdef ATO_USES_NLOPT
/**********************************************************************/
Optimizer_NLopt::Optimizer_NLopt(const Teuchos::ParameterList& optimizerParams) :
Optimizer(optimizerParams)
/**********************************************************************/
{ 
  p = NULL;
  p_last = NULL;
  objectiveValue = 0.0;
  objectiveValue_last = 0.0;
  constraintValue = 0.0;
  constraintValue_last = 0.0;
  opt = NULL;

  _optMethod     = optimizerParams.get<std::string>("Method");
  _nIterations = 0;

  if( optimizerParams.isType<std::string>("Objective") ){
    std::string objType = optimizerParams.get<std::string>("Objective");
    if( objType == "Measure" )
      objectiveType = ResponseType::Measure;
    else
    if( objType == "Aggregator" || objType == "Objective Aggregator" )
      objectiveType = ResponseType::Aggregate;
    else
     TEUCHOS_TEST_FOR_EXCEPTION(
       true, Teuchos::Exceptions::InvalidParameter, std::endl << "Unknown objective specified." << std::endl);
  } else
  objectiveType = ResponseType::Aggregate;

  if( optimizerParams.isType<std::string>("Constraint") ){
    std::string conType = optimizerParams.get<std::string>("Constraint");
    if( conType == "Measure" )
      primaryConstraintType = ResponseType::Measure;
    else
    if( conType == "Aggregator" || conType == "Constraint Aggregator" )
      primaryConstraintType = ResponseType::Aggregate;
    else
     TEUCHOS_TEST_FOR_EXCEPTION(
       true, Teuchos::Exceptions::InvalidParameter, std::endl << "Unknown constraint specified." << std::endl);
  } else
  primaryConstraintType = ResponseType::Measure;


  if( primaryConstraintType == ResponseType::Aggregate ){
    if( optimizerParams.isType<Teuchos::ParameterList>("Constraint Enforcement") ){
      const Teuchos::ParameterList& 
        conParams = optimizerParams.get<Teuchos::ParameterList>("Constraint Enforcement");
      _conConvTol = conParams.get<double>("Convergence Tolerance");
    } else
      TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error! Missing 'Constraint Enforcement' ParameterList." << std::endl);
  } else
  if( primaryConstraintType == ResponseType::Measure ){
    if( optimizerParams.isType<Teuchos::ParameterList>("Measure Enforcement") ){
      const Teuchos::ParameterList& 
        measureParams = optimizerParams.get<Teuchos::ParameterList>("Measure Enforcement");
  
      _measureConvTol    = measureParams.get<double>("Convergence Tolerance");
      _measureConstraint = measureParams.get<double>("Target");
    } else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error! Missing 'Measure Enforcement' ParameterList." << std::endl);
  }
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
  if( dgdp   ) delete [] dgdp;
  if( dmdp   ) delete [] dmdp;
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
Optimizer::computeNorm(const double* p, int n)
/******************************************************************************/
{
  double norm = 0.0;
  for(int i=0; i<n; i++){
    norm += p[i]*p[i];
  }
  double gnorm = 0.0;
  comm->SumAll(&norm, &gnorm, 1);
  gnorm = (gnorm > 0.0) ? sqrt(gnorm) : 0.0;
  return gnorm;
}


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
  dmdp   = new double[numOptDofs];

  std::fill_n(p,      numOptDofs, 0.0);
  std::fill_n(p_last, numOptDofs, 0.0);
  std::fill_n(dfdp,   numOptDofs, 0.0);
  std::fill_n(dmdp,   numOptDofs, 0.0);

  if( secondaryConstraintGradient != "None" ){
    dgdp = new double[numOptDofs];
    std::fill_n(dgdp, numOptDofs, 0.0);
  }

  solverInterface->ComputeMeasure(_measureType, _optMeasure);
  solverInterface->InitializeOptDofs(p);
}
/******************************************************************************/
void
Optimizer_OCG::Optimize()
/******************************************************************************/
{
  solverInterface->Compute(p, f, dfdp, g, dgdp);

  double global_f=0.0, pnorm = computeNorm(p, numOptDofs);
  comm->SumAll(&f, &global_f, 1);
  convergenceChecker->initNorm(global_f, pnorm);

  Teuchos::Array<double> upperBound, lowerBound;
  solverInterface->getOptDofsUpperBound(upperBound);
  solverInterface->getOptDofsLowerBound(lowerBound);

  int iter=0;
  bool optimization_converged = false;
  while(!optimization_converged) {

    f_last = f; g_last = g;
    solverInterface->Compute(p, f, dfdp, g, dgdp);
    for(int i=0; i<numOptDofs; i++) p_last[i] = p[i];

    double gmax_dgdp =0.0;
    comm->MaxAll(std::max_element(dgdp, dgdp+numOptDofs), &gmax_dgdp, 1);

    double vmid, v1=0.0, v2=0.0;
    double dfdp_tot = 0.0, dgdp_tot = 0.0;
    for(int i=0; i<numOptDofs; i++) {
      dfdp_tot += dfdp[i];
      dgdp_tot += (dgdp[i]-gmax_dgdp);
    }

    double g_dfdp_tot = 0.0, g_dgdp_tot = 0.0;
    comm->SumAll(&dfdp_tot, &g_dfdp_tot, 1);
    comm->SumAll(&dgdp_tot, &g_dgdp_tot, 1);

    v2 = 1.0e4 * g_dfdp_tot / g_dgdp_tot;
    int niters=0;
    double newResidual;
    do {
      vmid = (v2+v1)/2.0;

      // update topology
      for(int i=0; i<numOptDofs; i++) {
        double be = dfdp[i]/(dgdp[i]-gmax_dgdp)/vmid;
        be = (be > 0.0) ? be : 0.0;
        double p_old = p_last[i];
//        double p_new = (p_old-offset)*pow(be,_stabExponent)+offset;
        double p_new = p_old*pow(be,_stabExponent);
        // limit change
        double dval = p_new - p_old;
        if( fabs(dval) > _moveLimit) p_new = p_old+fabs(dval)/dval*_moveLimit;
        // enforce limits
        if( p_new < lowerBound[i] ) p_new = lowerBound[i];
        if( p_new > upperBound[i] ) p_new = upperBound[i];
        p[i] = p_new;
      }

      newResidual = -g;
      for(int i=0; i<numOptDofs; i++)
        newResidual += (dgdp[i]-gmax_dgdp)*(p[i] - p_last[i]);
  
      if( newResidual < 0.0 ){
        v1 = vmid;
      } else v2 = vmid;

      if(comm->MyPID()==0){
        std::cout << "Constraint enforcement (iteration " << niters << "): Residual = " << newResidual << std::endl;
      }

    } while ( niters < _measureMaxIter && fabs(newResidual) > 1e-2  );
  
    if(comm->MyPID()==0.0){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "** Optimization Status Check *******************************************" << std::endl;
      std::cout << "Status: Objective = " << f << std::endl;
    }

    double delta_f, ldelta_f = f-f_last;
    comm->SumAll(&ldelta_f, &delta_f, 1);
    double delta_p = computeDiffNorm(p, p_last, numOptDofs, /*result to cout*/ false);

    optimization_converged = convergenceChecker->isConverged(delta_f, delta_p, iter, comm->MyPID());

    iter++;
  }

  return;
}


/******************************************************************************/
void
Optimizer_OC::Optimize()
/******************************************************************************/
{

  double measure=0.0;

  if( secondaryConstraintGradient == "Adjoint" )
    solverInterface->Compute(p, f, dfdp, g, dgdp);
  else 
    solverInterface->Compute(p, f, dfdp, g);

  for(int i=0; i<numOptDofs; i++) p_last[i] = p[i];
  solverInterface->ComputeMeasure(_measureType, p, measure, dmdp);

  computeUpdatedTopology();

  double global_f=0.0, pnorm = computeNorm(p, numOptDofs);
  comm->SumAll(&f, &global_f, 1);
  convergenceChecker->initNorm(global_f, pnorm);

  int iter=0;
  double measureConstraint_last = _measureConstraint;
  std::list<double> dgdm_vals;
  bool optimization_converged = false;
  while(!optimization_converged) {

    f_last = f; g_last = g;
    if( secondaryConstraintGradient == "Adjoint" )
      solverInterface->Compute(p, f, dfdp, g, dgdp);
    else
      solverInterface->Compute(p, f, dfdp, g);

    solverInterface->ComputeMeasure(_measureType, p, measure, dmdp);

    for(int i=0; i<numOptDofs; i++) p_last[i] = p[i];

  
    if( g != 0.0 ){
      // if the constraint condition isn't satisfied, modify the measure budget.

      double deltam = 0.0;
      if( secondaryConstraintGradient == "Adjoint" ){
        double dm = 0.001;
        _measureConstraint += dm;
        computeUpdatedTopology();
        _measureConstraint -= dm;
  
        double dg = 0.0;
        for(int i=0; i<numOptDofs; i++){
          dg += dgdp[i]*(p[i]-p_last[i]);
        }
        double global_dg = 0.0;
        comm->SumAll(&dg, &global_dg, 1);
        double dgdm = global_dg/dm;
        deltam = -g / dgdm;

      } else 
      if( secondaryConstraintGradient == "Finite Difference" ){
        if( _measureConstraint != measureConstraint_last ){
          dgdm_vals.push_back( (g - g_last)/(_measureConstraint - measureConstraint_last) );
          if( dgdm_vals.size() > 10 ) dgdm_vals.pop_front();
          std::list<double>::iterator it;
          double dgdm = 0.0;
          for(it=dgdm_vals.begin(); it!=dgdm_vals.end(); ++it) dgdm += *it;
          dgdm /= dgdm_vals.size();
          deltam = -g / dgdm;
        } else {
          deltam = 0.001;
        }
      } else
      if( secondaryConstraintGradient == "Direct"){
      }
    
      double _dmeasureLimit = 0.1*_measureConstraint;
      if(fabs(deltam) > _dmeasureLimit) deltam = deltam/fabs(deltam) * _dmeasureLimit;

      measureConstraint_last = _measureConstraint;
      _measureConstraint += deltam;
  
      if(_measureConstraint < _minMeasure) _measureConstraint = _minMeasure;
      if(_measureConstraint > _maxMeasure) _measureConstraint = _maxMeasure;

 
    }

    computeUpdatedTopology();


    if(comm->MyPID()==0.0){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "** Optimization Status Check *******************************************" << std::endl;
      std::cout << "Status: Objective = " << f << std::endl;
    }

    double delta_f, ldelta_f = f-f_last;
    comm->SumAll(&ldelta_f, &delta_f, 1);
    double delta_p = computeDiffNorm(p, p_last, numOptDofs, /*result to cout*/ false);

    optimization_converged = convergenceChecker->isConverged(delta_f, delta_p, iter, comm->MyPID());

    iter++;
  }

  return;
}

/******************************************************************************/
void
ConvergenceTest::initNorm( double f, double pnorm )
/******************************************************************************/
{
    Teuchos::Array<Teuchos::RCP<ConTest> >::iterator it;
    for(it=conTests.begin(); it!=conTests.end(); it++)
      (*it)->initNorm(f, pnorm);
}

/******************************************************************************/
bool
ConvergenceTest::isConverged( double delta_f, double delta_p, int iter, int myPID )
/******************************************************************************/
{
    if(iter == 0) return false;

    bool writeToCout = false;
    if(myPID == 0) writeToCout = true;

    // check convergence based on user defined criteria
    if(writeToCout){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "** Optimization Convergence Check **************************************" << std::endl;
    }
    std::vector<bool> results;
    Teuchos::Array<Teuchos::RCP<ConTest> >::iterator it;
    for(it=conTests.begin(); it!=conTests.end(); it++)
      results.push_back((*it)->passed(delta_f, delta_p, writeToCout));
    
    bool converged = false;
    if( comboType == AND ){
      converged = ( find(results.begin(),results.end(),false) == results.end() );
    } else 
    if( comboType == OR ){
      converged = ( find(results.begin(),results.end(),true) != results.end() );
    }
    if(writeToCout){
      if(converged){
        if( iter < minIterations )
          std::cout << "Converged, but continuing because min iterations not reached." << std::endl;
        else 
          std::cout << "Converged!" << std::endl;
      } else
        std::cout << "Not converged." << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }

    if( iter < minIterations ) converged = false;

    // check iteration limit
    if( iter >= maxIterations && !converged ){
      converged = true;
      if(writeToCout){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "************************************************************************" << std::endl;
        std::cout << "**********  Not converged.  Exiting due to iteration limit.  ***********" << std::endl;
        std::cout << "************************************************************************" << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }

    return converged;
}

/******************************************************************************/
ConvergenceTest::ConvergenceTest(const Teuchos::ParameterList& convParams)
/******************************************************************************/
{

  if( convParams.isType<int>("Minimum Iterations") )
    minIterations = convParams.get<int>("Minimum Iterations");
  else minIterations = 0;

  if( convParams.isType<int>("Maximum Iterations") ){
    maxIterations = convParams.get<int>("Maximum Iterations");
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, 
      std::endl << "Optimization convergence:  'Maximum Iterations' parameter is required." << std::endl);

  if( convParams.isType<std::string>("Combo Type") ){
    std::string combo = convParams.get<std::string>("Combo Type");
    std::transform(combo.begin(), combo.end(), combo.begin(), ::tolower);
    if(combo == "or"){
      comboType = OR;
    }else
    if(combo == "and"){
      comboType = AND;
    }else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, 
        std::endl << "Optimization convergence:  Unknown 'Combo Type'.  Options are ('AND', 'OR') " << std::endl);
    }
  } else comboType = OR;

  if( convParams.isType<double>("Relative Topology Change") ){
    double conValue = convParams.get<double>("Relative Topology Change");
    conTests.push_back( Teuchos::rcp(new RelDeltaP(conValue)) );
  }
  if( convParams.isType<double>("Absolute Topology Change") ){
    double conValue = convParams.get<double>("Absolute Topology Change");
    conTests.push_back( Teuchos::rcp(new AbsDeltaP(conValue)) );
  }
  if( convParams.isType<double>("Relative Objective Change") ){
    double conValue = convParams.get<double>("Relative Objective Change");
    conTests.push_back( Teuchos::rcp(new RelDeltaF(conValue)) );
  }
  if( convParams.isType<double>("Absolute Objective Change") ){
    double conValue = convParams.get<double>("Absolute Objective Change");
    conTests.push_back( Teuchos::rcp(new AbsDeltaF(conValue)) );
  }
  if( convParams.isType<double>("Relative Objective Running Average Change") ){
    double conValue = convParams.get<double>("Relative Objective Running Average Change");
    conTests.push_back( Teuchos::rcp(new RelRunningDF(conValue)) );
  }
  if( convParams.isType<double>("Absolute Objective Running Average Change") ){
    double conValue = convParams.get<double>("Absolute Objective Running Average Change");
    conTests.push_back( Teuchos::rcp(new AbsRunningDF(conValue)) );
  }
}

/******************************************************************************/
bool ConvergenceTest::AbsDeltaP::passed(double delta_f, double delta_p, bool write)
{ 
  bool status = ( fabs(delta_p) < conValue );
  if( write )
    std::cout << "Test: Topology Change (Absolute): " << std::endl 
    << "     abs(dp) = " << fabs(delta_p) << " < " << conValue << ": " 
    << (status ? "true" : "false") << std::endl;
  return status;
}
bool ConvergenceTest::AbsDeltaF::passed(double delta_f, double delta_p, bool write)
{
  bool status = ( fabs(delta_f) < conValue );
  if( write )
    std::cout << "Test: Objective Change (Absolute): " << std::endl 
    << "     abs(df) = " << fabs(delta_f) << " < " << conValue << ": " 
    << (status ? "true" : "false") << std::endl;
  return status;
}
bool ConvergenceTest::AbsRunningDF::passed(double delta_f, double delta_p, bool write)
{
  dF.push_back(delta_f);
  runningDF += delta_f;
  if(dF.size()>nave) runningDF -= *(dF.end()-nave);
  bool status = ( runningDF < conValue );
  if( write )
    std::cout << "Test: Objective Change Running Average (Absolute): " << std::endl 
    << "     abs(<df>) = " << runningDF << " < " << conValue << ": " 
    << (status ? "true" : "false") << std::endl;
  return status;
}
bool ConvergenceTest::RelDeltaP::passed(double delta_f, double delta_p, bool write){
  bool status = (p0 != 0.0) ? ( fabs(delta_p/p0) < conValue ) : false;
  if( write )
    std::cout << "Test: Topology Change (Relative): " << std::endl 
    << "     abs(dp) = " << fabs(delta_p) << ", fabs(dp/p0) = " << fabs(delta_p/p0) << " < " << conValue 
    << ": " << (status ? "true" : "false") << std::endl;
  return status;
}
bool ConvergenceTest::RelDeltaF::passed(double delta_f, double delta_p, bool write){
  bool status = (f0 != 0.0) ? ( fabs(delta_f/f0) < conValue ) : false;
  if( write )
    std::cout << "Test: Objective Change (Relative): " << std::endl 
    << "     abs(df) = " << fabs(delta_f) << ", fabs(df/f0) = " << fabs(delta_f/f0) << " < " << conValue 
    << ": " << (status ? "true" : "false") << std::endl;
  return status;
}
bool ConvergenceTest::RelRunningDF::passed(double delta_f, double delta_p, bool write){
  dF.push_back(delta_f);
  runningDF += delta_f;
  int nvals = dF.size();
  int lastVal = nvals-1;
  if(nvals>nave){
     runningDF -= dF[lastVal-nave];
     nvals = nave;
   }
  bool status = (f0 != 0.0) ? ( fabs(runningDF/f0)/nvals < conValue ) : false;
  if( write )
    std::cout << "Test: Objective Change Running Average (Relative): " << std::endl 
    << "     abs(<df>) = " << fabs(runningDF)/nvals << ", fabs(<df/f0>) = " << fabs(runningDF/f0)/nvals << " < " << conValue 
    << ": " << (status ? "true" : "false") << std::endl;
  return status;
}
/******************************************************************************/


/******************************************************************************/
void
Optimizer_OC::computeUpdatedTopology()
/******************************************************************************/
{

  // find multiplier that enforces measure constraint
  Teuchos::Array<double> upperBound, lowerBound;
  solverInterface->getOptDofsUpperBound(upperBound);
  solverInterface->getOptDofsLowerBound(lowerBound);
  double vmid, v1=0.0, v2=0.0;
  double residRatio = 0.0;
  int niters=0;

  double dfdp_tot = 0.0, dmdp_tot = 0.0;
  for(int i=0; i<numOptDofs; i++) {
    dfdp_tot += dfdp[i];
    dmdp_tot += dmdp[i];
  }
  double g_dfdp_tot = 0.0, g_dmdp_tot = 0.0;
  comm->SumAll(&dfdp_tot, &g_dfdp_tot, 1);
  comm->SumAll(&dmdp_tot, &g_dmdp_tot, 1);

  v2 = -10.0* g_dfdp_tot / g_dmdp_tot;

  if(comm->MyPID()==0){
    std::cout << "Measure enforcement: Target = " << _measureConstraint <<  std::endl;
    std::cout << "Measure enforcement: Beginning search with recursive bisection." <<  std::endl;
  }

  bool converged = false;
  double measure = 0.0;
  do {
    measure = 0.0;
    vmid = (v2+v1)/2.0;

    // update topology
    for(int i=0; i<numOptDofs; i++) {
      double be = 0.0;
      if( dmdp[i] != 0.0 )
//        be = -dfdp[i]/fabs(dmdp[i])/vmid;
        be = -dfdp[i]/dmdp[i]/vmid;
      else
        be = -dfdp[i]/vmid;
      double p_old = p_last[i];
//      double p_new = (p_old-offset)*pow(be,_stabExponent)+offset;
      double p_new = p_old*pow(be,_stabExponent);
      // limit change
      double dval = p_new - p_old;
      if( fabs(dval) > _moveLimit) p_new = p_old+fabs(dval)/dval*_moveLimit;
      // enforce limits
      if( p_new < lowerBound[i] ) p_new = lowerBound[i];
      if( p_new > upperBound[i] ) p_new = upperBound[i];
      p[i] = p_new;
    }

    // compute new measure
    if( _useNewtonSearch ){
      double prevResidual = measure - _measureConstraint*_optMeasure;
      solverInterface->ComputeMeasure(_measureType, p, measure);
      double newResidual = measure - _measureConstraint*_optMeasure;
      if( newResidual > 0.0 ){
        residRatio = newResidual/prevResidual;
        v1 = vmid;
        niters++;
        break;
      } else v2 = vmid;
    } else {
      solverInterface->ComputeMeasure(_measureType, p, measure);
      double newResidual = measure - _measureConstraint*_optMeasure;
      if( newResidual > 0.0 ){
        v1 = vmid;
      } else v2 = vmid;
    }
    niters++;

    if(comm->MyPID()==0){
      double resid = (measure - _measureConstraint*_optMeasure)/_optMeasure;
      std::cout << "Measure enforcement (iteration " << niters << "): Residual = " << resid << std::endl;
    }

  } while ( niters < _measureMaxIter && fabs(measure - _measureConstraint*_optMeasure) > _measureConvTol*_optMeasure );


  if(_useNewtonSearch){

  if(comm->MyPID()==0){
    std::cout << "Measure enforcement: Bounds found.  Switching to Newton search." << std::endl;
  }

  int newtonMaxIters = niters + 10;
  double lambda = (residRatio*v2 - v1)/(residRatio-1.0);
  double epsilon = lambda*1e-5;
  if( lambda > 0.0 ) do {
    for(int i=0; i<numOptDofs; i++) {
      double be = 0.0;
      if( dmdp[i] != 0.0 )
//        be = -dfdp[i]/fabs(dmdp[i])/vmid;
        be = -dfdp[i]/dmdp[i]/vmid;
      else
        be = -dfdp[i]/vmid;
      double p_old = p_last[i];
      double p_new = p_old*pow(be,_stabExponent);
      // limit change
      double dval = p_new - p_old;
      if( fabs(dval) > _moveLimit) p_new = p_old+fabs(dval)/dval*_moveLimit;
      // enforce limits
      if( p_new < lowerBound[i] ) p_new = lowerBound[i];
      if( p_new > upperBound[i] ) p_new = upperBound[i];
      p[i] = p_new;
    }
    // compute new measure
    solverInterface->ComputeMeasure(_measureType, p, measure);
    double f0 =  (measure - _measureConstraint*_optMeasure);

    if(comm->MyPID()==0){
      std::cout << "Measure Enforcement (iteration " << niters << "): Residual = " << f0/_optMeasure << std::endl;
    }

    if( fabs(f0) < _measureConvTol*_optMeasure ){
      converged = true;
      break;
    }

    double plambda = lambda+epsilon;
    for(int i=0; i<numOptDofs; i++) {
      double be = 0.0;
      if( dmdp[i] != 0.0 )
//        be = -dfdp[i]/fabs(dmdp[i])/vmid;
        be = -dfdp[i]/dmdp[i]/vmid;
      else
        be = -dfdp[i]/vmid;
      double p_old = p_last[i];
//      double p_new = (p_old-offset)*pow(be,_stabExponent)+offset;
      double p_new = p_old*pow(be,_stabExponent);
      // limit change
      double dval = p_new - p_old;
      if( fabs(dval) > _moveLimit) p_new = p_old+fabs(dval)/dval*_moveLimit;
      // enforce limits
      if( p_new < lowerBound[i] ) p_new = lowerBound[i];
      if( p_new > upperBound[i] ) p_new = upperBound[i];
      p[i] = p_new;
    }
    // compute new measure
    solverInterface->ComputeMeasure(_measureType, p, measure);
    double f1 =  (measure - _measureConstraint*_optMeasure);

    if( f1-f0 == 0.0 ) break;
    lambda -= epsilon*f0/(f1-f0);

    niters++;
  } while ( niters < newtonMaxIters );

  if(!converged){
    if(comm->MyPID()==0){
      std::cout << "Measure enforcement: Newton search failed.  Switching back to recursive bisection." << std::endl;
    }
  
    niters = 0;
    do {
      measure = 0.0;
      vmid = (v2+v1)/2.0;
  
      // update topology
      for(int i=0; i<numOptDofs; i++) {
        double be = 0.0;
        if( dmdp[i] != 0.0 )
//          be = -dfdp[i]/fabs(dmdp[i])/vmid;
          be = -dfdp[i]/dmdp[i]/vmid;
        else
          be = -dfdp[i]/vmid;
        double p_old = p_last[i];
//        double p_new = (p_old-offset)*pow(be,_stabExponent)+offset;
        double p_new = p_old*pow(be,_stabExponent);
        // limit change
        double dval = p_new - p_old;
        if( fabs(dval) > _moveLimit) p_new = p_old+fabs(dval)/dval*_moveLimit;
        // enforce limits
        if( p_new < lowerBound[i] ) p_new = lowerBound[i];
        if( p_new > upperBound[i] ) p_new = upperBound[i];
        p[i] = p_new;
      }
  
      // compute new measure
      solverInterface->ComputeMeasure(_measureType, p, measure);
      double newResidual = measure - _measureConstraint*_optMeasure;
      if( newResidual > 0.0 ){
        v1 = vmid;
      } else v2 = vmid;
      niters++;
  
      if(comm->MyPID()==0){
        double resid = (measure - _measureConstraint*_optMeasure)/_optMeasure;
        std::cout << "Measure enforcement (iteration " << niters << "): Residual = " << resid << std::endl;
      }
    
    } while ( niters < _measureMaxIter && fabs(measure - _measureConstraint*_optMeasure) > _measureConvTol*_optMeasure );
  }

  }

  TEUCHOS_TEST_FOR_EXCEPTION(
    ( fabs(measure - _measureConstraint*_optMeasure) > _measureAccpTol*_optMeasure ),
    Teuchos::Exceptions::InvalidParameter, 
    std::endl << "Enforcement of measure constraint failed:  Exceeded max iterations" 
    << std::endl);



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
 
  Teuchos::Array<double> upperBound, lowerBound;
  solverInterface->getOptDofsUpperBound(upperBound);
  solverInterface->getOptDofsLowerBound(lowerBound);
  
  if( _optMethod == "MMA" )
    opt = nlopt_create(NLOPT_LD_MMA, numOptDofs);
  else
  if( _optMethod == "CCSA" )
    opt = nlopt_create(NLOPT_LD_CCSAQ, numOptDofs);
  else
  if( _optMethod == "SLSQP" )
    opt = nlopt_create(NLOPT_LD_SLSQP, numOptDofs);
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Optimization method: " << _optMethod << " Unknown!" << std::endl 
      << "Valid options are (MMA, CCSA, SLSQP)" << std::endl);

  
  // set bounds
  nlopt_set_lower_bounds(opt, lowerBound.getRawPtr());
  nlopt_set_upper_bounds(opt, upperBound.getRawPtr());

  // set objective function
  nlopt_set_min_objective(opt, this->evaluate, this);
  
  // set stop criteria
  nlopt_set_xtol_rel(opt, 1e-9);  // don't converge based on this.  I.e., use convergenceChecker.
  nlopt_set_maxeval(opt, convergenceChecker->getMaxIterations());

  p      = new double[numOptDofs];
  p_last = new double[numOptDofs];
  x_ref  = new double[numOptDofs];
  dfdp   = new double[numOptDofs];
  dgdp   = new double[numOptDofs];
  dmdp   = new double[numOptDofs];

  std::fill_n(dfdp,   numOptDofs, 0.0);
  std::fill_n(dgdp,   numOptDofs, 0.0);
  std::fill_n(dmdp,   numOptDofs, 0.0);
  std::fill_n(p,      numOptDofs, 0.0);
  std::fill_n(p_last, numOptDofs, 0.0);
  std::fill_n(x_ref,  numOptDofs, 0.0);

  solverInterface->InitializeOptDofs(p);

  if( primaryConstraintType == ResponseType::Measure )
    nlopt_add_inequality_constraint(opt, this->constraint, this, _measureConvTol*_optMeasure);
  else
  if( primaryConstraintType == ResponseType::Aggregate )
    nlopt_add_inequality_constraint(opt, this->constraint, this, _conConvTol);
}
#define ATO_XTOL_REACHED 104

/******************************************************************************/
void
Optimizer_NLopt::Optimize()
/******************************************************************************/
{
  double* dfdp_init = new double[numOptDofs];
  solverInterface->ComputeObjective(p, f, dfdp_init);
  delete [] dfdp_init;
  double global_f=0.0, pnorm = computeNorm(p, numOptDofs);
  comm->SumAll(&f, &global_f, 1);
  convergenceChecker->initNorm(global_f, pnorm);

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
   
  bool changed = isChanged(x);

  if( changed ){
    objectiveValue_last = objectiveValue;
    std::memcpy((void*)p_last, (void*)x, numOptDofs*sizeof(double));
    solverInterface->ComputeMeasure(_measureType, x, measure, dmdp);
    solverInterface->Compute(x, f, dfdp, g, dgdp);
  }

  if( objectiveType == ResponseType::Measure ){
    std::memcpy((void*)grad, (void*)dmdp, numOptDofs*sizeof(double));
    objectiveValue = measure;
  } else
  if( objectiveType == ResponseType::Aggregate ){
    std::memcpy((void*)grad, (void*)dfdp, numOptDofs*sizeof(double));
    objectiveValue = f;
  }

  if(comm->MyPID()==0){
    std::cout << "************************************************************************" << std::endl;
    std::cout << "  Optimizer:     measure: " << measure << std::endl;
    std::cout << "  Optimizer:   objective: " << objectiveValue << std::endl;
    std::cout << "  Optimizer:  constraint: " << g << std::endl;
    std::cout << "************************************************************************" << std::endl;
  }

  double delta_objective, ldelta_objective = objectiveValue-objectiveValue_last;
  comm->SumAll(&ldelta_objective, &delta_objective, 1);
  double delta_p = computeDiffNorm(x, p_last, numOptDofs, /*result to cout*/ false);

  if( convergenceChecker->isConverged(delta_objective, delta_p, _nIterations , comm->MyPID()) ){
    nlopt_set_force_stop(opt, ATO_XTOL_REACHED);
    nlopt_force_stop(opt);
  }
  _nIterations++;

  return objectiveValue;
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
  bool changed = isChanged(x);

  if( primaryConstraintType == ResponseType::Measure ){
    if( changed ) solverInterface->ComputeMeasure(_measureType, x, measure, dmdp);
    std::memcpy((void*)grad, (void*)dmdp, numOptDofs*sizeof(double));
    constraintValue = measure - _measureConstraint*_optMeasure;
    if(comm->MyPID()==0){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Optimizer:  computed measure is: " << measure << std::endl;
      std::cout << "  Optimizer:    target measure is: " << _measureConstraint*_optMeasure << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  } else
  if( primaryConstraintType == ResponseType::Aggregate ){
    if( changed ) solverInterface->Compute(x, f, dfdp, g, dgdp);
    std::memcpy((void*)grad, (void*)dgdp, numOptDofs*sizeof(double));
    constraintValue = g;
    if(comm->MyPID()==0){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Optimizer:  computed constraint is: " << constraintValue << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }

  return constraintValue;
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

/******************************************************************************/
bool
Optimizer_NLopt::isChanged(const double* x)
/******************************************************************************/
{
  for(int i=0; i<numOptDofs; i++){
    if( x[i] != x_ref[i] ){
      std::memcpy((void*)x_ref, (void*)x, numOptDofs*sizeof(double));
      return true;
    }
  }
  return false;
}
#endif //ATO_USES_NLOPT


}

