//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Aggregator.hpp"
#include "ATO_Solver.hpp"
#include "Teuchos_TestForException.hpp"

namespace ATO {


//**********************************************************************
Teuchos::RCP<Aggregator> 
AggregatorFactory::create(const Teuchos::ParameterList& aggregatorParams, std::string entityType)
{
  Teuchos::Array<std::string> objectives = 
    aggregatorParams.get<Teuchos::Array<std::string> >("Objectives");

  if( entityType == "State Variable" ){
    std::string weightingType = aggregatorParams.get<std::string>("Weighting");
    if( weightingType == "Scaled"  )  
      return Teuchos::rcp(new Aggregator_Scaled(aggregatorParams));
    else
      return Teuchos::rcp(new Aggregator_Uniform(aggregatorParams));
  } else
  if( entityType == "Distributed Parameter" ){
    std::string weightingType = aggregatorParams.get<std::string>("Weighting");
    if( weightingType == "Scaled"  )  
      return Teuchos::rcp(new Aggregator_DistScaled(aggregatorParams));
    else
      return Teuchos::rcp(new Aggregator_DistUniform(aggregatorParams));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Unknown 'Entity Type' requested." << std::endl);
  }
}

//**********************************************************************
Aggregator::Aggregator(const Teuchos::ParameterList& aggregatorParams)
//**********************************************************************
{ 
  parse(aggregatorParams);
}

//**********************************************************************
void 
Aggregator::parse(const Teuchos::ParameterList& aggregatorParams)
//**********************************************************************
{
  aggregatedObjectivesNames = 
    aggregatorParams.get<Teuchos::Array<std::string> >("Objectives");
  aggregatedDerivativesNames = 
    aggregatorParams.get<Teuchos::Array<std::string> >("Objective Derivatives");
  TEUCHOS_TEST_FOR_EXCEPTION(
    aggregatedObjectivesNames.size() != aggregatedDerivativesNames.size(),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  Number of objectives != number of derivatives." << std::endl
    << "        Check objective aggregator input." << std::endl);
  
  outputObjectiveName = aggregatorParams.get<std::string>("Objective Name");
  outputDerivativeName = aggregatorParams.get<std::string>("dFdTopology Name");


  if( aggregatorParams.isType<bool>("Normalize") ){
    if( aggregatorParams.get<bool>("Normalize") == false){
      int nObjs = aggregatedObjectivesNames.size();
      normalize.resize(nObjs,1.0);
    }
  }

  comm = Teuchos::null;
}

//**********************************************************************
void 
Aggregator_DistParamBased::
SetInputVariables(const std::vector<SolverSubSolver>& subProblems,
                  const std::map<std::string, Teuchos::RCP<const Epetra_Vector> > gMap,
                  const std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > dgdpMap)
//**********************************************************************
{


  outApp = subProblems[0].app;

  // loop through sub variable names and find the containing state manager
  int numVars = aggregatedObjectivesNames.size();
  objectives.resize(numVars);
  derivatives.resize(numVars);

  std::map<std::string, Teuchos::RCP<const Epetra_Vector> >::const_iterator git;
  std::map<std::string, Teuchos::RCP<Epetra_MultiVector> >::const_iterator gpit;
  for(int ir=0; ir<numVars; ir++){
    git = gMap.find(aggregatedObjectivesNames[ir]);
    gpit = dgdpMap.find(aggregatedDerivativesNames[ir]);
    TEUCHOS_TEST_FOR_EXCEPTION(
      git == gMap.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Aggregator: Requested response (" << aggregatedObjectivesNames[ir] 
      << ") not defined." << std::endl);
    TEUCHOS_TEST_FOR_EXCEPTION(
      gpit == dgdpMap.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Aggregator: Requested response derivative (" << aggregatedDerivativesNames[ir] 
      << ") not defined." << std::endl);
    objectives[ir].name = git->first;
    objectives[ir].value = git->second;
    derivatives[ir].name = gpit->first;
    derivatives[ir].value = gpit->second;
  }
}


//**********************************************************************
void
Aggregator_StateVarBased::SetInputVariables(const std::vector<SolverSubSolver>& subProblems)
//**********************************************************************
{
  outApp = subProblems[0].app;

  // loop through sub variable names and find the containing state manager
  int numVars = aggregatedObjectivesNames.size();
  objectives.resize(numVars);
  derivatives.resize(numVars);

  int numSubs = subProblems.size();
  for(int iv=0; iv<numVars; iv++){
    bool derFound = false;
    bool objFound = false;
    std::string& objName = aggregatedObjectivesNames[iv];
    std::string& derName = aggregatedDerivativesNames[iv];
    for(int is=0; is<numSubs; is++){
      const Teuchos::RCP<Albany::Application>& app = subProblems[is].app;
      Albany::StateArray& src = app->getStateMgr().getStateArrays().elemStateArrays[0];
      if(src.count(objName) > 0){
        TEUCHOS_TEST_FOR_EXCEPTION(
          objFound, Teuchos::Exceptions::InvalidParameter, std::endl 
          << "Objective '" << objName << "' found in two state managers." << std::endl
          << "Objective names must be unique to avoid ambiguity." << std::endl);
        objectives[iv].name = objName;
        objectives[iv].app = app;
        objFound = true;
      }
      if(src.count(derName) > 0){
        TEUCHOS_TEST_FOR_EXCEPTION(
          derFound, Teuchos::Exceptions::InvalidParameter, std::endl 
          << "Derivative '" << derName << "' found in two state managers." << std::endl
          << "Derivative names must be unique to avoid ambiguity." << std::endl);
        derivatives[iv].name = derName;
        derivatives[iv].app = app;
        derFound = true;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(
      !derFound, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Derivative '" << derName << "' not found in any state manager." << std::endl);
    TEUCHOS_TEST_FOR_EXCEPTION(
      !objFound, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Objective '" << objName << "' not found in any state manager." << std::endl);
  }
}

//**********************************************************************
Aggregator_Uniform::Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams) :
Aggregator(aggregatorParams),
Aggregator_StateVarBased()
//**********************************************************************
{ 
  int nAgg = aggregatedObjectivesNames.size();
  double weight = 1.0/nAgg;
  weights.resize(nAgg);
  for(int i=0; i<nAgg; i++) weights[i] = weight;
}


//**********************************************************************
Aggregator_DistUniform::Aggregator_DistUniform(const Teuchos::ParameterList& aggregatorParams) :
Aggregator(aggregatorParams)
//**********************************************************************
{ 
  int nAgg = aggregatedObjectivesNames.size();
  double weight = 1.0/nAgg;
  weights.resize(nAgg);
  for(int i=0; i<nAgg; i++) weights[i] = weight;
}

//**********************************************************************
Aggregator_Scaled::Aggregator_Scaled(const Teuchos::ParameterList& aggregatorParams) :
Aggregator(aggregatorParams)
//**********************************************************************
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(
    !aggregatorParams.isType<Teuchos::Array<double> >("Weights"),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator requires weights.  None given." << std::endl );

  weights = aggregatorParams.get<Teuchos::Array<double> >("Weights");

  TEUCHOS_TEST_FOR_EXCEPTION(
    weights.size() != aggregatedObjectivesNames.size(),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator requires weights.  None given." << std::endl );

}
//**********************************************************************
Aggregator_DistScaled::Aggregator_DistScaled(const Teuchos::ParameterList& aggregatorParams) :
Aggregator(aggregatorParams)
//**********************************************************************
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(
    !aggregatorParams.isType<Teuchos::Array<double> >("Weights"),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator requires weights.  None given." << std::endl );

  weights = aggregatorParams.get<Teuchos::Array<double> >("Weights");

  TEUCHOS_TEST_FOR_EXCEPTION(
    weights.size() != aggregatedObjectivesNames.size(),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator requires weights.  None given." << std::endl );

}


//**********************************************************************
void
Aggregator_Scaled::Evaluate()
//**********************************************************************
{
  int numVariables = derivatives.size();

  dgdpAggregated->PutScalar(0.0);
  *gAggregated=0.0;
 
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();

  int nObjs = objectives.size();
  if(normalize.size() == 0){
    normalize.resize(nObjs);
    for(int i=0; i<nObjs; i++){
      Albany::StateArrayVec& src = derivatives[i].app->getStateMgr().getStateArrays().elemStateArrays;
      Albany::MDArray& objSrc = src[0][objectives[i].name];
      double val = objSrc(0);
      double globalVal = val;
      if( comm != Teuchos::null )
        comm->SumAll(&val, &globalVal, /*numvals=*/ 1);
      normalize[i] = (globalVal != 0.0) ? 1.0/fabs(globalVal) : 1.0;
    }
  }

  for(int sv=0; sv<numVariables; sv++){
    Albany::StateArrayVec& src = derivatives[sv].app->getStateMgr().getStateArrays().elemStateArrays;
    int numWorksets = src.size();
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& derSrc = src[ws][derivatives[sv].name];
      int numCells = derSrc.dimension(0);
      int numNodes = derSrc.dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++)
          dgdpAggregated->SumIntoGlobalValue(wsElNodeID[ws][cell][node], 0, normalize[sv]*weights[sv]*derSrc(cell,node));
    }

    Albany::MDArray& objSrc = src[0][objectives[sv].name];
    if( comm != Teuchos::null ){
      double globalVal, val = objSrc(0);
      comm->SumAll(&val, &globalVal, /*numvals=*/ 1);
      if( comm->MyPID()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: " << objectives[sv].name << " = " << globalVal << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }

    *gAggregated += normalize[sv]*weights[sv]*objSrc(0);
    objSrc(0)=0.0;
  }

  if( comm != Teuchos::null ){
    if( comm->MyPID()==0 ){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: " << outputObjectiveName << " = " << *gAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }
}

//**********************************************************************
void
Aggregator_DistScaled::Evaluate()
//**********************************************************************
{
  dgdpAggregated->PutScalar(0.0);
  double *derDest; dgdpAggregated->ExtractView(&derDest);
  int nLocalVals = dgdpAggregated->MyLength();
  *gAggregated = 0.0;

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();

  int nObjs = objectives.size();
  if(normalize.size() == 0){
    normalize.resize(nObjs);
    for(int i=0; i<nObjs; i++){
      double* objView; objectives[i].value->ExtractView(&objView);
      normalize[i] = (objView[0] != 0.0) ? 1.0/fabs(objView[0]) : 1.0;
    }
  }

  for(int i=0; i<objectives.size(); i++){
    SubObjective& objective = objectives[i];
    SubDerivative& derivative = derivatives[i];

    const Epetra_BlockMap& srcMap = derivative.value->Map();
    double* srcView; (*derivative.value)(0)->ExtractView(&srcView);

    double* objView; objective.value->ExtractView(&objView);
    *gAggregated += objView[0]*normalize[i]*weights[i];

    for(int lid=0; lid<nLocalVals; lid++)
      derDest[lid] += srcView[lid]*normalize[i]*weights[i];

    if( comm != Teuchos::null ){
      if( comm->MyPID()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: Input variable " << i << std::endl;
        std::cout << "   " << objective.name << " = " << objView[0] << std::endl;
        std::cout << "   " << objective.name << " (scaled) = " << objView[0]*normalize[i] << std::endl;
        std::cout << "   Weight = " << weights[i] << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }
  if( comm != Teuchos::null ){
    if( comm->MyPID()==0 ){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: Output " << std::endl;
      std::cout << "   Objective = " << *gAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }
}
}
