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
    if (objectives.size() == 1) 
     return Teuchos::rcp(new Aggregator_PassThru(aggregatorParams));
  
    std::string weightingType = aggregatorParams.get<std::string>("Weighting");
    if( weightingType == "Uniform"  )  
      return Teuchos::rcp(new Aggregator_Uniform(aggregatorParams));
    else
    if( weightingType == "Scaled"  )  
      return Teuchos::rcp(new Aggregator_Scaled(aggregatorParams));
    else
      TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter, std::endl 
        << "Error!  Weighting type " << weightingType << " Unknown!" << std::endl 
        << "Valid weighting types are (Uniform, Scaled)" << std::endl);
  } else
  if( entityType == "Distributed Parameter" ){
    if (objectives.size() == 1) 
     return Teuchos::rcp(new Aggregator_DistSingle(aggregatorParams));
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
}

//**********************************************************************
Aggregator_DistSingle::Aggregator_DistSingle(const Teuchos::ParameterList& aggregatorParams) :
Aggregator(aggregatorParams)
//**********************************************************************
{ 
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

  if(aggregatorParams.isType<bool>("Shift to Scaled"))
    shiftToZero = aggregatorParams.get<bool>("Shift to Scaled");
  else 
    shiftToZero = false;
  
}

//**********************************************************************
void
Aggregator_Uniform::Evaluate()
//**********************************************************************
{

  Albany::StateArrays& stateArrays = outApp->getStateMgr().getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  int numWorksets = dest.size();
  
  // uniform weighting
  int numVariables = derivatives.size();
  double weight = 1.0/numVariables;

  // zero out the destination variable
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& derDest = dest[ws][outputDerivativeName];
    int dim0 = derDest.dimension(0);
    int dim1 = derDest.dimension(1);
    for(int i=0; i<dim0; i++)
      for(int j=0; j<dim1; j++)
        derDest(i,j)=0.0;
  }
  Albany::MDArray& objDest = dest[0][outputObjectiveName];
  objDest(0)=0.0;

  // dest = (Var1 + Var2 + ...) / nvars
  for(int sv=0; sv<numVariables; sv++){
    Albany::StateArrays& inStateArrays = derivatives[sv].app->getStateMgr().getStateArrays();
    Albany::StateArrayVec& src = inStateArrays.elemStateArrays;
    std::string derName = derivatives[sv].name;
    std::string objName = objectives[sv].name;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& derSrc = src[ws][derName];
      Albany::MDArray& derDest = dest[ws][outputDerivativeName];
      int dim0 = derSrc.dimension(0);
      int dim1 = derSrc.dimension(1);
      for(int i=0; i<dim0; i++)
        for(int j=0; j<dim1; j++)
          derDest(i,j) += weight*derSrc(i,j);
    }
    Albany::MDArray& objSrc = src[0][objName];


    if( comm != Teuchos::null ){
      double globalVal, val = objSrc(0);
      comm->SumAll(&val, &globalVal, /*numvals=*/ 1);
      if( comm->MyPID()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: " << objName << " = " << globalVal << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }

    Albany::MDArray& objDest = dest[0][outputObjectiveName];
    objDest(0) += weight*objSrc(0);
    // reset sources to zero.  this shouldn't be done here.
    objSrc(0)=0.0;
  }

  if( comm != Teuchos::null ){
    double globalVal, val = objDest(0);
    comm->SumAll(&val, &globalVal, /*numvals=*/ 1);
    if( comm->MyPID()==0 ){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: " << outputObjectiveName << " = " << globalVal << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }
}


//**********************************************************************
void
Aggregator_Scaled::Evaluate()
//**********************************************************************
{

  Albany::StateArrays& stateArrays = outApp->getStateMgr().getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  int numWorksets = dest.size();
  
  int numVariables = derivatives.size();

  // zero out the destination variable
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& derDest = dest[ws][outputDerivativeName];
    int dim0 = derDest.dimension(0);
    int dim1 = derDest.dimension(1);
    for(int i=0; i<dim0; i++)
      for(int j=0; j<dim1; j++)
        derDest(i,j)=0.0;
  }
  Albany::MDArray& objDest = dest[0][outputObjectiveName];
  objDest(0)=0.0;

  // dest = (Var1 + Var2 + ...) / nvars
  for(int sv=0; sv<numVariables; sv++){
    Albany::StateArrays& inStateArrays = derivatives[sv].app->getStateMgr().getStateArrays();
    Albany::StateArrayVec& src = inStateArrays.elemStateArrays;
    std::string derName = derivatives[sv].name;
    std::string objName = objectives[sv].name;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& derSrc = src[ws][derName];
      Albany::MDArray& derDest = dest[ws][outputDerivativeName];
      int dim0 = derSrc.dimension(0);
      int dim1 = derSrc.dimension(1);
      for(int i=0; i<dim0; i++)
        for(int j=0; j<dim1; j++)
          derDest(i,j) += weights[sv]*derSrc(i,j);
    }
    Albany::MDArray& objSrc = src[0][objName];


    if( comm != Teuchos::null ){
      double globalVal, val = objSrc(0);
      comm->SumAll(&val, &globalVal, /*numvals=*/ 1);
      if( comm->MyPID()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: " << objName << " = " << globalVal << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }

    Albany::MDArray& objDest = dest[0][outputObjectiveName];
    objDest(0) += weights[sv]*objSrc(0);
    // reset sources to zero.  this shouldn't be done here.
    objSrc(0)=0.0;
  }

  if( comm != Teuchos::null ){
    double globalVal, val = objDest(0);
    comm->SumAll(&val, &globalVal, /*numvals=*/ 1);
    if( comm->MyPID()==0 ){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: " << outputObjectiveName << " = " << globalVal << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }

  if(shiftToZero){
    double maxValue = 0.0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& derDest = dest[ws][outputDerivativeName];
      int dim0 = derDest.dimension(0);
      int dim1 = derDest.dimension(1);
      for(int i=0; i<dim0; i++)
        for(int j=0; j<dim1; j++)
          if(derDest(i,j) > maxValue)
            maxValue = derDest(i,j);
    }
    if(maxValue > 0.0){
      for(int ws=0; ws<numWorksets; ws++){
        Albany::MDArray& derDest = dest[ws][outputDerivativeName];
        int dim0 = derDest.dimension(0);
        int dim1 = derDest.dimension(1);
        for(int i=0; i<dim0; i++)
          for(int j=0; j<dim1; j++)
            derDest(i,j) -= maxValue;
      }
    }
  }
}

//**********************************************************************
void
Aggregator_DistSingle::Evaluate()
//**********************************************************************
{

  Albany::StateArrays& stateArrays = outApp->getStateMgr().getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  int numWorksets = dest.size();
  
  // zero out the destination variable
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& derDest = dest[ws][outputDerivativeName];
    int dim0 = derDest.dimension(0);
    int dim1 = derDest.dimension(1);
    for(int i=0; i<dim0; i++)
      for(int j=0; j<dim1; j++)
        derDest(i,j)=0.0;
  }

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();

  SubObjective& objective = objectives[0];
  SubDerivative& derivative = derivatives[0];

  const Epetra_BlockMap& srcMap = derivative.value->Map();
  double* srcView; (*derivative.value)(0)->ExtractView(&srcView);
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& derDest = dest[ws][outputDerivativeName];
    int numCells = derDest.dimension(0);
    int numNodes = derDest.dimension(1);
    for(int cell=0; cell<numCells; cell++)
      for(int node=0; node<numNodes; node++){
        int gid = wsElNodeID[ws][cell][node];
        int lid = srcMap.LID(gid);
        if( lid >= 0 )
          derDest(cell,node) = srcView[lid];
        else derDest(cell,node) = 0.0;
      }
  }

  double* objView; objective.value->ExtractView(&objView);
  Albany::MDArray& objDest = dest[0][outputObjectiveName];
  objDest(0) = objView[0];

}


//**********************************************************************
Aggregator_PassThru::Aggregator_PassThru(const Teuchos::ParameterList& aggregatorParams) :
Aggregator(aggregatorParams)
//**********************************************************************
{ 
  outputObjectiveName = aggregatedObjectivesNames[0];
  outputDerivativeName = aggregatedDerivativesNames[0];
}


}

