//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Aggregator.hpp"
#include "ATO_Solver.hpp"
#include "Teuchos_TestForException.hpp"
#include <functional>

namespace ATO {


//**********************************************************************
Teuchos::RCP<Aggregator> 
AggregatorFactory::create(const Teuchos::ParameterList& aggregatorParams, std::string entityType, int nTopos)
{
  Teuchos::Array<std::string> values = 
    aggregatorParams.get<Teuchos::Array<std::string> >("Values");

  if( entityType == "State Variable" ){
    std::string weightingType = aggregatorParams.get<std::string>("Weighting");
    if( weightingType == "Scaled"  )  
      return Teuchos::rcp(new Aggregator_Scaled(aggregatorParams, nTopos));
    else
    if( weightingType == "Maximum"  )  
      return Teuchos::rcp(new Aggregator_Extremum<std::greater<double> >(aggregatorParams, nTopos));
    else
    if( weightingType == "Minimum"  )  
      return Teuchos::rcp(new Aggregator_Extremum<std::less<double> >(aggregatorParams, nTopos));
    else
      return Teuchos::rcp(new Aggregator_Uniform(aggregatorParams, nTopos));
  } else
  if( entityType == "Distributed Parameter" ){
    std::string weightingType = aggregatorParams.get<std::string>("Weighting");
    if( weightingType == "Scaled"  )  
      return Teuchos::rcp(new Aggregator_DistScaled(aggregatorParams, nTopos));
    else
    if( weightingType == "Maximum"  )  
      return Teuchos::rcp(new Aggregator_DistExtremum<std::greater<double> >(aggregatorParams, nTopos));
    else
    if( weightingType == "Minimum"  )  
      return Teuchos::rcp(new Aggregator_DistExtremum<std::less<double> >(aggregatorParams, nTopos));
    else
      return Teuchos::rcp(new Aggregator_DistUniform(aggregatorParams, nTopos));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Unknown 'Entity Type' requested." << std::endl);
  }
}

//**********************************************************************
Aggregator::Aggregator(const Teuchos::ParameterList& aggregatorParams, int nTopos) : numTopologies(nTopos)
//**********************************************************************
{ 
  parse(aggregatorParams);
}

//**********************************************************************
void 
Aggregator::parse(const Teuchos::ParameterList& aggregatorParams)
//**********************************************************************
{
  if(aggregatorParams.isType<Teuchos::Array<std::string> >("Values"))
    aggregatedValuesNames = aggregatorParams.get<Teuchos::Array<std::string> >("Values");

  if(aggregatorParams.isType<Teuchos::Array<std::string> >("Derivatives"))
    aggregatedDerivativesNames = aggregatorParams.get<Teuchos::Array<std::string> >("Derivatives");

  TEUCHOS_TEST_FOR_EXCEPTION(
    (aggregatedValuesNames.size() != 0) &&
    (aggregatedDerivativesNames.size() != 0) &&
    (aggregatedValuesNames.size() != aggregatedDerivativesNames.size()),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  Number of values != number of derivatives." << std::endl
    << "        Check value aggregator input." << std::endl);
  
  TEUCHOS_TEST_FOR_EXCEPTION(
    (aggregatedValuesNames.size() == 0) &&
    (aggregatedDerivativesNames.size() == 0),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  No values and no derivatives provided." << std::endl
    << "        Check value aggregator input." << std::endl);
  
  if( aggregatedValuesNames.size() > 0 )
    outputValueName = aggregatorParams.get<std::string>("Output Value Name");

  if( aggregatedDerivativesNames.size() > 0 )
    outputDerivativeName = aggregatorParams.get<std::string>("Output Derivative Name");


  if( aggregatorParams.isType<bool>("Normalize") ){
    if( aggregatorParams.get<bool>("Normalize") == false){
      int nObjs = aggregatedValuesNames.size();
      normalize.resize(nObjs,1.0);
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(
    (aggregatedValuesNames.size() == 0) &&
    (normalize.size() == 0),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  'Normalize' must be set to 'false' if only derivatives are being aggregated." << std::endl
    << "        Check value aggregator input." << std::endl);

  if( aggregatorParams.isType<double>("Shift Output") )
    shiftValueAggregated = aggregatorParams.get<double>("Shift Output");
  else
    shiftValueAggregated = 0.0;

  if( aggregatorParams.isType<double>("Scale Output") )
    scaleValueAggregated = aggregatorParams.get<double>("Scale Output");
  else
    scaleValueAggregated = 1.0;

  comm = Teuchos::null;
}

//**********************************************************************
void 
Aggregator_DistParamBased::
SetInputVariablesT(const std::vector<SolverSubSolver>& subProblems,
                   const std::map<std::string, Teuchos::RCP<const Tpetra_Vector> > valueMap,
                   const std::map<std::string, Teuchos::RCP<Tpetra_MultiVector> > derivMap)
//**********************************************************************
{
  outApp = subProblems[0].app;

  // loop through sub variable names and find the containing state manager
  int numVars = aggregatedValuesNames.size();
  valuesT.resize(numVars);

  std::map<std::string, Teuchos::RCP<const Tpetra_Vector> >::const_iterator git;
  for(int ir=0; ir<numVars; ir++){
    git = valueMap.find(aggregatedValuesNames[ir]);
    TEUCHOS_TEST_FOR_EXCEPTION(
      git == valueMap.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Aggregator: Requested response (" << aggregatedValuesNames[ir] 
      << ") not defined." << std::endl);
    valuesT[ir].name = git->first;
    valuesT[ir].value = git->second;
  }


  numVars = aggregatedDerivativesNames.size();
  derivativesT.resize(numVars);

  std::map<std::string, Teuchos::RCP<Tpetra_MultiVector> >::const_iterator gpit;
  for(int ir=0; ir<numVars; ir++){
    gpit = derivMap.find(aggregatedDerivativesNames[ir]);
    TEUCHOS_TEST_FOR_EXCEPTION(
      gpit == derivMap.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Aggregator: Requested response derivative (" << aggregatedDerivativesNames[ir] 
      << ") not defined." << std::endl);
    derivativesT[ir].name = gpit->first;
    derivativesT[ir].value = gpit->second;
  }
}

//**********************************************************************
void 
Aggregator_DistParamBased::
SetInputVariables(const std::vector<SolverSubSolver>& subProblems,
                  const std::map<std::string, Teuchos::RCP<const Epetra_Vector> > valueMap,
                  const std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > derivMap)
//**********************************************************************
{


  outApp = subProblems[0].app;

  // loop through sub variable names and find the containing state manager
  int numVars = aggregatedValuesNames.size();
  values.resize(numVars);

  std::map<std::string, Teuchos::RCP<const Epetra_Vector> >::const_iterator git;
  for(int ir=0; ir<numVars; ir++){
    git = valueMap.find(aggregatedValuesNames[ir]);
    TEUCHOS_TEST_FOR_EXCEPTION(
      git == valueMap.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Aggregator: Requested response (" << aggregatedValuesNames[ir] 
      << ") not defined." << std::endl);
    values[ir].name = git->first;
    values[ir].value = git->second;
  }


  numVars = aggregatedDerivativesNames.size();
  derivatives.resize(numVars);

  std::map<std::string, Teuchos::RCP<Epetra_MultiVector> >::const_iterator gpit;
  for(int ir=0; ir<numVars; ir++){
    gpit = derivMap.find(aggregatedDerivativesNames[ir]);
    TEUCHOS_TEST_FOR_EXCEPTION(
      gpit == derivMap.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Aggregator: Requested response derivative (" << aggregatedDerivativesNames[ir] 
      << ") not defined." << std::endl);
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
  int numVars = aggregatedValuesNames.size();
  values.resize(numVars);

  int numSubs = subProblems.size();
  for(int iv=0; iv<numVars; iv++){
    bool objFound = false;
    std::string& objName = aggregatedValuesNames[iv];
    for(int is=0; is<numSubs; is++){
      const Teuchos::RCP<Albany::Application>& app = subProblems[is].app;
      Albany::StateArray& src = app->getStateMgr().getStateArrays().elemStateArrays[0];
      if(src.count(objName) > 0){
        TEUCHOS_TEST_FOR_EXCEPTION(
          objFound, Teuchos::Exceptions::InvalidParameter, std::endl 
          << "Value '" << objName << "' found in two state managers." << std::endl
          << "Value names must be unique to avoid ambiguity." << std::endl);
        values[iv].name = objName;
        values[iv].app = app;
        objFound = true;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(
      !objFound, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Value '" << objName << "' not found in any state manager." << std::endl);
  }

  numVars = aggregatedDerivativesNames.size();
  derivatives.resize(numVars);
  for(int iv=0; iv<numVars; iv++){
    bool derFound = false;
    std::string derName = aggregatedDerivativesNames[iv];
    for(int is=0; is<numSubs; is++){
      const Teuchos::RCP<Albany::Application>& app = subProblems[is].app;
      Albany::StateArray& src = app->getStateMgr().getStateArrays().elemStateArrays[0];
      if(src.count(Albany::strint(derName,0)) > 0){
        TEUCHOS_TEST_FOR_EXCEPTION(
          derFound, Teuchos::Exceptions::InvalidParameter, std::endl 
          << "Derivative '" << derName << "' found in two state managers." << std::endl
          << "Derivative names must be unique to avoid ambiguity." << std::endl);
        derivatives[iv].name.resize(numTopologies);
        for(int itopo=0; itopo<numTopologies; itopo++)
          derivatives[iv].name[itopo] = Albany::strint(derName, itopo);
        derivatives[iv].app = app;
        derFound = true;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(
      !derFound, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Derivative '" << derName << "' not found in any state manager." << std::endl);
  }
}

//**********************************************************************
Aggregator_Uniform::Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams, int nTopos) :
Aggregator(aggregatorParams, nTopos),
Aggregator_StateVarBased()
//**********************************************************************
{ 
  int nAgg = aggregatedValuesNames.size();
  if(nAgg == 0) nAgg = aggregatedDerivativesNames.size();
  double weight = 1.0/nAgg;
  weights.resize(nAgg);
  for(int i=0; i<nAgg; i++) weights[i] = weight;
}


//**********************************************************************
Aggregator_DistUniform::Aggregator_DistUniform(const Teuchos::ParameterList& aggregatorParams, int nTopos) :
Aggregator(aggregatorParams, nTopos)
//**********************************************************************
{ 
  int nAgg = aggregatedValuesNames.size();
  if(nAgg == 0) nAgg = aggregatedDerivativesNames.size();
  double weight = 1.0/nAgg;
  weights.resize(nAgg);
  for(int i=0; i<nAgg; i++) weights[i] = weight;
}

//**********************************************************************
Aggregator_Scaled::Aggregator_Scaled(const Teuchos::ParameterList& aggregatorParams, int nTopos) :
Aggregator(aggregatorParams, nTopos)
//**********************************************************************
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(
    !aggregatorParams.isType<Teuchos::Array<double> >("Weights"),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator requires weights.  None given." << std::endl );

  weights = aggregatorParams.get<Teuchos::Array<double> >("Weights");

  TEUCHOS_TEST_FOR_EXCEPTION(
    weights.size() != aggregatedValuesNames.size() &&
    weights.size() != aggregatedDerivativesNames.size(),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator: Number of weights != number of values or derivatives." << std::endl );
}

//**********************************************************************
template <typename C>
Aggregator_Extremum<C>::Aggregator_Extremum(const Teuchos::ParameterList& aggregatorParams, int nTopos) :
Aggregator(aggregatorParams, nTopos)
//**********************************************************************
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    aggregatedValuesNames.size() == 0 &&
    aggregatedDerivativesNames.size() > 0,
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Extremum aggregator: Minimum/Maximum weighting requires values (only derivatives provided)." << std::endl );
}

//**********************************************************************
Aggregator_DistScaled::Aggregator_DistScaled(const Teuchos::ParameterList& aggregatorParams, int nTopos) :
Aggregator(aggregatorParams, nTopos)
//**********************************************************************
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(
    !aggregatorParams.isType<Teuchos::Array<double> >("Weights"),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator requires weights.  None given." << std::endl );

  weights = aggregatorParams.get<Teuchos::Array<double> >("Weights");

  TEUCHOS_TEST_FOR_EXCEPTION(
    weights.size() != aggregatedValuesNames.size() &&
    weights.size() != aggregatedDerivativesNames.size(),
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Scaled aggregator: Number of weights != number of values or derivatives." << std::endl );
}

//**********************************************************************
template <typename C>
Aggregator_DistExtremum<C>::Aggregator_DistExtremum(const Teuchos::ParameterList& aggregatorParams, int nTopos) :
Aggregator(aggregatorParams, nTopos)
//**********************************************************************
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    aggregatedValuesNames.size() == 0 &&
    aggregatedDerivativesNames.size() > 0,
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Extremum aggregator: Minimum/Maximum weighting requires values (only derivatives provided)." << std::endl );
}



//**********************************************************************
void
Aggregator_Scaled::Evaluate()
//**********************************************************************
{
  int numValues = values.size();

  *valueAggregated=shiftValueAggregated;
 
  if(normalize.size() == 0){
    normalize.resize(numValues);
    for(int i=0; i<numValues; i++){
      Albany::StateArrayVec& src = values[i].app->getStateMgr().getStateArrays().elemStateArrays;
      Albany::MDArray& valSrc = src[0][values[i].name];
      double val = valSrc(0);
      double globalVal = val;
      if( comm != Teuchos::null )
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &val, &globalVal);
      normalize[i] = (globalVal != 0.0) ? 1.0/fabs(globalVal) : 1.0;
    }
    if( comm != Teuchos::null ){
      if( comm->getRank()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Normalizing:" << std::endl;
        for(int i=0; i<numValues; i++){
          std::cout << "   " << values[i].name << " init = " << normalize[i] << std::endl;
        }
        std::cout << "************************************************************************" << std::endl;
      }
    } 
  }

  for(int sv=0; sv<numValues; sv++){
    Albany::StateArrayVec& src = values[sv].app->getStateMgr().getStateArrays().elemStateArrays;
    Albany::MDArray& valSrc = src[0][values[sv].name];
    double globalVal, val = valSrc(0);
    if( comm != Teuchos::null ){
      Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &val, &globalVal);
      if( comm->getRank()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: " << values[sv].name << " = " << globalVal << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    } else globalVal = val;
    *valueAggregated += normalize[sv]*weights[sv]*globalVal;
    valSrc(0)=0.0;
  }
  
  *valueAggregated *= scaleValueAggregated;

  if( comm != Teuchos::null ){
    if( comm->getRank()==0 ){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: " << outputValueName << " = " << *valueAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }


  int numDerivatives = derivatives.size();

 
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();

  for(int itopo=0; itopo<numTopologies; itopo++){

    Epetra_Vector& deriv = *(derivAggregated[itopo]);

    deriv.PutScalar(0.0);

    for(int sv=0; sv<numDerivatives; sv++){
      Albany::StateArrayVec& src = derivatives[sv].app->getStateMgr().getStateArrays().elemStateArrays;
      int numWorksets = src.size();
      for(int ws=0; ws<numWorksets; ws++){
        Albany::MDArray& derSrc = src[ws][derivatives[sv].name[itopo]];
        int numCells = derSrc.dimension(0);
        int numNodes = derSrc.dimension(1);
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++)
            deriv.SumIntoGlobalValue(wsElNodeID[ws][cell][node], 0, 
                                     scaleValueAggregated*normalize[sv]*weights[sv]*derSrc(cell,node));
      }
    }
  }
}

//**********************************************************************
template <typename C>
void Aggregator_Extremum<C>::Evaluate()
//**********************************************************************
{

  *valueAggregated=shiftValueAggregated;

  int extremum_index = 0;
  int numValues = values.size();
  if(numValues > 0){
    Albany::StateArrayVec& src = values[0].app->getStateMgr().getStateArrays().elemStateArrays;
    Albany::MDArray& valSrc = src[0][values[0].name];
    double extremum = valSrc(0);
    for(int sv=0; sv<numValues; sv++){
      double globalVal, val = valSrc(0);
      if( comm != Teuchos::null ){
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &val, &globalVal);
        if( comm->getRank()==0 ){
          std::cout << "************************************************************************" << std::endl;
          std::cout << "  Aggregator: " << values[sv].name << " = " << globalVal << std::endl;
          std::cout << "************************************************************************" << std::endl;
        }
      } else globalVal = val;
      if( compare(globalVal,extremum) ){
        extremum_index = sv;
        extremum = globalVal;
      }
      valSrc(0)=0.0;
    }
 
    *valueAggregated += extremum;
  
    if( comm != Teuchos::null ){
      if( comm->getRank()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: " << outputValueName << " = " << *valueAggregated << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }


  int numDerivatives = derivatives.size();

  if(numDerivatives > 0){

    for(int itopo=0; itopo<numTopologies; itopo++){

      Epetra_Vector& deriv = *(derivAggregated[itopo]);

      deriv.PutScalar(0.0);
 
      const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
       wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();
  
      Albany::StateArrayVec& src = derivatives[extremum_index].app->getStateMgr().getStateArrays().elemStateArrays;
      int numWorksets = src.size();
      for(int ws=0; ws<numWorksets; ws++){
        Albany::MDArray& derSrc = src[ws][derivatives[extremum_index].name[itopo]];
        int numCells = derSrc.dimension(0);
        int numNodes = derSrc.dimension(1);
        for(int cell=0; cell<numCells; cell++)
            for(int node=0; node<numNodes; node++)
            deriv.SumIntoGlobalValue(wsElNodeID[ws][cell][node], 0, derSrc(cell,node));
      }
    }
  }
}

//**********************************************************************
template <typename C>
void Aggregator_DistExtremum<C>::Evaluate()
//**********************************************************************
{
  *valueAggregated = shiftValueAggregated;


  int extremum_index = 0;
  int numValues = values.size();
  if(numValues > 0){
    SubValue& value = values[0];
    double* valView; value.value->ExtractView(&valView);
    double extremum = valView[0];
    for(int i=0; i<numValues; i++){
      SubValue& value = values[i];
      double* valView; value.value->ExtractView(&valView);
      if( compare(valView[0],extremum) ){
        extremum = valView[0];
        extremum_index = i;
      }

      if( comm != Teuchos::null ){
        if( comm->getRank()==0 ){
          std::cout << "************************************************************************" << std::endl;
          std::cout << "  Aggregator: Input variable " << i << std::endl;
          std::cout << "   " << value.name << " = " << valView[0] << std::endl;
          std::cout << "************************************************************************" << std::endl;
        }
      }
    }

    *valueAggregated += extremum;

    if( comm != Teuchos::null ){
      if( comm->getRank()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: Output " << std::endl;
        std::cout << "   Value = " << *valueAggregated << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }

  int numDerivatives = derivatives.size();

  if(numDerivatives > 0){

    for(int itopo=0; itopo<numTopologies; itopo++){

      Epetra_Vector& deriv = *(derivAggregated[itopo]);

      deriv.PutScalar(0.0);

      double *derDest; deriv.ExtractView(&derDest);
  
      const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
        wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();

      SubDerivative& derivative = derivatives[extremum_index];

      double* srcView; (*derivative.value)(0)->ExtractView(&srcView);

      int nLocalVals = deriv.MyLength();
      for(int lid=0; lid<nLocalVals; lid++)
        derDest[lid] += srcView[lid];
    }
  }
}
//**********************************************************************
void
Aggregator_DistScaled::Evaluate()
//**********************************************************************
{
  *valueAggregated = shiftValueAggregated;

  int nValues = values.size();
  if(normalize.size() == 0){
    normalize.resize(nValues);
    for(int i=0; i<nValues; i++){
      double* valView; values[i].value->ExtractView(&valView);
      normalize[i] = (valView[0] != 0.0) ? 1.0/fabs(valView[0]) : 1.0;
    }
  }

  for(int i=0; i<values.size(); i++){
    SubValue& value = values[i];

    double* valView; value.value->ExtractView(&valView);
    *valueAggregated += valView[0]*normalize[i]*weights[i];

    if( comm != Teuchos::null ){
      if( comm->getRank()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: Input variable " << i << std::endl;
        std::cout << "   " << value.name << " = " << valView[0] << std::endl;
        std::cout << "   " << value.name << " (scaled) = " << valView[0]*normalize[i] << std::endl;
        std::cout << "   Weight = " << weights[i] << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }

  *valueAggregated *= scaleValueAggregated;

  if( comm != Teuchos::null ){
    if( comm->getRank()==0 ){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: Output " << std::endl;
      std::cout << "   Value = " << *valueAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }


  for(int itopo=0; itopo<numTopologies; itopo++){

    Epetra_Vector& deriv = *(derivAggregated[itopo]);

    deriv.PutScalar(0.0);
    double *derDest; deriv.ExtractView(&derDest);
    int nLocalVals = deriv.MyLength();

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
      wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();

    for(int i=0; i<derivatives.size(); i++){
      SubDerivative& derivative = derivatives[i];

      double* srcView; (*derivative.value)(0)->ExtractView(&srcView);
  
      for(int lid=0; lid<nLocalVals; lid++)
        derDest[lid] += srcView[lid]*normalize[i]*weights[i]*scaleValueAggregated;
    }
  }
}
//**********************************************************************
void
Aggregator_DistScaled::EvaluateT()
//**********************************************************************
{
  *valueAggregated = shiftValueAggregated;

  int nValues = valuesT.size();
  if(normalize.size() == 0){
    normalize.resize(nValues);
    for(int i=0; i<nValues; i++){
      Teuchos::ArrayRCP<const double> valView = valuesT[i].value->get1dView(); 
      normalize[i] = (valView[0] != 0.0) ? 1.0/fabs(valView[0]) : 1.0;
    }
  }

  for(int i=0; i<valuesT.size(); i++){
    SubValueT& value = valuesT[i];

    Teuchos::ArrayRCP<const double> valView = valuesT[i].value->get1dView(); 
    *valueAggregated += valView[0]*normalize[i]*weights[i];

    if( comm != Teuchos::null ){
      if( comm->getRank()==0 ){
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: Input variable " << i << std::endl;
        std::cout << "   " << value.name << " = " << valView[0] << std::endl;
        std::cout << "   " << value.name << " (scaled) = " << valView[0]*normalize[i] << std::endl;
        std::cout << "   Weight = " << weights[i] << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }

 
  *valueAggregated *= scaleValueAggregated;

  if( comm != Teuchos::null ){
    if( comm->getRank()==0 ){
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: Output " << std::endl;
      std::cout << "   Value = " << *valueAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }


  for(int itopo=0; itopo<numTopologies; itopo++){

    Tpetra_Vector& derivT = *(derivAggregatedT[itopo]);

    derivT.putScalar(0.0);
    Teuchos::ArrayRCP<double> derDest = derivT.get1dViewNonConst(); 
    int nLocalVals = derivT.getLocalLength();

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
      wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();

    for(int i=0; i<derivatives.size(); i++){
      SubDerivativeT& derivative = derivativesT[i];

      Teuchos::ArrayRCP<const double> srcView = derivative.value->getData(0); 

      for(int lid=0; lid<nLocalVals; lid++)
        derDest[lid] += srcView[lid]*normalize[i]*weights[i]*scaleValueAggregated;
    }
  } 
}
}
