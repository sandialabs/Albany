//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Aggregator.hpp"
#include "ATO_Solver.hpp"

#include "Albany_Application.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Teuchos_TestForException.hpp"

#include <functional>

namespace ATO {

//**********************************************************************
Teuchos::RCP<Aggregator> 
AggregatorFactory::create(const Teuchos::ParameterList& aggregatorParams,
                          const std::string& entityType, int nTopos)
{
  Teuchos::Array<std::string> values = aggregatorParams.get<Teuchos::Array<std::string> >("Values");

  if (entityType == "State Variable") {
    std::string weightingType = aggregatorParams.get<std::string>("Weighting");
    if (weightingType == "Scaled" || weightingType == "Fixed") {  
      return Teuchos::rcp(new Aggregator_Scaled(aggregatorParams, nTopos));
    } else if (weightingType == "Maximum") {
      return Teuchos::rcp(new Aggregator_Extremum<std::greater<double> >(aggregatorParams, nTopos));
    } else if (weightingType == "Minimum"  ) {
      return Teuchos::rcp(new Aggregator_Extremum<std::less<double> >(aggregatorParams, nTopos));
    } else {
      return Teuchos::rcp(new Aggregator_Uniform(aggregatorParams, nTopos));
    }
  } else if (entityType == "Distributed Parameter") {
    std::string weightingType = aggregatorParams.get<std::string>("Weighting");
    if (weightingType == "Homogenized") {
      return Teuchos::rcp(new Aggregator_Homogenized(aggregatorParams, nTopos));
    } else if (weightingType == "Scaled") {
      return Teuchos::rcp(new Aggregator_DistScaled(aggregatorParams, nTopos));
    } else if (weightingType == "Maximum") {
      return Teuchos::rcp(new Aggregator_DistExtremum<std::greater<double> >(aggregatorParams, nTopos));
    } else if (weightingType == "Minimum") {
      return Teuchos::rcp(new Aggregator_DistExtremum<std::less<double> >(aggregatorParams, nTopos));
    } else {
      return Teuchos::rcp(new Aggregator_DistUniform(aggregatorParams, nTopos));
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               "Error!  Unknown 'Entity Type' requested.\n");
  }
}

//**********************************************************************
Aggregator::Aggregator(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : numTopologies(nTopos)
{ 
  parse(aggregatorParams);
}

//**********************************************************************
void Aggregator::parse(const Teuchos::ParameterList& aggregatorParams)
//**********************************************************************
{
  if (aggregatorParams.isType<Teuchos::Array<std::string> >("Values")) {
    aggregatedValuesNames = aggregatorParams.get<Teuchos::Array<std::string> >("Values");
  }

  if (aggregatorParams.isType<Teuchos::Array<std::string> >("Derivatives")) {
    aggregatedDerivativesNames = aggregatorParams.get<Teuchos::Array<std::string> >("Derivatives");
  }

  TEUCHOS_TEST_FOR_EXCEPTION((aggregatedValuesNames.size() != 0) &&
                             (aggregatedDerivativesNames.size() != 0) &&
                             (aggregatedValuesNames.size() != aggregatedDerivativesNames.size()),
                              Teuchos::Exceptions::InvalidParameter,
                             "Error!  Number of values != number of derivatives.\n"
                             "        Check value aggregator input.\n");
  
  TEUCHOS_TEST_FOR_EXCEPTION((aggregatedValuesNames.size() == 0) &&
                             (aggregatedDerivativesNames.size() == 0),
                             Teuchos::Exceptions::InvalidParameter,
                             "Error!  No values and no derivatives provided.\n"
                             "        Check value aggregator input.\n");
  
  if (aggregatedValuesNames.size() > 0) {
    outputValueName = aggregatorParams.get<std::string>("Output Value Name");
  }

  if (aggregatedDerivativesNames.size() > 0) {
    outputDerivativeName = aggregatorParams.get<std::string>("Output Derivative Name");
  }

  if (aggregatorParams.isType<bool>("Normalize")) {
    if (aggregatorParams.get<bool>("Normalize") == false) {
      int nObjs = aggregatedValuesNames.size();
      normalize.resize(nObjs,1.0);
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION((aggregatedValuesNames.size() == 0) &&
                             (normalize.size() == 0),
                             Teuchos::Exceptions::InvalidParameter,
                             "Error! 'Normalize' must be set to 'false' if only derivatives are being aggregated\n."
                             "       Check value aggregator input.\n");

  if (aggregatorParams.isType<double>("Shift Output")) {
    shiftValueAggregated = aggregatorParams.get<double>("Shift Output");
  } else {
    shiftValueAggregated = 0.0;
  }

  if (aggregatorParams.isType<double>("Scale Output")) {
    scaleValueAggregated = aggregatorParams.get<double>("Scale Output");
  } else {
    scaleValueAggregated = 1.0;
  }

  comm = Teuchos::null;
}

//**********************************************************************
Aggregator_DistParamBased::
Aggregator_DistParamBased(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator(aggregatorParams,nTopos)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    nTopos > 1,
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Distributed parameter based aggregator only supports one topology." << std::endl );
}

//**********************************************************************
void Aggregator_DistParamBased::
SetInputVariables (const std::vector<SolverSubSolver>& subProblems,
                   const std::map<std::string, std::vector<Teuchos::RCP<const Thyra_Vector>>>& valueMap,
                   const std::map<std::string, std::vector<Teuchos::RCP<Thyra_MultiVector>>>& derivMap)
//**********************************************************************
{
  outApp = subProblems[0].app;

  // loop through sub variable names and find the containing state manager
  int numVars = aggregatedValuesNames.size();
  values.resize(numVars);

  std::map<std::string, std::vector<Teuchos::RCP<const Thyra_Vector>>>::const_iterator git;
  for (int ir=0; ir<numVars; ir++) {
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

  std::map<std::string, std::vector<Teuchos::RCP<Thyra_MultiVector>>>::const_iterator gpit;
  for (int ir=0; ir<numVars; ir++) {
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
void Aggregator_StateVarBased::SetInputVariables(const std::vector<SolverSubSolver>& subProblems)
//**********************************************************************
{
  outApp = subProblems[0].app;

  // loop through sub variable names and find the containing state manager

  int numVars = aggregatedValuesNames.size();
  values.resize(numVars);
    
  int numSubs = subProblems.size();
  for (int iv=0; iv<numVars; iv++) {
    bool objFound = false;
    std::string& objName = aggregatedValuesNames[iv];
    for (int is=0; is<numSubs; is++) {
      const Teuchos::RCP<Albany::Application>& app = subProblems[is].app;
      Albany::StateArray& src = app->getStateMgr().getStateArrays().elemStateArrays[0];
      if (src.count(objName) > 0) {
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
  
  for (int iv=0; iv<numVars; iv++) {
    bool derFound = false;
    std::string derName = aggregatedDerivativesNames[iv];
    for (int is=0; is<numSubs; is++) {
      const Teuchos::RCP<Albany::Application>& app = subProblems[is].app;
      Albany::StateArray& src = app->getStateMgr().getStateArrays().elemStateArrays[0];
      if (src.count(Albany::strint(derName,0)) > 0) {
        TEUCHOS_TEST_FOR_EXCEPTION(
          derFound, Teuchos::Exceptions::InvalidParameter, std::endl
          << "Derivative '" << derName << "' found in two state managers." << std::endl
          << "Derivative names must be unique to avoid ambiguity." << std::endl);
        derivatives[iv].name.resize(numTopologies);
        for (int itopo=0; itopo<numTopologies; ++itopo)
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
Aggregator_Uniform::
Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator(aggregatorParams, nTopos)
 , Aggregator_StateVarBased()
{ 
  int nAgg = aggregatedValuesNames.size();
  if (nAgg == 0) nAgg = aggregatedDerivativesNames.size();
  double weight = 1.0/nAgg;
  weights.resize(nAgg);
  for (int i=0; i<nAgg; ++i) weights[i] = weight;

  normalizeMethod = "Scaled";
}


//**********************************************************************
Aggregator_DistUniform::
Aggregator_DistUniform(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator_DistParamBased(aggregatorParams,nTopos)
{ 
  int nAgg = aggregatedValuesNames.size();
  if (nAgg == 0) nAgg = aggregatedDerivativesNames.size();
  double weight = 1.0/nAgg;
  weights.resize(nAgg);
  for (int i=0; i<nAgg; ++i) weights[i] = weight;
}

//**********************************************************************
Aggregator_Scaled::Aggregator_Scaled(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator(aggregatorParams, nTopos)
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(!aggregatorParams.isType<Teuchos::Array<double> >("Weights"),
                             Teuchos::Exceptions::InvalidParameter,
                             "Scaled aggregator requires weights.  None given.\n");

  weights = aggregatorParams.get<Teuchos::Array<double> >("Weights");

  TEUCHOS_TEST_FOR_EXCEPTION(weights.size() != aggregatedValuesNames.size() &&
                             weights.size() != aggregatedDerivativesNames.size(),
                             Teuchos::Exceptions::InvalidParameter,
                             "Scaled aggregator: Number of weights != number of values or derivatives.\n");
  
  normalizeMethod = aggregatorParams.get<std::string>("Weighting");
  if (normalizeMethod == "Fixed") {
    weights[0] = 1.0;
  }

  iteration=0;
  if (aggregatorParams.isType<int>("Ramp Interval")) {
    rampInterval = aggregatorParams.get<int>("Ramp Interval");
  }

  maxScale = 1e6;
  if (aggregatorParams.isType<double>("Maximum Scale Factor")) {
    maxScale = aggregatorParams.get<double>("Maximum Scale Factor");
  }
}

//**********************************************************************
template <typename CompareType>
Aggregator_Extremum<CompareType>::Aggregator_Extremum(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator(aggregatorParams, nTopos)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    aggregatedValuesNames.size() == 0 &&
    aggregatedDerivativesNames.size() > 0,
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Extremum aggregator: Minimum/Maximum weighting requires values (only derivatives provided)." << std::endl );
}

//**********************************************************************
Aggregator_DistScaled::
Aggregator_DistScaled(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator_DistParamBased(aggregatorParams, nTopos)
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(!aggregatorParams.isType<Teuchos::Array<double> >("Weights"),
                              Teuchos::Exceptions::InvalidParameter,
                              "Scaled aggregator requires weights.  None given.\n");

  weights = aggregatorParams.get<Teuchos::Array<double> >("Weights");

  TEUCHOS_TEST_FOR_EXCEPTION(weights.size() != aggregatedValuesNames.size() &&
                             weights.size() != aggregatedDerivativesNames.size(),
                             Teuchos::Exceptions::InvalidParameter,
                             "Scaled aggregator: Number of weights != number of values or derivatives.\n");
}

//**********************************************************************
Aggregator_Homogenized::
Aggregator_Homogenized(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator_DistParamBased(aggregatorParams, nTopos)
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(!aggregatorParams.isSublist("Homogenization"),
                              Teuchos::Exceptions::InvalidParameter,
                              "'Homogenized' aggregator requires 'Homogenization' ParameterList.\n");

  const Teuchos::ParameterList& homogParams = aggregatorParams.sublist("Homogenization");

  // TODO: parse
  TEUCHOS_TEST_FOR_EXCEPTION(!homogParams.isType<Teuchos::Array<double> >("Assumed State"),
                             Teuchos::Exceptions::InvalidParameter,
                             "'Homogenized' aggregator requires 'Assumed State' array.  None given.\n");

  m_assumedState = homogParams.get<Teuchos::Array<double> >("Assumed State");

  TEUCHOS_TEST_FOR_EXCEPTION(m_assumedState.size() != aggregatedValuesNames.size() &&
                             m_assumedState.size() != aggregatedDerivativesNames.size(),
                             Teuchos::Exceptions::InvalidParameter,
                             "'Homogenized' aggregator: Length of 'Assumed State' array != number of values or derivatives.\n");

  if (homogParams.isType<bool>("Return Reciprocal")) {
    m_reciprocate = homogParams.get<bool>("Return Reciprocal");
  } else {
    m_reciprocate = false;
  }

  m_initialValue = 0.0;
}

//**********************************************************************
template <typename CompareType>
Aggregator_DistExtremum<CompareType>::
Aggregator_DistExtremum(const Teuchos::ParameterList& aggregatorParams, int nTopos)
//**********************************************************************
 : Aggregator(aggregatorParams, nTopos)
{
  TEUCHOS_TEST_FOR_EXCEPTION(aggregatedValuesNames.size() == 0 &&
                             aggregatedDerivativesNames.size() > 0,
                             Teuchos::Exceptions::InvalidParameter,
                             "Extremum aggregator: Minimum/Maximum weighting requires values (only derivatives provided).\n");
}

//**********************************************************************
void Aggregator_Scaled::Evaluate()
//**********************************************************************
{
  int numValues = values.size();

  *valueAggregated=shiftValueAggregated;

  if (normalizeMethod == "Scaled" ) {

    if (normalize.size() == 0) {
      normalize.resize(numValues);
      for (int i=0; i<numValues; ++i) {
        Albany::StateArrayVec& src = values[i].app->getStateMgr().getStateArrays().elemStateArrays;
        Albany::MDArray& valSrc = src[0][values[i].name];
        double val = valSrc(0);
        double globalVal = val;
        if (comm != Teuchos::null )
          Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &val, &globalVal);
        normalize[i] = (globalVal != 0.0) ? 1.0/fabs(globalVal) : 1.0;
      }
      if (comm != Teuchos::null ) {
        if (comm->getRank()==0 ) {
          std::cout << "************************************************************************" << std::endl;
          std::cout << "  Normalizing:" << std::endl;
          for (int i=0; i<numValues; ++i) {
            std::cout << "   " << values[i].name << " init = " << normalize[i] << std::endl;
          }
          std::cout << "************************************************************************" << std::endl;
        }
      }
    }
  } else if (normalizeMethod == "Fixed") {
    normalize.resize(numValues);
    std::vector<double> F(numValues);
    for (int i=0; i<numValues; ++i) {
      Albany::StateArrayVec& src = values[i].app->getStateMgr().getStateArrays().elemStateArrays;
      Albany::MDArray& valSrc = src[0][values[i].name];
      double val = valSrc(0);
      double globalVal = val;
      if (comm != Teuchos::null )
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &val, &globalVal);
      F[i] = globalVal;
    }
    Teuchos::Array<double> mu(weights);
    if (rampInterval && iteration <= rampInterval) {
      double increment = 1.0/rampInterval;
      for (int i=1; i<numValues; ++i) {
        mu[i]*=(iteration*increment);
      }
      iteration++;
    }
    double etaSum=0.0;
    for (int i=1; i<numValues; ++i) {
      etaSum+=mu[i];
    }
    normalize[0]=1.0;
    for (int i=1; i<numValues; ++i) {
      normalize[i] = F[0]*mu[i]/(F[i]*(1.0-etaSum));
    }
    for (int i=1; i<numValues; ++i) {
      if (normalize[i] > maxScale) {
        normalize[i] = maxScale;
      }
    }
    
    if (comm != Teuchos::null ) {
      if (comm->getRank()==0 ) {
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Normalizing:" << std::endl;
        for (int i=0; i<numValues; ++i) {
          std::cout << "   " << values[i].name << " init = " << normalize[i] << std::endl;
        }
        std::cout << "************************************************************************" << std::endl;
      }
    }
    // The weights are still applied below.  Divide by 'weights' so 'normalize' act as coefficients.
    for (int i=1; i<numValues; ++i) {
      normalize[i]/=weights[i];
    }
  }

  for (int sv=0; sv<numValues; ++sv) {
    Albany::StateArrayVec& src = values[sv].app->getStateMgr().getStateArrays().elemStateArrays;
    Albany::MDArray& valSrc = src[0][values[sv].name];
    double globalVal, val = valSrc(0);
    if (comm != Teuchos::null ) {
      Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &val, &globalVal);
      if (comm->getRank()==0 ) {
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: " << values[sv].name << " = " << globalVal << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    } else globalVal = val;
    *valueAggregated += normalize[sv]*weights[sv]*globalVal;
    valSrc(0)=0.0;
  }

  *valueAggregated *= scaleValueAggregated;

  if (comm != Teuchos::null ) {
    if (comm->getRank()==0 ) {
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Aggregator: " << outputValueName << " = " << *valueAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }

  const int numDerivatives = derivatives.size();
  const auto& wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();
  for (int itopo=0; itopo<numTopologies; ++itopo) {

    Teuchos::RCP<Thyra_Vector> deriv = derivAggregated[itopo];
    deriv->assign(0.0);

    auto deriv_data = Albany::getNonconstLocalData(deriv);
    for (int sv=0; sv<numDerivatives; ++sv) {
      Albany::StateArrayVec& src = derivatives[sv].app->getStateMgr().getStateArrays().elemStateArrays;
      const int numWorksets = src.size();
      for (int ws=0; ws<numWorksets; ++ws) {
        Albany::MDArray& derSrc = src[ws][derivatives[sv].name[itopo]];
        const int numCells = derSrc.dimension(0);
        const int numNodes = derSrc.dimension(1);
        for (int cell=0; cell<numCells; ++cell) {
          for (int node=0; node<numNodes; ++node) {
            const LO lid = Albany::getLocalElement(deriv->range(),wsElNodeID[ws][cell][node]);
            deriv_data[lid] += scaleValueAggregated*normalize[sv]*weights[sv]*derSrc(cell,node);
          }
        }
      }
    }
  }
}

//**********************************************************************
template <typename CompareType>
void Aggregator_Extremum<CompareType>::Evaluate()
//**********************************************************************
{
  *valueAggregated=shiftValueAggregated;

  int extremum_index = 0;
  int numValues = values.size();
  if (numValues > 0) {
    Albany::StateArrayVec& src = values[0].app->getStateMgr().getStateArrays().elemStateArrays;
    Albany::MDArray& valSrc = src[0][values[0].name];
    double extremum = valSrc(0);
    for (int sv=0; sv<numValues; ++sv) {
      double globalVal, val = valSrc(0);
      if (comm != Teuchos::null ) {
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &val, &globalVal);
        if (comm->getRank()==0 ) {
          std::cout << "************************************************************************" << std::endl;
          std::cout << "  Aggregator: " << values[sv].name << " = " << globalVal << std::endl;
          std::cout << "************************************************************************" << std::endl;
        }
      } else {
        globalVal = val;
      }
      if (compare(globalVal,extremum) ) {
        extremum_index = sv;
        extremum = globalVal;
      }
      valSrc(0)=0.0;
    }

    *valueAggregated += extremum;
    
    if (comm != Teuchos::null ) {
      if (comm->getRank()==0 ) {
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  Aggregator: " << outputValueName << " = " << *valueAggregated << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }

  const int numDerivatives = derivatives.size();
  if (numDerivatives > 0) {
    const auto& wsElNodeID = outApp->getStateMgr().getDiscretization()->getWsElNodeID();
    Albany::StateArrayVec& src = derivatives[extremum_index].app->getStateMgr().getStateArrays().elemStateArrays;
    const int numWorksets = src.size();
    for (int itopo=0; itopo<numTopologies; ++itopo) {

      Teuchos::RCP<Thyra_Vector> deriv = derivAggregated[itopo];
      deriv->assign(0.0);
      auto deriv_data = Albany::getNonconstLocalData(deriv);

      for (int ws=0; ws<numWorksets; ++ws) {
        Albany::MDArray& derSrc = src[ws][derivatives[extremum_index].name[itopo]];
        int numCells = derSrc.dimension(0);
        int numNodes = derSrc.dimension(1);
        for (int cell=0; cell<numCells; ++cell) {
          for (int node=0; node<numNodes; ++node) {
            const LO lid = Albany::getLocalElement(deriv->range(),wsElNodeID[ws][cell][node]);
            deriv_data[lid] += derSrc(cell,node);
          }
        }
      }
    }
  }
}

//**********************************************************************
double Aggregator_DistParamBased::
sum(std::vector<Teuchos::RCP<const Thyra_Vector>> valVector, int index)
//**********************************************************************
{
  double retVal = 0.0;
  for (auto vec : valVector) {
    auto vec_data = Albany::getLocalData(vec);
    retVal += vec_data[index];
  }
  return retVal;
}

//**********************************************************************
std::vector<double>
Aggregator_DistParamBased::
sum(std::vector<Teuchos::RCP<const Thyra_Vector>> valVector)
//**********************************************************************
{
  std::vector<double> retVal;
  if (valVector.size() == 0) {
    return retVal;
  }

  int length = Albany::getSpmdVectorSpace(valVector[0]->range())->localSubDim();
  retVal.resize(length,0.0);

  for (auto vec : valVector) {
    Teuchos::ArrayRCP<const double> valView = Albany::getLocalData(vec);
    for (int i=0; i<length; ++i) {
      retVal[i] += valView[i];
    }
  }
  return retVal;
}

//**********************************************************************
template <typename CompareType>
void Aggregator_DistExtremum<CompareType>::Evaluate()
//**********************************************************************
{
  *valueAggregated = shiftValueAggregated;

  int extremum_index = 0;
  int numValues = values.size();
  if (numValues > 0) {
    double extremum = sum(values[0].value, /*index=*/ 0);
    for (int i=0; i<numValues; ++i) {
      SubValueType& value = values[i];
      double myVal = sum(value.value, /*index=*/ 0);
      if (compare(myVal,extremum) ) {
        extremum = myVal;
        extremum_index = i;
      }

      if (comm != Teuchos::null ) {
        if (comm->getRank()==0 ) {
          std::cout << "************************************************************************" << std::endl;
          std::cout << "  DistExtremum Aggregator: Input variable " << i << std::endl;
          std::cout << "   " << value.name << " = " << myVal << std::endl;
          std::cout << "************************************************************************" << std::endl;
        }
      }
    }

    *valueAggregated += extremum;

    if (comm != Teuchos::null ) {
      if (comm->getRank()==0 ) {
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  DistExtremum Aggregator: Output " << std::endl;
        std::cout << "   Value = " << *valueAggregated << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }

  const int numDerivatives = derivatives.size();

  if (numDerivatives > 0) {
    SubDerivativeType& derivative = derivatives[extremum_index];
    for (int itopo=0; itopo<numTopologies; ++itopo) {

      Teuchos::RCP<Thyra_Vector> derivVec = derivAggregated[itopo];
      derivVec->assign(0.0);
      Teuchos::ArrayRCP<double> derDest = Albany::getNonconstLocalData(derivVec);
     
      for (auto deriv : derivative.value) {
        Teuchos::ArrayRCP<const double> srcView = Albany::getLocalData(deriv->col(0).getConst());
        const int nLocalVals = derDest.size();
        for (int lid=0; lid<nLocalVals; ++lid) {
          derDest[lid] += srcView[lid];
        }
      }
    }
  }

}

//**********************************************************************
void Aggregator_Homogenized::Evaluate()
//**********************************************************************
{
  double localValue = 0.0;

  int nValues = values.size();
  for (int i=0; i<nValues; ++i) {
    SubValueType& value = values[i];

    std::vector<double> vals = sum(value.value);
    int voigtLength = vals.size();
    double rowProd = 0.0;
    for ( int j=0; j<voigtLength; ++j) {
      rowProd += vals[j]*m_assumedState[j];
    }
    localValue += rowProd*m_assumedState[i];
  }

  if (comm->getRank()==0 ) {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "  Homogenized Constants " << std::endl;
    for (int i=0; i<nValues; ++i) {
      SubValueType& value = values[i];
      std::vector<double> vals = sum(value.value);
      int voigtLength = vals.size();
      for ( int j=0; j<voigtLength; ++j) {
        std::cout << " " << vals[j];
      }
      std::cout << std::endl;
    }
    std::cout << "************************************************************************" << std::endl;
  }

  double globalValue;
  if (comm != Teuchos::null ) {
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, &localValue, &globalValue);
  } else {
    globalValue = localValue;
  }

  if (m_initialValue == 0.0) m_initialValue = globalValue;

  globalValue /= m_initialValue;

  if (m_reciprocate) *valueAggregated = 1.0/globalValue;

  *valueAggregated += shiftValueAggregated;
  *valueAggregated *= scaleValueAggregated;

  if (comm != Teuchos::null ) {
    if (comm->getRank()==0 ) {
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  Homogenized Aggregator: Output " << std::endl;
      std::cout << "   Value = " << *valueAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }

  Teuchos::RCP<Thyra_Vector> deriv = derivAggregated[0];
  deriv->assign(0.0);
  Teuchos::ArrayRCP<double> derDest = Albany::getNonconstLocalData(deriv);
  const int nLocalVals = derDest.size();


  const int voigtSize = derivatives.size();
  const int numBlocks = derivatives[0].value.size();
  std::vector< Teuchos::ArrayRCP<const double> > C(voigtSize*voigtSize*numBlocks);

  for (int i=0; i<voigtSize; ++i) {
    SubDerivativeType& derivative = derivatives[i];
    for ( int j=0; j<voigtSize; ++j) {
      for ( int k=0; k<numBlocks; ++k) {
        C[(i*voigtSize+j)*numBlocks+k] = Albany::getLocalData(derivative.value[k]->col(j).getConst());
      }
    }
  }

  for (int lid=0; lid<nLocalVals; ++lid) {
    double scalarProd = 0.0;
    for (int i=0; i<voigtSize; ++i) {
      double rowProd = 0.0;
      for ( int j=0; j<voigtSize; ++j) {
        double blockSum = 0.0;
        for ( int k=0; k<numBlocks; ++k) {
          blockSum += C[(i*voigtSize+j)*numBlocks+k][lid];
        }
        rowProd += blockSum*m_assumedState[j];
      }
      scalarProd += rowProd*m_assumedState[i];
    }
    derDest[lid] += scalarProd/m_initialValue*scaleValueAggregated;
  }

  if (m_reciprocate) {
    double coef = -1.0/(globalValue*globalValue);
    for (int lid=0; lid<nLocalVals; ++lid) {
      derDest[lid] *= coef;
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
  if (normalize.size() == 0) {
    normalize.resize(nValues);
    for (int i=0; i<nValues; ++i) {
      double val = sum(values[i].value,/*index=*/0);
      normalize[i] = (val != 0.0) ? 1.0/fabs(val) : 1.0;
    }
  }

  for (size_t i=0; i<values.size(); ++i) {
    SubValueType& value = values[i];

    double val = sum(value.value,/*index=*/0);
    *valueAggregated += val*normalize[i]*weights[i];

    if (comm != Teuchos::null) {
      if (comm->getRank()==0) {
        std::cout << "************************************************************************" << std::endl;
        std::cout << "  DistScaled Aggregator: Input variable " << i << std::endl;
        std::cout << "   " << value.name << " = " << val << std::endl;
        std::cout << "   " << value.name << " (scaled) = " << val*normalize[i] << std::endl;
        std::cout << "   Weight = " << weights[i] << std::endl;
        std::cout << "************************************************************************" << std::endl;
      }
    }
  }

  *valueAggregated *= scaleValueAggregated;

  if (comm != Teuchos::null ) {
    if (comm->getRank()==0 ) {
      std::cout << "************************************************************************" << std::endl;
      std::cout << "  DistScaled Aggregator: Output " << std::endl;
      std::cout << "   Value = " << *valueAggregated << std::endl;
      std::cout << "************************************************************************" << std::endl;
    }
  }

  for (int itopo=0; itopo<numTopologies; ++itopo) {
    Teuchos::RCP<Thyra_Vector> deriv = derivAggregated[itopo];

    deriv->assign(0.0);
    auto derDest = Albany::getNonconstLocalData(deriv);
    const int nLocalVals = derDest.size();

    for (size_t i=0; i<derivatives.size(); ++i) {
      SubDerivativeType& derivative = derivatives[i];

      for (auto der : derivative.value) {
        auto srcView = Albany::getLocalData(der->col(0).getConst());
        for (int lid=0; lid<nLocalVals; ++lid) {
          derDest[lid] += srcView[lid]*normalize[i]*weights[i]*scaleValueAggregated;
        }
      }
    }
  }
}

} // namespace ATO
