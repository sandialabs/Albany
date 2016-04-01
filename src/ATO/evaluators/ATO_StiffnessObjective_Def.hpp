//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
#include "ATO_TopoTools.hpp"

template<typename EvalT, typename Traits>
ATO::StiffnessObjectiveBase<EvalT, Traits>::
StiffnessObjectiveBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl,
                  const Albany::MeshSpecsStruct* meshSpecs) :
  qp_weights ("Weights", dl->qp_scalar     ),
  BF         ("BF",      dl->node_qp_scalar)

{

  elementBlockName = meshSpecs->ebName;

  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  topologies = paramsFromProblem->get<Teuchos::RCP<TopologyArray> >("Topologies");

  Teuchos::ParameterList& configParams = paramsFromProblem->sublist("Configuration");
  Teuchos::ParameterList& blocksParams = configParams.sublist("Element Blocks");
  int nblocks = blocksParams.get<int>("Number of Element Blocks");
  bool blockFound = false;
  for(int ib=0; ib<nblocks; ib++){
    Teuchos::ParameterList& blockParams = blocksParams.sublist(Albany::strint("Element Block", ib));
    std::string blockName = blockParams.get<std::string>("Name");
    if( blockName != elementBlockName) continue;
    blockFound = true;

   if( blockParams.isSublist("Material") )
     penaltyModel = Teuchos::rcp( new PenaltyMaterial<ScalarT>( blockParams, p, dl) );
   else
   if( blockParams.isSublist("Mixture") )
     penaltyModel = Teuchos::rcp( new PenaltyMixture<ScalarT>( blockParams, p, dl) );
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std:: endl <<
      "Neither 'Material' spec nor 'Mixture' spec found." << std::endl);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(blockFound == false, Teuchos::Exceptions::InvalidParameter, std:: endl <<
    "Block spec not found." << std::endl);

  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  int nTopos = topologies->size();
  FName = responseParams->get<std::string>("Response Name");
  std::string dFdpBaseName = responseParams->get<std::string>("Response Derivative Name");
  dFdpNames.resize(nTopos);
  for(int itopo=0; itopo<nTopos; itopo++){
    dFdpNames[itopo] = Albany::strint(dFdpBaseName,itopo);
  }

  this->pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");
  this->pStateMgr->registerStateVariable(FName, dl->workset_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, false);
  for(int itopo=0; itopo<nTopos; itopo++){
    this->pStateMgr->registerStateVariable(dFdpNames[itopo], 
                                           dl->node_scalar, dl->dummy, 
                                           "all", "scalar", 0.0, false, false);
  }

  this->addDependentField(qp_weights);
  this->addDependentField(BF);

  Teuchos::Array< PHX::MDField<ScalarT> > depFields;
  penaltyModel->getDependentFields(depFields);

  int nFields = depFields.size();
  for(int ifield=0; ifield<nFields; ifield++)
    this->addDependentField(depFields[ifield]);

  stiffness_objective_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));
  this->addEvaluatedField(*stiffness_objective_tag);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::StiffnessObjectiveBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qp_weights,fm);
  this->utils.setFieldData(BF,fm);

  Teuchos::Array<PHX::MDField<ScalarT>* > depFields;
  penaltyModel->getDependentFields(depFields);

  int nFields = depFields.size();
  for(int ifield=0; ifield<nFields; ifield++)
    this->utils.setFieldData(*depFields[ifield],fm);

}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************
template<typename Traits>
ATO::
StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
StiffnessObjective(Teuchos::ParameterList& p, 
                   const Teuchos::RCP<Albany::Layouts>& dl,
                   const Albany::MeshSpecsStruct* meshSpecs) :
  StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl, meshSpecs)
{
}

template<typename Traits>
void ATO::StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}


template<typename Traits>
void ATO::StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if( elementBlockName != workset.EBName ) return;

  int nTopos = topologies->size();

  Teuchos::Array<double> drdz(nTopos), topoVals(nTopos);
  Albany::MDArray F = (*workset.stateArrayPtr)[FName];
  Teuchos::Array<Albany::MDArray> dEdp(nTopos), topo(nTopos);
  for(int itopo=0; itopo<nTopos; itopo++){
    dEdp[itopo] = (*workset.stateArrayPtr)[dFdpNames[itopo]];
    topo[itopo] = (*workset.stateArrayPtr)[(*topologies)[itopo]->getName()];
  }

 
  std::vector<int> dims;
  penaltyModel->getFieldDimensions(dims);
  int size = dims.size();

  double internalEnergy=0.0;

  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = dims[2];
  int numNodes = topo[0].dimension(1);

  double response;
  Teuchos::Array<double> dResponse(nTopos);
  
  for(int cell=0; cell<numCells; cell++){

    for(int itopo=0; itopo<nTopos; itopo++)
      for(int node=0; node<numNodes; node++) 
        dEdp[itopo](cell,node) = 0.0;

    for(int qp=0; qp<numQPs; qp++){

      // compute topology values at this qp
      for(int itopo=0; itopo<nTopos; itopo++){
        topoVals[itopo] = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVals[itopo] += topo[itopo](cell,node)*BF(cell,node,qp);
      }

      penaltyModel->Evaluate(topoVals, topologies, cell, qp, response, dResponse);

      internalEnergy += response*qp_weights(cell,qp);

      // assemble
      for(int itopo=0; itopo<nTopos; itopo++)
        for(int node=0; node<numNodes; node++)
          dEdp[itopo](cell,node) += dResponse[itopo]*BF(cell,node,qp)*qp_weights(cell,qp);
    }
  }
  F(0) += internalEnergy;
}

template<typename Traits>
void ATO::StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyModel<N>::
PenaltyModel(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
/******************************************************************************/
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string gfLayout = responseParams->get<std::string>("Gradient Field Layout");
  
  Teuchos::RCP<PHX::DataLayout> layout;
  if(gfLayout == "QP Tensor3"){ layout = dl->qp_tensor3; rank = 3; }
  else
  if(gfLayout == "QP Tensor"){ layout = dl->qp_tensor; rank = 2; }
  else
  if(gfLayout == "QP Vector"){ layout = dl->qp_vector; rank = 1; }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Gradient Field Layout " << gfLayout <<
                               "!" << std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _gradX(responseParams->get<std::string>("Gradient Field Name"), layout);
  gradX = _gradX;

  std::vector<int> dims;
  gradX.dimensions(dims);
  numDims = dims[2];
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMixture<N>::
PenaltyMixture(Teuchos::ParameterList& blockParams,
               Teuchos::ParameterList& p, 
               const Teuchos::RCP<Albany::Layouts>& dl) :
  PenaltyModel<N>(p, dl)
/******************************************************************************/
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string wcLayout = responseParams->get<std::string>("Work Conjugate Layout");
  
  Teuchos::RCP<PHX::DataLayout> layout;
  if(wcLayout == "QP Tensor3") layout = dl->qp_tensor3;
  else
  if(wcLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(wcLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Work Conjugate Layout " << wcLayout <<
                               "!" << std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector)" <<
                               std::endl);

  std::string workConjBaseName = responseParams->get<std::string>("Work Conjugate Name");

  Teuchos::ParameterList& mixtureParams = blockParams.sublist("Mixture");
  int nMats = mixtureParams.get<int>("Number of Materials");

  workConj.resize(nMats);
  for(int imat=0; imat<nMats; imat++){
    PHX::MDField<ScalarT> _workConj(Albany::strint(workConjBaseName,imat), layout);
    workConj[imat] = _workConj;
  }

  bool fieldFound = false;
  Teuchos::ParameterList& fieldsParams = blockParams.sublist("Mixture").sublist("Mixed Fields");
  int nFields = fieldsParams.get<int>("Number of Mixed Fields");
  for(int ifield=0; ifield<nFields; ifield++){
    Teuchos::ParameterList& 
      fieldParams = fieldsParams.sublist(Albany::strint("Mixed Field", ifield));
    std::string fieldName = fieldParams.get<std::string>("Field Name");
    if( fieldName != workConjBaseName ) continue;
    fieldFound = true;
    
    TEUCHOS_TEST_FOR_EXCEPTION(
      fieldParams.get<std::string>("Field Layout") != wcLayout,
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Field layout mismatch." << std::endl);

    Teuchos::ParameterList& 
      ruleParams = fieldParams.sublist(fieldParams.get<std::string>("Rule Type"));

    materialIndices = ruleParams.get<Teuchos::Array<int> >("Material Indices");
    mixtureTopologyIndices = ruleParams.get<Teuchos::Array<int> >("Topology Indices");
    mixtureFunctionIndices = ruleParams.get<Teuchos::Array<int> >("Function Indices");
    
  }

  topologyIndex = responseParams->get<int>("Topology Index");
  functionIndex = responseParams->get<int>("Function Index");
  
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
void ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyModel<N>::
getFieldDimensions(std::vector<int>& dims)
/******************************************************************************/
{
  gradX.dimensions(dims);
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
void ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMixture<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N> >& depFields)
/******************************************************************************/
{
  int nWCs = workConj.size();
  depFields.resize(nWCs+1);
  for(int iwc=0; iwc<nWCs; iwc++)
    depFields[iwc] = workConj[iwc];

  depFields[nWCs] = gradX;
  
}
/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
void ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMixture<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N>* >& depFields)
/******************************************************************************/
{
  int nWCs = workConj.size();
  depFields.resize(nWCs+1);
  for(int iwc=0; iwc<nWCs; iwc++)
    depFields[iwc] = &workConj[iwc];

  depFields[nWCs] = &gradX;
  
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
void ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMaterial<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N> >& depFields)
/******************************************************************************/
{
  depFields.resize(2);
  depFields[0] = workConj;
  depFields[1] = gradX;
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
void ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMaterial<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N>* >& depFields)
/******************************************************************************/
{
  depFields.resize(2);
  depFields[0] = &workConj;
  depFields[1] = &gradX;
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMaterial<N>::
PenaltyMaterial(Teuchos::ParameterList& blockParams,
               Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  PenaltyModel<N>(p, dl)
/******************************************************************************/
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string wcLayout = responseParams->get<std::string>("Work Conjugate Layout");
  
  Teuchos::RCP<PHX::DataLayout> layout;
  if(wcLayout == "QP Tensor3") layout = dl->qp_tensor3;
  else
  if(wcLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(wcLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Work Conjugate Layout " << wcLayout <<
                               "!" << std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _workConj(responseParams->get<std::string>("Work Conjugate Name"), layout);
  workConj = _workConj;

  topologyIndex = responseParams->get<int>("Topology Index");
  functionIndex = responseParams->get<int>("Function Index");
  
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
void ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMixture<N>::
Evaluate(Teuchos::Array<double>& topoVals, Teuchos::RCP<TopologyArray>& topologies,
         int cell, int qp, N& response, Teuchos::Array<N>& dResponse)
/******************************************************************************/
{
  response = 0.0;
  N unityRemainder = 1.0;

  int nTopos = dResponse.size();
  for(int itopo=0; itopo<nTopos; itopo++)
    dResponse[itopo]=0.0;

  int nMats = mixtureTopologyIndices.size();
  int lastMatIndex = materialIndices[nMats];
  N lastMatdw = 0.0;
  if( rank == 1 ){
    for(int i=0; i<numDims; i++){
      lastMatdw += gradX(cell,qp,i)*workConj[lastMatIndex](cell,qp,i)/2.0;
    }
  } else
  if( rank == 2 ){
    for(int i=0; i<numDims; i++){
      for(int j=0; j<numDims; j++){
        lastMatdw += gradX(cell,qp,i,j)*workConj[lastMatIndex](cell,qp,i,j)/2.0;
      }
    }
  } else
  if( rank == 3 ){
    for(int i=0; i<numDims; i++){
      for(int j=0; j<numDims; j++){
        for(int k=0; k<numDims; k++){
          lastMatdw += gradX(cell,qp,i,j,k)*workConj[lastMatIndex](cell,qp,i,j,k)/2.0;
        }
      }
    }
  }
  N topoP = (*topologies)[topologyIndex]->Penalize(functionIndex, topoVals[topologyIndex]);
  for(int imat=0; imat<nMats; imat++){
    int matIdx = materialIndices[imat];
    int topoIdx = mixtureTopologyIndices[imat];
    int fncIdx = mixtureFunctionIndices[imat];
    N P = (*topologies)[topoIdx]->Penalize(fncIdx, topoVals[topoIdx]);
    N dP = (*topologies)[topoIdx]->dPenalize(fncIdx, topoVals[topoIdx]);
    unityRemainder -= P;
    N dw = 0.0;
    if( rank == 1 ){
      for(int i=0; i<numDims; i++){
        dw += gradX(cell,qp,i)*workConj[matIdx](cell,qp,i)/2.0;
      }
    } else
    if( rank == 2 ){
      for(int i=0; i<numDims; i++){
        for(int j=0; j<numDims; j++){
          dw += gradX(cell,qp,i,j)*workConj[matIdx](cell,qp,i,j)/2.0;
        }
      }
    } else
    if( rank == 3 ){
      for(int i=0; i<numDims; i++){
        for(int j=0; j<numDims; j++){
          for(int k=0; k<numDims; k++){
            dw += gradX(cell,qp,i,j,k)*workConj[matIdx](cell,qp,i,j,k)/2.0;
          }
        }
      }
    }
    response += P*dw;
    dResponse[topoIdx] = topoP*(dP*dw + (1.0-dP)*lastMatdw);
  }

  response += unityRemainder*lastMatdw;

  dResponse[topologyIndex] = 
    response * (*topologies)[topologyIndex]->dPenalize(functionIndex, topoVals[topologyIndex]);

  response *= (*topologies)[topologyIndex]->Penalize(functionIndex,topoVals[topologyIndex]);

}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename N>
void ATO::StiffnessObjectiveBase<EvalT,Traits>::PenaltyMaterial<N>::
Evaluate(Teuchos::Array<double>& topoVals, Teuchos::RCP<TopologyArray>& topologies,
         int cell, int qp, N& response, Teuchos::Array<N>& dResponse)
/******************************************************************************/
{
  response = 0.0;

  if( rank == 1 ){
    for(int i=0; i<numDims; i++)
      response += gradX(cell,qp,i)*workConj(cell,qp,i)/2.0;
  } else
  if( rank == 2 ){
    for(int i=0; i<numDims; i++)
      for(int j=0; j<numDims; j++)
        response += gradX(cell,qp,i,j)*workConj(cell,qp,i,j)/2.0;
  } else
  if( rank == 3 ){
    for(int i=0; i<numDims; i++)
      for(int j=0; j<numDims; j++)
        for(int k=0; k<numDims; k++)
          response += gradX(cell,qp,i,j,k)*workConj(cell,qp,i,j,k)/2.0;
  }
  
  dResponse[topologyIndex] = 
    response * (*topologies)[topologyIndex]->dPenalize(functionIndex, topoVals[topologyIndex]);

  response *= (*topologies)[topologyIndex]->Penalize(functionIndex,topoVals[topologyIndex]);
}
