//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
#include "ATO_TopoTools.hpp"
#include "ATO_PenaltyModel.hpp"

template<typename EvalT, typename Traits>
ATO::StiffnessObjectiveBase<EvalT, Traits>::
StiffnessObjectiveBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl,
                  const Albany::MeshSpecsStruct* meshSpecs) :
  qp_weights ("Weights", dl->qp_scalar     ),
  BF         ("BF",      dl->node_qp_scalar)

{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  elementBlockName = meshSpecs->ebName;

  m_excludeBlock = false;
  if(responseParams->isType<Teuchos::Array<std::string>>("Blocks")){
    Teuchos::Array<std::string> 
      blocks = responseParams->get<Teuchos::Array<std::string>>("Blocks");
    if(find(blocks.begin(),blocks.end(),elementBlockName) == blocks.end()){
      m_excludeBlock = true;

      stiffness_objective_tag =
        Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));
      this->addEvaluatedField(*stiffness_objective_tag);
      return;
    }
  }

  ATO::PenaltyModelFactory<ScalarT> penaltyFactory;
  penaltyModel = penaltyFactory.create(p, dl, elementBlockName);

  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  topologies = paramsFromProblem->get<Teuchos::RCP<TopologyArray> >("Topologies");

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
  if(m_excludeBlock) return;

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

  if(m_excludeBlock) return;

  if( elementBlockName != workset.EBName ) return;

  int nTopos = topologies->size();

  Teuchos::Array<double> topoVals(nTopos);
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
//          dEdp[itopo](cell,node) += dResponse[itopo]*BF(cell,node,qp)*qp_weights(cell,qp);
//        JR:  The negative sign is to make this a total derivative 
          dEdp[itopo](cell,node) -= dResponse[itopo]*BF(cell,node,qp)*qp_weights(cell,qp);
    }
  }
  F(0) += internalEnergy;
}

template<typename Traits>
void ATO::StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}

