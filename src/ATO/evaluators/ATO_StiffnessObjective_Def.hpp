//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "ATO_TopoTools.hpp"

template<typename EvalT, typename Traits>
ATO::StiffnessObjectiveBase<EvalT, Traits>::
StiffnessObjectiveBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
  qp_weights ("Weights", dl->qp_scalar     ),
  BF         ("BF",      dl->node_qp_scalar)

{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string gfLayout = responseParams->get<std::string>("Gradient Field Layout");
  std::string wcLayout = responseParams->get<std::string>("Work Conjugate Layout");
  
  Teuchos::RCP<PHX::DataLayout> layout;
  if(gfLayout == "QP Tensor3") layout = dl->qp_tensor3;
  else
  if(gfLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(gfLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Gradient Field Layout " << gfLayout <<
                               "!" << std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _gradX(responseParams->get<std::string>("Gradient Field Name"), layout);
  gradX = _gradX;

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


  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");


  topology = paramsFromProblem->get<Teuchos::RCP<Topology> >("Topology");

  FName = responseParams->get<std::string>("Response Name");
  dFdpName = responseParams->get<std::string>("Response Derivative Name");
  if(responseParams->isType<int>("Penalty Function")){
    functionIndex = responseParams->get<int>("Penalty Function");
  } else functionIndex = 0;

  //! Register with state manager
  this->pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");
  this->pStateMgr->registerStateVariable(FName, dl->workset_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, false);
  this->pStateMgr->registerStateVariable(dFdpName, dl->node_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, false);


  this->addDependentField(qp_weights);
  this->addDependentField(BF);
  this->addDependentField(gradX);
  this->addDependentField(workConj);

  // Create tag
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
  this->utils.setFieldData(gradX,fm);
  this->utils.setFieldData(workConj,fm);
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************
template<typename Traits>
ATO::
StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
StiffnessObjective(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
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

  Albany::MDArray F = (*workset.stateArrayPtr)[FName];
  Albany::MDArray dEdp = (*workset.stateArrayPtr)[dFdpName];
  Albany::MDArray topo = (*workset.stateArrayPtr)[topology->getName()];
  std::vector<int> dims;
  gradX.dimensions(dims);
  int size = dims.size();

  double internalEnergy=0.0;

  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = dims[2];
  int numNodes = topo.dimension(1);

  if( size == 3 ){
    for(int cell=0; cell<numCells; cell++){
      for(int node=0; node<numNodes; node++) dEdp(cell,node) = 0.0;
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        double P = topology->Penalize(functionIndex,topoVal);
        double dP = topology->dPenalize(functionIndex,topoVal);
        double dE = 0.0;
        for(int i=0; i<numDims; i++)
          dE += gradX(cell,qp,i)*workConj(cell,qp,i)/2.0;
        dE *= qp_weights(cell,qp);
        internalEnergy += P*dE;
        for(int node=0; node<numNodes; node++)
          dEdp(cell,node) -= dP*dE*BF(cell,node,qp);
      }
    }
  } else
  if( size == 4 ){
    for(int cell=0; cell<numCells; cell++){
      for(int node=0; node<numNodes; node++) dEdp(cell,node) = 0.0;
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        double P = topology->Penalize(functionIndex,topoVal);
        double dP = topology->dPenalize(functionIndex,topoVal);
        double dE = 0.0;
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            dE += gradX(cell,qp,i,j)*workConj(cell,qp,i,j)/2.0;
        dE *= qp_weights(cell,qp);
        internalEnergy += P*dE;
        for(int node=0; node<numNodes; node++)
          dEdp(cell,node) -= dP*dE*BF(cell,node,qp);
      }
    }
  } else
  if( size == 5 ){
    for(int cell=0; cell<numCells; cell++){
      for(int node=0; node<numNodes; node++) dEdp(cell,node) = 0.0;
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        double P = topology->Penalize(functionIndex,topoVal);
        double dP = topology->dPenalize(functionIndex,topoVal);
        double dE = 0.0;
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            for(int k=0; k<numDims; k++)
              dE += gradX(cell,qp,i,j,k)*workConj(cell,qp,i,j,k)/2.0;
        dE *= qp_weights(cell,qp);
        internalEnergy += P*dE;
        for(int node=0; node<numNodes; node++)
          dEdp(cell,node) -= dP*dE*BF(cell,node,qp);
      }
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>5, Teuchos::Exceptions::InvalidParameter,
      "Unexpected array dimensions in StiffnessObjective:" << size << std::endl);
  }

  F(0) += internalEnergy;
}

template<typename Traits>
void ATO::StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}

