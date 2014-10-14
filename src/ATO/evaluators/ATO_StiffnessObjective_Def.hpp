//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Adapt_NodalDataBlock.hpp"
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
  if(gfLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(gfLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Gradient Field Layout " << gfLayout <<
                               "!" << std::endl << "Options are (QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _gradX(responseParams->get<std::string>("Gradient Field Name"), layout);
  gradX = _gradX;

  if(wcLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(wcLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Work Conjugate Layout " << wcLayout <<
                               "!" << std::endl << "Options are (QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _workConj(responseParams->get<std::string>("Work Conjugate Name"), layout);
  workConj = _workConj;


  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");


  Teuchos::ParameterList& topoParams = paramsFromProblem->get<Teuchos::ParameterList>("Topology");
  ATO::TopoToolsFactory topoFactory;
  topoTools = topoFactory.create(topoParams);

  FName = responseParams->get<std::string>("Response Name");
  dFdpName = responseParams->get<std::string>("Response Derivative Name");
  topoName = topoParams.get<std::string>("Topology Name");
  topoCentering = topoParams.get<std::string>("Centering");

  //! Register with state manager
  this->pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");
  this->pStateMgr->registerStateVariable(FName, dl->workset_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, true);
  if( topoCentering == "Element" ){
    this->pStateMgr->registerStateVariable(dFdpName, dl->cell_scalar, dl->dummy, 
                                           "all", "scalar", 0.0, false, true);
  } else
  if( topoCentering == "Node" ){
    this->pStateMgr->registerStateVariable(dFdpName, dl->node_scalar, dl->dummy, 
                                           "all", "scalar", 0.0, false, true);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Unknown centering " << topoCentering << "!" << std::endl 
      << "Options are (Element, Node)" << std::endl);
  }


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
  Albany::MDArray topo = (*workset.stateArrayPtr)[topoName];
  std::vector<int> dims;
  gradX.dimensions(dims);
  int size = dims.size();

  double internalEnergy=0.0;

  if( topoCentering == "Element" ){
  
    if( size == 3 ){
      for(int cell=0; cell<dims[0]; cell++){
        double dE = 0.0;
        double P = topoTools->Penalize(topo(cell));
        double dP = topoTools->dPenalize(topo(cell));
        for(int qp=0; qp<dims[1]; qp++)
          for(int i=0; i<dims[2]; i++)
            dE += gradX(cell,qp,i)*workConj(cell,qp,i)*qp_weights(cell,qp);
        internalEnergy += P*dE;
        dEdp(cell) = -dP*dE;
      }
    } else
    if( size == 4 ){
      for(int cell=0; cell<dims[0]; cell++){
        double dE = 0.0;
        double P = topoTools->Penalize(topo(cell));
        double dP = topoTools->dPenalize(topo(cell));
        for(int qp=0; qp<dims[1]; qp++)
          for(int i=0; i<dims[2]; i++)
            for(int j=0; j<dims[3]; j++)
              dE += gradX(cell,qp,i,j)*workConj(cell,qp,i,j)*qp_weights(cell,qp);
        internalEnergy += P*dE;
        dEdp(cell) = -dP*dE;
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>4, Teuchos::Exceptions::InvalidParameter,
        "Unexpected array dimensions in StiffnessObjective:" << size << std::endl);
    }
  } else
  if( topoCentering == "Node" ){
    int numCells = dims[0];
    int numQPs   = dims[1];
    int numDims  = dims[2];
    int numNodes = topo.dimension(1);

    if( size == 3 ){
      for(int cell=0; cell<numCells; cell++){
        for(int node=0; node<numNodes; node++) dEdp(cell,node) = 0.0;
        double dE = 0.0;
        for(int qp=0; qp<numQPs; qp++){
          double topoVal = 0.0;
          for(int node=0; node<numNodes; node++)
            topoVal += topo(cell,node)*BF(cell,node,qp);
          double P = topoTools->Penalize(topoVal);
          double dP = topoTools->dPenalize(topoVal);
          for(int i=0; i<numDims; i++)
            dE += gradX(cell,qp,i)*workConj(cell,qp,i);
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
        double dE = 0.0;
        for(int qp=0; qp<numQPs; qp++){
          double topoVal = 0.0;
          for(int node=0; node<numNodes; node++)
            topoVal += topo(cell,node)*BF(cell,node,qp);
          double P = topoTools->Penalize(topoVal);
          double dP = topoTools->dPenalize(topoVal);
          for(int i=0; i<numDims; i++)
            for(int j=0; j<numDims; j++)
              dE += gradX(cell,qp,i,j)*workConj(cell,qp,i,j);
          dE *= qp_weights(cell,qp);
          internalEnergy += P*dE;
          for(int node=0; node<numNodes; node++)
            dEdp(cell,node) -= dP*dE*BF(cell,node,qp);
        }
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>4, Teuchos::Exceptions::InvalidParameter,
        "Unexpected array dimensions in StiffnessObjective:" << size << std::endl);
    }
  }

  F(0) += internalEnergy;
}

template<typename Traits>
void ATO::StiffnessObjective<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}

