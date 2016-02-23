//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
//#include "Adapt_NodalDataBlock.hpp"
#include "ATO_TopoTools.hpp"

template<typename EvalT, typename Traits>
ATO::ModalObjectiveBase<EvalT, Traits>::
ModalObjectiveBase(Teuchos::ParameterList& p,
 		   const Teuchos::RCP<Albany::Layouts>& dl) :
  qp_weights ("Weights", dl->qp_scalar     ),
  BF         ("BF",      dl->node_qp_scalar),
  val_qp     ("Evec_Re0", dl->qp_vector ),
  gradX      ("EigenStrain0",dl->qp_tensor ),
  workConj   ("EigenStress0",dl->qp_tensor ),
  eigval     ("Eval_Re0", dl->workset_scalar )
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");

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
                                         "all", "scalar", 0.0, false, true);
  this->pStateMgr->registerStateVariable(dFdpName, dl->node_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, true);

  this->addDependentField(qp_weights);
  this->addDependentField(val_qp);
  this->addDependentField(gradX);
  this->addDependentField(workConj);
  this->addDependentField(eigval);

  // Create tag
  modal_objective_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));
  this->addEvaluatedField(*modal_objective_tag);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::ModalObjectiveBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qp_weights,fm);
  this->utils.setFieldData(val_qp,fm);
  this->utils.setFieldData(gradX,fm);
  this->utils.setFieldData(workConj,fm);
  this->utils.setFieldData(eigval,fm);
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************
template<typename Traits>
ATO::
ModalObjective<PHAL::AlbanyTraits::Residual, Traits>::
ModalObjective(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
}

template<typename Traits>
void ATO::ModalObjective<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void ATO::ModalObjective<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  Albany::MDArray F    = (*workset.stateArrayPtr)[FName];
  Albany::MDArray dEdp = (*workset.stateArrayPtr)[dFdpName];
  Albany::MDArray topo = (*workset.stateArrayPtr)[topology->getName()];
  std::vector<int> dims;
  gradX.dimensions(dims);
  int size = dims.size();

  int numCells = dims[0];
  int numQPs = dims[1];
  int numDims = dims[2];
  int numNodes = topo.dimension(1);

  if( size == 4 ){
    for(int cell=0; cell<numCells; cell++){
      double dE = 0.0;
      double dmass_term = 0.;
      double dstiffness_term = 0.;
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        double P = topology->Penalize(functionIndex,topoVal);
        double dP = topology->dPenalize(functionIndex,topoVal);
        double dE = 0.0;
        double dmass_term = 0.;
        double dstiffness_term = 0.;
        for(int i=0; i<numDims; i++) {
          dmass_term += val_qp(cell,qp,i)*val_qp(cell,qp,i) * qp_weights(cell,qp);
          for(int j=0; j<numDims; j++)
            dstiffness_term += dP*gradX(cell,qp,i,j)*workConj(cell,qp,i,j)*qp_weights(cell,qp);
        }
        for(int node=0; node<numNodes; node++)
        dEdp(cell,node) = -(dstiffness_term - dmass_term*eigval(0))*BF(cell,node,qp);
      }
//tevhack        std::cout << "dEdp(" << cell << ") = " << dEdp(cell) << std::endl;
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      "Unexpected array dimensions in StiffnessObjective:" << size << std::endl);
  }

/*
  if( size == 3 ){
    for(int cell=0; cell<numCells; cell++){
      double dE = 0.0;
      double P = topology->Penalize(topo(cell));
      for(int qp=0; qp<numQPs; qp++) {
        for(int i=0; i<numDims; i++) {
          dE += val_qp(cell,qp,i) * val_qp(cell,qp,i);
        }
        for(int node=0; node<numNodes; node++)
          dEdp(cell,node) = dE/P*BF(cell,node,qp);
      }
      std::cout << "dEdp(" << cell << ") = " << dEdp(cell) << std::endl;
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, 
      Teuchos::Exceptions::InvalidParameter,
      "Unexpected array dimensions in ModalObjective:" << size << std::endl
    );
  }
*/

  F(0) = -eigval(0);

}

template<typename Traits>
void ATO::ModalObjective<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}

