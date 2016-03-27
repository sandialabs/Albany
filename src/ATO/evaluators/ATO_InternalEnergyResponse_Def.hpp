//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "ATO_TopoTools.hpp"
#include "PHAL_Utilities.hpp"


template<typename EvalT, typename Traits>
ATO::InternalEnergyResponse<EvalT, Traits>::
InternalEnergyResponse(Teuchos::ParameterList& p,
		    const Teuchos::RCP<Albany::Layouts>& dl) :
  qp_weights ("Weights", dl->qp_scalar     ),
  BF         ("BF",      dl->node_qp_scalar)
{
  using Teuchos::RCP;


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
  TEUCHOS_TEST_FOR_EXCEPTION(
    topology->getEntityType() != "Distributed Parameter", 
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  InternalEnergyResponse requires 'Distributed Parameter' based topology" << std::endl);

  topo = PHX::MDField<ParamScalarT,Cell,Node>(topology->getName(),dl->node_scalar);

  if(responseParams->isType<int>("Penalty Function")){
    functionIndex = responseParams->get<int>("Penalty Function");
  } else functionIndex = 0;

  this->addDependentField(qp_weights);
  this->addDependentField(BF);
  this->addDependentField(gradX);
  this->addDependentField(workConj);
  this->addDependentField(topo);

  // Create tag
  stiffness_objective_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));
  this->addEvaluatedField(*stiffness_objective_tag);
  
  std::string responseID = "ATO Internal Energy";
  this->setName(responseID + PHX::typeAsString<EvalT>());

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);

  int responseSize = 1;
  int worksetSize = dl->qp_scalar->dimension(0);
  Teuchos::RCP<PHX::DataLayout> 
    global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(responseSize));
  Teuchos::RCP<PHX::DataLayout> 
    local_response_layout  = Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(worksetSize, responseSize));

  std::string local_response_name  = FName + " Local Response";
  std::string global_response_name = FName + " Global Response";

  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  p.set("Local Response Field Tag", local_response_tag);

  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Global Response Field Tag", global_response_tag);

  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::InternalEnergyResponse<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qp_weights,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(gradX,fm);
  this->utils.setFieldData(workConj,fm);
  this->utils.setFieldData(topo,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::InternalEnergyResponse<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response, 0.0);

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::InternalEnergyResponse<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  PHAL::set(this->local_response, 0.0);

  std::vector<int> dims;
  gradX.dimensions(dims);
  int size = dims.size();

  ScalarT internalEnergy=0.0;

  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = dims[2];
  int numNodes = topo.dimension(1);

  if( size == 3 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        ScalarT dE = 0.0;
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex,topoVal);
        for(int i=0; i<numDims; i++)
          dE += gradX(cell,qp,i)*workConj(cell,qp,i)/2.0;
        dE *= qp_weights(cell,qp);
        internalEnergy += P*dE;
        this->local_response(cell,0) += P*dE;
      }
    }
  } else
  if( size == 4 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        ScalarT dE = 0.0;
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex,topoVal);
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            dE += gradX(cell,qp,i,j)*workConj(cell,qp,i,j)/2.0;
        dE *= qp_weights(cell,qp);
        internalEnergy += P*dE;
        this->local_response(cell,0) += P*dE;
      }
    }
  } else
  if( size == 5 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        ScalarT dE = 0.0;
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex,topoVal);
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            for(int k=0; k<numDims; k++)
              dE += gradX(cell,qp,i,j,k)*workConj(cell,qp,i,j,k)/2.0;
        dE *= qp_weights(cell,qp);
        internalEnergy += P*dE;
        this->local_response(cell,0) += P*dE;
      }
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>5, Teuchos::Exceptions::InvalidParameter,
      "Unexpected array dimensions in StiffnessObjective:" << size << std::endl);
  }

  PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
  *gr += internalEnergy;

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::InternalEnergyResponse<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
    PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM,
                             this->global_response);

    // Do global scattering
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}


