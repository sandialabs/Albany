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
#include "PHAL_Utilities.hpp"


template<typename EvalT, typename Traits>
ATO::TensorPNormResponse<EvalT, Traits>::
TensorPNormResponse(Teuchos::ParameterList& p,
		    const Teuchos::RCP<Albany::Layouts>& dl) :
  qp_weights ("Weights", dl->qp_scalar),
  BF         ("BF",      dl->node_qp_scalar)
{
  using Teuchos::RCP;


  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string tLayout = responseParams->get<std::string>("Tensor Field Layout");

  Teuchos::RCP<PHX::DataLayout> layout;
  if(tLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(tLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Tensor Field Layout " << tLayout <<
                               "!" << std::endl << "Options are (QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _tensor(responseParams->get<std::string>("Tensor Field Name"), layout);
  tensor = _tensor;

  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  pVal = responseParams->get<double>("Exponent");

  topology = paramsFromProblem->get<Teuchos::RCP<Topology> >("Topology");
  if(responseParams->isType<int>("Penalty Function")){
    functionIndex = responseParams->get<int>("Penalty Function");
  } else functionIndex = 0;

  TEUCHOS_TEST_FOR_EXCEPTION(
    topology->getEntityType() != "Distributed Parameter",
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  TensorPNormResponse requires 'Distributed Parameter' based topology" << std::endl);

  topo = PHX::MDField<ParamScalarT,Cell,Node>(topology->getName(),dl->node_scalar);

  this->addDependentField(qp_weights);
  this->addDependentField(BF);
  this->addDependentField(tensor);
  this->addDependentField(topo);

  // Create tag
  objective_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>("Tensor PNorm", dl->dummy));
  this->addEvaluatedField(*objective_tag);
  
  std::string responseID = "ATO Tensor PNorm";
  this->setName(responseID);

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
void ATO::TensorPNormResponse<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qp_weights,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(tensor,fm);
  this->utils.setFieldData(topo,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  std::vector<int> dims;
  tensor.dimensions(dims);
  int size = dims.size();

  ScalarT pNorm=0.0;

  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = dims[2];
  int numNodes = topo.dimension(1);

  if( size == 3 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT devNorm = 0.0;
        for(int i=0; i<numDims; i++)
          devNorm += tensor(cell,qp,i)*tensor(cell,qp,i);
        ScalarT P = topology->Penalize(functionIndex, topoVal);
        devNorm = P*sqrt(devNorm);
        ScalarT dS = pow(devNorm,pVal) * qp_weights(cell,qp);
        this->local_response(cell,0) += dS;
        pNorm += dS;
      }
    }
  } else
  if( size == 4 && numDims == 2 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT s11 = tensor(cell,qp,0,0);
        ScalarT s22 = tensor(cell,qp,1,1);
        ScalarT d12 = s11 - s22;
        ScalarT s12 = tensor(cell,qp,0,1);
        ScalarT P = topology->Penalize(functionIndex, topoVal);
        ScalarT devNorm = P*sqrt((d12*d12+s11*s11+s22*s22)/2.0 + 3.0*s12*s12);
        ScalarT dS = pow(devNorm,pVal) * qp_weights(cell,qp);
        this->local_response(cell,0) += dS;
        pNorm += dS;
      }
    }
  } else
  if( size == 4 && numDims == 3 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT d12 = tensor(cell,qp,0,0)-tensor(cell,qp,1,1);
        ScalarT d23 = tensor(cell,qp,1,1)-tensor(cell,qp,2,2);
        ScalarT d31 = tensor(cell,qp,2,2)-tensor(cell,qp,0,0);
        ScalarT s23 = tensor(cell,qp,0,1);
        ScalarT s13 = tensor(cell,qp,0,2);
        ScalarT s12 = tensor(cell,qp,0,1);
        ScalarT P = topology->Penalize(functionIndex, topoVal);
        ScalarT devMag = P*sqrt((d12*d12+d23*d23+d31*d31)/2.0 + 3.0*(s23*s23+s13*s13+s12*s12));
        ScalarT dS = pow(devMag,pVal) * qp_weights(cell,qp);
        this->local_response(cell,0) += dS;
        pNorm += dS;
      }
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>4, Teuchos::Exceptions::InvalidParameter,
      "Unexpected array dimensions in Tensor PNorm Objective:" << size << std::endl);
  }

  this->global_response[0] += pNorm;


  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
    // Add contributions across processors
    PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);

    TensorPNormResponseSpec<EvalT,Traits>::postEvaluate(workset);
    
    // Do global scattering
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponseSpec<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  this->global_response[0] = pow(this->global_response[0],1.0/pVal);
}


// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  double gVal = this->global_response[0].val();
  double scale = pow(gVal,1.0/pVal-1.0)/pVal;

  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxT = workset.overlapped_dgdxT;
  if (overlapped_dgdxT != Teuchos::null) overlapped_dgdxT->scale(scale);

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdotT = workset.overlapped_dgdxdotT;
  if (overlapped_dgdxdotT != Teuchos::null) overlapped_dgdxdotT->scale(scale);
}

// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  double gVal = this->global_response[0].val();
  double scale = pow(gVal,1.0/pVal-1.0)/pVal;

  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdp = workset.overlapped_dgdp;
  if(overlapped_dgdp != Teuchos::null) overlapped_dgdp->Scale(scale);
#endif
}

#ifdef ALBANY_SG
// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::SGJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdx_sg = workset.overlapped_sg_dgdx;
  if(overlapped_dgdx_sg != Teuchos::null){
    for(int block=0; block<overlapped_dgdx_sg->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdx_sg)[block].Scale(scale);
    }
  }

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdxdot_sg = workset.overlapped_sg_dgdxdot;
  if(overlapped_dgdxdot_sg != Teuchos::null){
    for(int block=0; block<overlapped_dgdxdot_sg->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdxdot_sg)[block].Scale(scale);
    }
  }
}

#endif 
#ifdef ALBANY_ENSEMBLE 
// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::MPJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdx_mp = workset.overlapped_mp_dgdx;
  if(overlapped_dgdx_mp != Teuchos::null){
    for(int block=0; block<overlapped_dgdx_mp->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdx_mp)[block].Scale(scale);
    }
  }

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdxdot_mp = workset.overlapped_mp_dgdxdot;
  if(overlapped_dgdxdot_mp != Teuchos::null){
    for(int block=0; block<overlapped_dgdxdot_mp->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdxdot_mp)[block].Scale(scale);
    }
  }
}
#endif
