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
ATO::TensorAverageResponse<EvalT, Traits>::
TensorAverageResponse(Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl) :
  weights ("Weights", dl->qp_scalar),
  BF      ("BF",      dl->node_qp_scalar)
{
  using Teuchos::RCP;

  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<TopologyArray>
    topologies = paramsFromProblem->get<Teuchos::RCP<TopologyArray> >("Topologies");

  TEUCHOS_TEST_FOR_EXCEPTION(
    topologies->size() != 1,
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  TensorAverageResponse not implemented for multiple topologies." << std::endl);

  topology = (*topologies)[0];

  if(responseParams->isType<int>("Penalty Function")){
    functionIndex = responseParams->get<int>("Penalty Function");
  } else functionIndex = 0;

  TEUCHOS_TEST_FOR_EXCEPTION(
    topology->getEntityType() != "Distributed Parameter",
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  TensorAverageResponse requires 'Distributed Parameter' based topology" << std::endl);

  topo = decltype(topo)(topology->getName(),dl->node_scalar);

  std::string field_name = responseParams->get<std::string>("Field Name");
  std::string fieldType = responseParams->get<std::string>("Field Type", "Scalar");

  Teuchos::RCP<PHX::DataLayout> field_layout;
  if (fieldType == "Scalar"){
    field_layout = dl->qp_scalar;
  } else
  if (fieldType == "Vector"){
    field_layout = dl->qp_vector;
  } else
  if (fieldType == "Tensor"){
    field_layout = dl->qp_tensor;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      "Invalid field type " << fieldType << ".  Support values are " <<
      "Scalar, Vector, and Tensor." << std::endl);
  }
  field = decltype(field)(field_name, field_layout);

  int field_rank = field_layout->rank();
  tensorRank = field_rank - 2; //first 2 dimensions are cell and qp.

  int numCells = field_layout->dimension(0);
  int numQPs   = field_layout->dimension(1);

  Teuchos::RCP<PHX::DataLayout> local_response_layout;
  Teuchos::RCP<PHX::DataLayout> global_response_layout;
  if(fieldType == "Scalar"){
    local_response_layout = dl->cell_scalar;
    global_response_layout = dl->workset_scalar;
  } else
  if(fieldType == "Vector"){
    int numDims = field_layout->dimension(2);
    local_response_layout = Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(numCells,numDims));
    global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(numDims));
  } else
  if(fieldType == "Tensor"){
    int numDims = field_layout->dimension(2);
    int nVoigt = 0;
    for(int i=1; i<=numDims; i++) nVoigt += i;
    local_response_layout = Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(numCells,nVoigt));
    global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(nVoigt));

    component0.resize(nVoigt);
    component1.resize(nVoigt);
    component0(0) = 0; component1(0) = 0;
    if(numDims==2){
      component0(1) = 1; component1(1) = 1;
      component0(2) = 0; component1(2) = 1;
    } else
    if(numDims==3){
      component0(1) = 1; component1(1) = 1;
      component0(2) = 2; component1(2) = 2;
      component0(3) = 1; component1(3) = 2;
      component0(4) = 0; component1(4) = 2;
      component0(5) = 0; component1(5) = 1;
    }
  }

  this->addDependentField(weights);
  this->addDependentField(BF);
  this->addDependentField(field);
  this->addDependentField(topo);

  // Create tag
  objective_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>("Field Average", dl->dummy));
  this->addEvaluatedField(*objective_tag);

  std::string responseID = "Field Average";
  this->setName(responseID);

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);

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
void ATO::TensorAverageResponse<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(topo,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorAverageResponse<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0;
       i<this->global_response_eval.size(); i++)
    this->global_response_eval[i] = 0.0;

  local_measure = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorAverageResponse<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0;
       i<this->local_response_eval.size(); i++)
    this->local_response_eval[i] = 0.0;

  ScalarT s;

  std::vector<int> dims;
  field.dimensions(dims);
  int numQPs   = dims[1];
  int numDims  = 0;
  if(tensorRank > 0) numDims = dims[2];

  int numNodes = topo.dimension(1);

  for(int cell=0; cell<workset.numCells; cell++){
    for(int qp=0; qp<numQPs; qp++){
      ScalarT topoVal = 0.0;
      for(int node=0; node<numNodes; node++)
        topoVal += topo(cell,node)*BF(cell,node,qp);
      ScalarT P = topology->Penalize(functionIndex, topoVal);
      if( tensorRank == 0 ){
        s = - P * field(cell,qp) * weights(cell,qp);
        this->local_response_eval(cell,0) += s;
        this->global_response_eval[0] += s;
      } else
      if( tensorRank == 1 ){
        for(std::size_t idim=0; idim<numDims; idim++){
          s = - P * field(cell,qp,idim) * weights(cell,qp);
          this->local_response_eval(cell,idim) += s;
          this->global_response_eval[idim] += s;
        }
      } else
      if( tensorRank == 2 ){
        int nterms = component0.size();
        for(std::size_t ic=0; ic<nterms; ic++){
          s = - P * field(cell,qp,component0(ic),component1(ic)) * weights(cell,qp);
          this->local_response_eval(cell,ic) += s;
          this->global_response_eval[ic] += s;
        }
      }
      local_measure += Albany::ADValue(weights(cell,qp));
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorAverageResponse<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
    PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response_eval);
    Teuchos::reduceAll(*workset.comm, Teuchos::REDUCE_SUM, 1, &local_measure, &global_measure);

    TensorAverageResponseSpec<EvalT,Traits>::postEvaluate(workset);

    // Do global scattering
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorAverageResponseSpec<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  int nterms = this->global_response_eval.size();
  for(int i=0; i<nterms; i++)
    this->global_response_eval[i] /= global_measure;
}


// **********************************************************************
template<typename Traits>
void ATO::TensorAverageResponseSpec<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  RealType scale = 1.0/global_measure;

  int nterms = this->global_response_eval.size();
  for(int i=0; i<nterms; i++)
    this->global_response_eval[i] *= scale;

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxT = workset.overlapped_dgdxT;
  if (overlapped_dgdxT != Teuchos::null) overlapped_dgdxT->scale(scale);

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdotT = workset.overlapped_dgdxdotT;
  if (overlapped_dgdxdotT != Teuchos::null) overlapped_dgdxdotT->scale(scale);
}

// **********************************************************************
template<typename Traits>
void ATO::TensorAverageResponseSpec<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  RealType scale = 1.0/global_measure;

  int nterms = this->global_response_eval.size();
  for(int i=0; i<nterms; i++)
    this->global_response_eval[i] *= scale;


  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdpT = workset.overlapped_dgdpT;
  if(overlapped_dgdpT != Teuchos::null) overlapped_dgdpT->scale(scale);
#ifndef ALBANY_EPETRA
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "\n WARNING: This run is using Distributed Parameters (ATO::TensorPNormResponse) "
       << "with Epetra turned OFF.  It is not yet clear if this works correctly, so use at your own risk!\n";
#endif
}

