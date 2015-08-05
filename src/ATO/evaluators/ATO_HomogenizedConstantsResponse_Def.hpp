//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
ATO::HomogenizedConstantsResponse<EvalT, Traits>::
HomogenizedConstantsResponse(Teuchos::ParameterList& p,
		    const Teuchos::RCP<Albany::Layouts>& dl) :
  weights ("Weights", dl->qp_scalar)
{
  using Teuchos::RCP;

  Teuchos::ParameterList* rParams = p.get<Teuchos::ParameterList*>("Parameter List");

  std::string field_name = rParams->get<std::string>("Field Name");
  std::string fieldType = rParams->get<std::string>("Field Type", "Scalar");

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
  field = PHX::MDField<ScalarT>(field_name, field_layout);

  int field_rank = field_layout->rank();
  tensorRank = field_rank - 2; //first 2 dimensions are cell and qp.

  std::vector<int> dims;
  field.dimensions(dims);
  int numCells = dims[0];
  int numQPs   = dims[1];

  components.resize(0);
  
  Teuchos::RCP<PHX::DataLayout> local_response_layout;
  Teuchos::RCP<PHX::DataLayout> global_response_layout;
  if(fieldType == "Scalar"){
    local_response_layout = dl->cell_scalar;
    global_response_layout = dl->workset_scalar;
  } else
  if(fieldType == "Vector"){
    int numDims = dims[2];
    local_response_layout = Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(numCells,numDims));
    global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(numDims));
  } else
  if(fieldType == "Tensor"){
    int numDims = dims[2];
    int nVoigt = 0;
    for(int i=1; i<=numDims; i++) nVoigt += i;
    local_response_layout = Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(numCells,nVoigt));
    global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(nVoigt));

    components.resize(nVoigt,tensorRank);
    components(0,0) = 0; components(0,1) = 0;
    if(numDims==2){
      components(1,0) = 1; components(1,1) = 1;
      components(2,0) = 0; components(2,1) = 1;
    } else
    if(numDims==3){
      components(1,0) = 1; components(1,1) = 1;
      components(2,0) = 2; components(2,1) = 2;
      components(3,0) = 1; components(3,1) = 2;
      components(4,0) = 0; components(4,1) = 2;
      components(5,0) = 0; components(5,1) = 1;
    }
  }

  this->addDependentField(weights);
  this->addDependentField(field);

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
void ATO::HomogenizedConstantsResponse<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights,fm);
  this->utils.setFieldData(field,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::HomogenizedConstantsResponse<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  local_measure = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::HomogenizedConstantsResponse<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  ScalarT s;

  std::vector<int> dims;
  field.dimensions(dims);
  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = 0;
  if(tensorRank > 0) numDims = dims[2];

  for(int cell=0; cell<numCells; cell++){
    for(int qp=0; qp<numQPs; qp++){
      if( tensorRank == 0 ){
        s = -field(cell,qp) * weights(cell,qp);
        this->local_response(cell,0) += s;
        this->global_response(0) += s;
      } else
      if( tensorRank == 1 ){
        for(std::size_t idim=0; idim<numDims; idim++){
          s = -field(cell,qp,idim) * weights(cell,qp);
          this->local_response(cell,idim) += s;
          this->global_response(idim) += s;
        }
      } else 
      if( tensorRank == 2 ){
        int nterms = components.dimension(0);
        Intrepid::FieldContainer<int>& c = components;
        for(std::size_t ic=0; ic<nterms; ic++){
          s = -field(cell,qp,c(ic,0),c(ic,1)) * weights(cell,qp);
          this->local_response(cell,ic) += s;
          this->global_response(ic) += s;
        }
      }
      local_measure += weights(cell,qp);
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::HomogenizedConstantsResponse<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
    PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);
    Teuchos::reduceAll(*workset.comm, Teuchos::REDUCE_SUM, 1, &local_measure, &global_measure);

    HomogenizedConstantsResponseSpec<EvalT,Traits>::postEvaluate(workset);
    
    // Do global scattering
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::HomogenizedConstantsResponseSpec<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  int nterms = this->global_response.size();
  for(int i=0; i<nterms; i++)
    this->global_response[i] /= global_measure;
}


// **********************************************************************
template<typename Traits>
void ATO::HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  RealType scale = 1.0/global_measure;

  int nterms = this->global_response.size();
  for(int i=0; i<nterms; i++)
    this->global_response[i] *= scale;

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxT = workset.overlapped_dgdxT;
  if (overlapped_dgdxT != Teuchos::null) overlapped_dgdxT->scale(scale);

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdotT = workset.overlapped_dgdxdotT;
  if (overlapped_dgdxdotT != Teuchos::null) overlapped_dgdxdotT->scale(scale);
}

// **********************************************************************
template<typename Traits>
void ATO::HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  RealType scale = 1.0/global_measure;

  int nterms = this->global_response.size();
  for(int i=0; i<nterms; i++)
    this->global_response[i] *= scale;

#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdp = workset.overlapped_dgdp;
  if(overlapped_dgdp != Teuchos::null) overlapped_dgdp->Scale(scale);
#endif
}

#ifdef ALBANY_SG
// **********************************************************************
template<typename Traits>
void ATO::HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::SGJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  RealType scale = 1.0/global_measure;
  int nterms = this->global_response.size();
  for(int i=0; i<nterms; i++)
    this->global_response[i] *= scale;

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdx_sg = workset.overlapped_sg_dgdx;
  if(overlapped_dgdx_sg != Teuchos::null){
    for(int block=0; block<overlapped_dgdx_sg->size(); block++){
      (*overlapped_dgdx_sg)[block].Scale(scale);
    }
  }

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdxdot_sg = workset.overlapped_sg_dgdxdot;
  if(overlapped_dgdxdot_sg != Teuchos::null){
    for(int block=0; block<overlapped_dgdxdot_sg->size(); block++){
      (*overlapped_dgdxdot_sg)[block].Scale(scale);
    }
  }
}

#endif 
#ifdef ALBANY_ENSEMBLE 
// **********************************************************************
template<typename Traits>
void ATO::HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::MPJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  RealType scale = 1.0/global_measure;
  int nterms = this->global_response.size();
  for(int i=0; i<nterms; i++)
    this->global_response[i] *= scale;

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdx_mp = workset.overlapped_mp_dgdx;
  if(overlapped_dgdx_mp != Teuchos::null){
    for(int block=0; block<overlapped_dgdx_mp->size(); block++){
      (*overlapped_dgdx_mp)[block].Scale(scale);
    }
  }

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdxdot_mp = workset.overlapped_mp_dgdxdot;
  if(overlapped_dgdxdot_mp != Teuchos::null){
    for(int block=0; block<overlapped_dgdxdot_mp->size(); block++){
      (*overlapped_dgdxdot_mp)[block].Scale(scale);
    }
  }
}
#endif
