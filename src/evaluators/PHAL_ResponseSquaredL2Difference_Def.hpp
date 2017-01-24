//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PHAL_Utilities.hpp"

#include "Albany_Utils.hpp"

template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
ResponseSquaredL2DifferenceBase(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");

  // Gathering dimensions
  numQPs   = dl->qp_scalar->dimension(1);

  Teuchos::RCP<PHX::DataLayout> layout;
  std::string rank,fname,target_fname;

  rank           = plist->get<std::string>("Field Rank");
  fname          = plist->get<std::string>("Source Field Name");
  target_fname   = plist->get<std::string>("Target Field Name");

  fieldDim = getLayout(dl,rank,layout);
  layout->dimensions(dims);

  sourceField = PHX::MDField<SourceScalarT>(fname,layout);
  targetField   = PHX::MDField<TargetScalarT>(target_fname,layout);
  w_measure     = PHX::MDField<RealType,Cell,QuadPoint>("Weights",dl->qp_scalar);
  scaling       = plist->get("Scaling",1.0);

  this->addDependentField(sourceField.fieldTag());
  if (target_fname=="ZERO")
  {
    target_zero = true;
    this->addEvaluatedField(targetField);
  }
  else
  {
    target_zero = false;
    this->addDependentField(targetField.fieldTag());
  }
  this->addDependentField(w_measure.fieldTag());

  this->setName("Response Squared L2 Error " + PHX::typeAsString<EvalT>());

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response Squared L2 Error ";
  std::string global_response_name = "Global Response Squared L2 Error ";
  int worksetSize = dl->cell_scalar->dimension(0);
  int responseSize = 1;
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new PHX::MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::setup(p, dl);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sourceField,fm);
  this->utils.setFieldData(targetField,fm);
  this->utils.setFieldData(w_measure,fm);

  if (target_zero)
  {
    targetField.deep_copy(TargetScalarT(0.0));
    //PHAL::set(targetField, 0.0);
  }

  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::preEvaluate(typename Traits::PreEvalData workset)
{
  //PHAL::set(this->global_response, 0.0);
  this->global_response.deep_copy(ScalarT(0.0));

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
//  PHAL::set(this->local_response, 0.0);

  this->local_response.deep_copy(ScalarT(0.0));
  for (int cell=0; cell<workset.numCells; ++cell)
  {
    ScalarT sum = 0;
    for (int qp=0; qp<numQPs; ++qp)
    {
      ScalarT sq = 0;
      // Computing squared difference at qp
      switch (fieldDim)
      {
        case 0:
          sq += std::pow(sourceField(cell,qp)-targetField(cell,qp),2);
          break;
        case 1:
          for (int j=0; j<dims[2]; ++j)
            sq += std::pow(sourceField(cell,qp,j)-targetField(cell,qp,j),2);
          break;
        case 2:
          for (int j=0; j<dims[2]; ++j)
            for (int k=0; k<dims[3]; ++k)
              sq += std::pow(sourceField(cell,qp,j,k)-targetField(cell,qp,j,k),2);
          break;
      }
      sum += sq * w_measure(cell,qp);
    }
    this->local_response(cell, 0) = sum*scaling;
    this->global_response(0) += sum*scaling;
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);

  if(workset.comm->getRank()==0)
    std::cout << "resp" << PHX::typeAsString<EvalT>() << ": " << this->global_response(0) << "\n" << std::flush;

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
int PHAL::ResponseSquaredL2DifferenceBase<EvalT,Traits,SourceScalarT,TargetScalarT>::
getLayout (const Teuchos::RCP<Albany::Layouts>& dl, const std::string& rank, Teuchos::RCP<PHX::DataLayout>& layout)
{
  int dim = -1;
  if (rank=="Scalar")
  {
    layout = dl->qp_scalar;
    dim = 0;
  }
  else if (rank=="Vector")
  {
    layout = dl->qp_vector;
    dim = 1;
  }
  else if (rank=="Gradient")
  {
    layout = dl->qp_gradient;
    dim = 1;
  }
  else if (rank=="Tensor")
  {
    layout = dl->qp_tensor;
    dim = 2;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid 'Field Rank'.\n");
  }

  return dim;
}
