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
PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
ResponseSquaredL2DifferenceSideBase(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");

  sideSetName = plist->get<std::string>("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);

  // Gathering dimensions
  sideDim = dl_side->cell_gradient->extent(2);
  numQPs  = dl_side->qp_scalar->extent(2);

  Teuchos::RCP<PHX::DataLayout> layout;
  std::string rank,fname;

  rank         = plist->get<std::string>("Field Rank");
  fname        = plist->get<std::string>("Source Field Name");

  fieldDim = getLayout(dl_side,rank,layout);
  layout->dimensions(dims);

  if (fieldDim>0)
  {
    metric = decltype(metric)("Metric " + sideSetName, dl_side->qp_tensor);
    this->addDependentField(metric);
  }

  sourceField = decltype(sourceField)(fname,layout);
  w_measure   = decltype(w_measure)("Weighted Measure " + sideSetName, dl_side->qp_scalar);
  scaling     = plist->get("Scaling",1.0);

  this->addDependentField(sourceField);
  if (plist->isParameter("Target Field Name")) {
    TEUCHOS_TEST_FOR_EXCEPTION(plist->isParameter("Target Value"), std::logic_error,
                               "[ResponseSquaredL2DifferenceSideBase] Error! Both target value and target field provided.\n")
    std::string target_fname;
    target_fname = plist->get<std::string>("Target Field Name");
    targetField = decltype(targetField)(target_fname,layout);
    this->addDependentField(targetField);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(!plist->isParameter("Target Value"), std::logic_error,
                               "[ResponseSquaredL2DifferenceSideBase] Error! No target value or target field provided.\n")
    target_value = true;
    target_value_val = TargetScalarT(plist->get<double>("Target Value"));
  }
  this->addDependentField(w_measure);

  this->setName("Response Squared L2 Error Side" + PHX::typeAsString<EvalT>());

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response Squared L2 Error Side";
  std::string global_response_name = "Global Response Squared L2 Error Side";
  int worksetSize = dl->cell_scalar->extent(0);
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
void PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sourceField,fm);
  this->utils.setFieldData(w_measure,fm);

  if (!target_value) {
    this->utils.setFieldData(targetField,fm);
  }

  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response_eval, 0.0);

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // Zero out local response
  PHAL::set(this->local_response_eval, 0.0);

  std::vector<ScalarT> diff_1;
  std::vector<std::vector<ScalarT>> diff_2;
  if (fieldDim==1)
    diff_1.resize(dims[3]);
  else if (fieldDim==2)
    diff_2.resize(dims[3],std::vector<ScalarT>(dims[3]));

  if (workset.sideSets->find(sideSetName) != workset.sideSets->end())
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      ScalarT sum = 0;
      for (int qp=0; qp<numQPs; ++qp)
      {
        ScalarT sq = 0;
        // Computing squared difference at qp
        switch (fieldDim)
        {
          case 0:
            sq += std::pow(sourceField(cell,side,qp)-(target_value ? target_value_val : targetField(cell,side,qp)),2);
            break;
          case 1:
            // Precompute differentce and access fields only n times (not n^2)
            for (int i=0; i<dims[3]; ++i)
              diff_1[i] = sourceField(cell,side,qp,i) - (target_value ? target_value_val : targetField(cell,side,qp,i));

            for (int i=0; i<dims[3]; ++i)
              for (int j=0; j<dims[3]; ++j)
                sq += diff_1[i]*metric(cell,side,qp,i,j)*diff_1[j];
            break;
          case 2:
            // Precompute differentce and access fields only n^2 times (not n^4)
            for (int i=0; i<dims[3]; ++i)
              for (int j=0; j<dims[3]; ++j)
                diff_2[i][j] = sourceField(cell,side,qp,i,j) - (target_value ? target_value_val : targetField(cell,side,qp,i,j));

            for (int i=0; i<dims[3]; ++i)
              for (int j=0; j<dims[3]; ++j)
                for (int k=0; k<dims[3]; ++k)
                  for (int l=0; l<dims[3]; ++l)
                    sq += metric(cell,side,qp,k,i)*diff_2[i][j] * metric(cell,side,qp,j,l)*diff_2[l][k];
            break;
        }
        sum += sq * w_measure(cell,side,qp);
      }

      this->local_response_eval(cell, 0) = sum*scaling;
      this->global_response_eval(0) += sum*scaling;
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response_eval);

  if(workset.comm->getRank()==0)
    std::cout << "resp" << PHX::typeAsString<EvalT>() << ": " << this->global_response_eval(0) << "\n" << std::flush;

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
int PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
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
