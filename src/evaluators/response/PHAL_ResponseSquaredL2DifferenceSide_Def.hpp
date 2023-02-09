//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_ResponseSquaredL2DifferenceSide.hpp"
#include "PHAL_Utilities.hpp"

#include "Albany_Utils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"

template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
ResponseSquaredL2DifferenceSideBase(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Get response parameter list
  const Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");

  // Validate response parameter list
  Teuchos::ParameterList validPL;
  validPL.set<std::string>("Name", "", "Name Of The Response");
  validPL.set<std::string>("Side Set Name", "", "Side Set Name");
  validPL.set<std::string>("Field Rank", "Scalar", "Field Rank: Scalar, Vector, Gradient, Vector Gradient, Tensor");
  validPL.set<std::string>("Source Field Name", "", "Name of the Source field");
  validPL.set<std::string>("Target Field Name", "", "Name of the Target Field");
  validPL.set<std::string>("Root Mean Square Error Field Name", "", "Name of the RMSE for the Target");
  validPL.set<double>("Target Value", 1.0, "Constant Value of Target");
  validPL.set<double>("Scaling", 1.0, "Global Scaling of the Response, Default=1.0");
  validPL.set<bool>("Is Side Set Planar", false, "Whether the side set is planar");
  validPL.set<bool>("Response Depends On Solution Column", false, "");
  validPL.set<bool>("Response Depends On Extruded Parameters", false, "");
  plist->validateParameters(validPL,0);

  sideSetName = plist->get<std::string>("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);

  // Gathering dimensions
  sideDim = dl_side->cell_gradient->extent(1);
  numQPs  = dl_side->qp_scalar->extent(1);

  Teuchos::RCP<PHX::DataLayout> layout;
  std::string rank,fname;

  rank         = plist->get<std::string>("Field Rank");
  fname        = plist->get<std::string>("Source Field Name");

  fieldDim = getLayout(dl_side,rank,layout);

  layout->dimensions(dims);

  std::string sideSetNameForMetric =
      (plist->isParameter("Is Side Set Planar") && plist->get<bool>("Is Side Set Planar")) ?
          sideSetName + "_planar" :
          sideSetName;

  isFieldGradient = (rank == "Gradient") || (rank == "Vector Gradient");
  if (isFieldGradient)
  {
    metric = decltype(metric)(Albany::metric_name + "_" + sideSetNameForMetric, dl_side->qp_tensor);
    this->addDependentField(metric);
  }

  sourceField = decltype(sourceField)(fname,layout);
  w_measure   = decltype(w_measure)(Albany::weighted_measure_name + "_" + sideSetNameForMetric, dl_side->qp_scalar);
  scaling     = plist->isParameter("Scaling") ? plist->get<double>("Scaling") : 1.0;

  this->addDependentField(sourceField);

  rmsScaling = plist->isParameter("Root Mean Square Error Field Name");
  if(rmsScaling) {
    //Only considering scalar rmsScaling
    rootMeanSquareField = decltype(rootMeanSquareField)(plist->get<std::string>("Root Mean Square Error Field Name"),dl_side->qp_scalar);
    this->addDependentField(rootMeanSquareField);
  }

  if (plist->isParameter("Target Field Name")) {
    TEUCHOS_TEST_FOR_EXCEPTION(plist->isParameter("Target Value"), std::logic_error,
                               "[ResponseSquaredL2DifferenceSideBase] Error! Both target value and target field provided.\n")
    std::string target_fname;
    target_fname = plist->get<std::string>("Target Field Name");
    targetField = decltype(targetField)(target_fname,layout);
    this->addDependentField(targetField);

    target_value = false;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(!plist->isParameter("Target Value"), std::logic_error,
                               "[ResponseSquaredL2DifferenceSideBase] Error! No target value or target field provided.\n")
    target_value = true;
    target_value_val = TargetScalarT(plist->get<double>("Target Value"));
  }
  this->addDependentField(w_measure);

  this->setName("Response Squared L2 Error Side" + PHX::print<EvalT>());

  if(plist->isParameter("Response Depends On Solution Column") && plist->get<bool>("Response Depends On Solution Column"))
    cell_topo = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem")->get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");

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

  extrudedParams = (plist->isParameter("Response Depends On Extruded Parameters") && plist->get<bool>("Response Depends On Extruded Parameters"));

  if(extrudedParams)
    PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::setup(p, dl);
  else
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

  if(extrudedParams)
    PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::postRegistrationSetup(d, fm);
  else
    PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response_eval, 0.0);

  // Do global initialization
  if(extrudedParams)
    PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::preEvaluate(workset);
  else
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

  if (fieldDim == 1) {
    diff_1.resize(dims[2]);
  } else if (fieldDim == 2) {
    diff_2.resize(dims[2]);
    for(size_t i=0; i<dims[2]; i++)
      diff_2[i].resize(dims[3]);
  }

  if (workset.sideSets->find(sideSetName) != workset.sideSets->end())
  {
    sideSet = workset.sideSetViews->at(sideSetName);

    switch (fieldDim)
    {
      case 0:
        for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
        {
          // Get the local data of cell
          const int cell = sideSet.elem_LID(sideSet_idx);

          ScalarT sum = 0;
          for (int qp=0; qp<numQPs; ++qp)
          {
            ScalarT sq = 0;
            // Computing squared difference at qp
            RealType rms = rmsScaling ? rootMeanSquareField(sideSet_idx,qp) : 1.0;
            TargetScalarT trgt = target_value ? target_value_val : targetField(sideSet_idx,qp);
            sq += std::pow((sourceField(sideSet_idx,qp)-trgt)/rms,2);
            sum += sq * w_measure(sideSet_idx,qp);
          }

          this->local_response_eval(cell, 0) = sum*scaling;
          this->global_response_eval(0) += sum*scaling;
        }
        break;
      case 1:
        for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
        {
          // Get the local data of cell
          const int cell = sideSet.elem_LID(sideSet_idx);

          ScalarT sum = 0;
          for (int qp=0; qp<numQPs; ++qp)
          {
            ScalarT sq = 0;
            RealType rms2 = rmsScaling ? std::pow(rootMeanSquareField(sideSet_idx,qp),2) : 1.0;
            // Computing squared difference at qp
            // Precompute differentce and access fields only n times (not n^2)
            for (size_t i=0; i<dims[2]; ++i)
              diff_1[i] = sourceField(sideSet_idx,qp,i) - (target_value ? target_value_val : targetField(sideSet_idx,qp,i));

            if(isFieldGradient){
              for (int i=0; i<sideDim; ++i)
                for (int j=0; j<sideDim; ++j)
                  sq += diff_1[i]*metric(sideSet_idx,qp,i,j)*diff_1[j];
            } else {
              for (size_t i=0; i<dims[2]; ++i)
                sq += diff_1[i]*diff_1[i];
            }
            sum += sq / rms2 * w_measure(sideSet_idx,qp);
          }

          this->local_response_eval(cell, 0) = sum*scaling;
          this->global_response_eval(0) += sum*scaling;
        }
        break;
      case 2:
        for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
        {
          // Get the local data of cell
          const int cell = sideSet.elem_LID(sideSet_idx);

          ScalarT sum = 0;
          for (int qp=0; qp<numQPs; ++qp)
          {
            ScalarT sq = 0;
            RealType rms2 = rmsScaling ? std::pow(rootMeanSquareField(sideSet_idx,qp),2) : 1.0;
            // Computing squared difference at qp
            // Precompute differentce and access fields only n^2 times (not n^4)
            for (size_t i=0; i<dims[2]; ++i)
              for (size_t j=0; j<dims[3]; ++j)
                diff_2[i][j] = sourceField(sideSet_idx,qp,i,j) - (target_value ? target_value_val : targetField(sideSet_idx,qp,i,j));

            if(isFieldGradient){
              for (size_t i=0; i<dims[2]; ++i)
                for (int j=0; j<sideDim; ++j)
                  for (int k=0; k<sideDim; ++k)
                    sq += diff_2[i][j] * metric(sideSet_idx,qp,j,k) * diff_2[i][k];
            } else {
              for (size_t i=0; i<dims[2]; ++i)
                for (size_t j=0; j<dims[3]; ++j)
                  sq += diff_2[i][j] * diff_2[i][j];
            }

            sum += sq / rms2 * w_measure(sideSet_idx,qp);
          }

          this->local_response_eval(cell, 0) = sum*scaling;
          this->global_response_eval(0) += sum*scaling;
        }
        break;
    }

  }

  // Do any local-scattering necessary
  if(extrudedParams)
    PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::evaluateFields(workset);
  else
    PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);

  if(Teuchos::nonnull(cell_topo))
    PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::evaluate2DFieldsDerivativesDueToColumnContraction(workset,sideSetName, cell_topo);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceSideBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response_eval);

//  if(workset.comm->getRank()==0)
//    std::cout << "resp" << PHX::print<EvalT>() << ": " << this->global_response_eval(0) << "\n" << std::flush;

  // Do global scattering
  if(extrudedParams)
    PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::postEvaluate(workset);
  else
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
  else if (rank=="Vector Gradient")
  {
    layout = dl->qp_vecgradient;
    dim = 2;
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
