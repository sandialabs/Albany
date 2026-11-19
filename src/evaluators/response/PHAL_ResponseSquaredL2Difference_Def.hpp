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
  numQPs   = dl->qp_scalar->extent(1);

  Teuchos::RCP<PHX::DataLayout> layout;
  std::string rank,fname;

  rank           = plist->get<std::string>("Field Rank");
  fname          = plist->get<std::string>("Source Field Name");

  fieldDim = getLayout(dl,rank,layout);
  layout->dimensions(dims);
  dims_2 = (fieldDim >= 1) ? dims[2] : 0;
  dims_3 = (fieldDim >= 2) ? dims[3] : 0;

  sourceField = decltype(sourceField)(fname,layout);
  w_measure   = decltype(w_measure)("Weights",dl->qp_scalar);
  scaling     = plist->get("Scaling",1.0);

  this->addDependentField(sourceField);
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

  this->setName("Response Squared L2 Error " + PHX::print<EvalT>());

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response Squared L2 Error ";
  std::string global_response_name = "Global Response Squared L2 Error ";
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
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sourceField,fm);
  this->utils.setFieldData(w_measure,fm);

  if (!target_value) {
    this->utils.setFieldData(targetField,fm);
  }

  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::preEvaluate(typename Traits::PreEvalData workset)
{
  Kokkos::deep_copy(this->global_response_eval.get_view(), 0.0);

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  Kokkos::deep_copy(this->local_response_eval.get_view(), 0.0);

  // Capture only device-copyable data in the kernel (not the whole evaluator)
  const auto local    = this->local_response_eval.get_view();
  const auto source   = sourceField.get_view();
  const auto target   = targetField.get_view();
  const auto measure  = w_measure.get_static_view();
  const bool use_target_value = target_value;
  const TargetScalarT target_val = target_value_val;
  const RealType scale = scaling;
  const int nqp   = numQPs;
  const int fdim  = fieldDim;
  const int ndim2 = dims_2;
  const int ndim3 = dims_3;

  // Compute the contribution of each cell
  Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_LAMBDA(const int& cell) {
    ScalarT sum = 0;
    for (int qp=0; qp<nqp; ++qp)
    {
      ScalarT sq = 0;
      // Computing squared difference at qp
      switch (fdim) {
        case 0:
          sq += std::pow(source(cell,qp)-(use_target_value ? target_val : target(cell,qp)),2);
          break;
        case 1:
          for (int j=0; j<ndim2; ++j)
            sq += std::pow(source(cell,qp,j)-(use_target_value ? target_val : target(cell,qp,j)),2);
          break;
        case 2:
          for (int j=0; j<ndim2; ++j)
            for (int k=0; k<ndim3; ++k)
              sq += std::pow(source(cell,qp,j,k)-(use_target_value ? target_val : target(cell,qp,j,k)),2);
          break;
      }
      sum += sq * measure(cell,qp);
    }
    local(cell, 0) = sum*scale;
  });

  // Sum the cell contributions into the global response
  const auto global = this->global_response_eval.get_view();
  constexpr bool fixed_size =
      Sacado::IsStaticallySized<ScalarT>::value &&
      std::is_same<typename Sacado::ScalarType<ScalarT>::type,
                   typename Sacado::ValueType<ScalarT>::type>::value;
  if constexpr (fixed_size) {
    // Value and derivatives are reduced as an array of doubles: this avoids
    // contended (and non-deterministic) atomics on the same entry.
    constexpr int ncomp = Sacado::StaticSize<ScalarT>::value + 1;
    if (m_sums.extent_int(0)!=ncomp) {
      m_sums = Kokkos::View<RealType*, PHX::Device>("sums",ncomp);
    }
    const auto sums = m_sums;
    Kokkos::parallel_reduce(this->getName(),RangePolicy(0,workset.numCells),
                            SumScalarComponents<decltype(local),ncomp>{local},sums);
    Kokkos::parallel_for(this->getName(),RangePolicy(0,1),
                         AddScalarComponents<decltype(global),decltype(sums),ncomp>{global,sums});
  } else {
    // Dynamically sized (e.g., SLFad) or nested (HessianVec) AD types: Sacado provides no
    // Kokkos reduction for these, so fall back to atomics. Note: these are contended (all
    // cells add to the same entry), and the summation order is not deterministic.
    Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                         KOKKOS_LAMBDA(const int& cell) {
      KU::atomic_add<ExecutionSpace>(&(global(0)), local(cell,0));
    });
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
void PHAL::ResponseSquaredL2DifferenceBase<EvalT, Traits, SourceScalarT, TargetScalarT>::postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response_eval);

  //if(workset.comm->getRank()==0)
  //  std::cout << "resp" << PHX::print<EvalT>() << ": " << this->global_response_eval(0) << "\n" << std::flush;

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
