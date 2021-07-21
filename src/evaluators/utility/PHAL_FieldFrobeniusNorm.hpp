//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_FIELD_NORM_HPP
#define PHAL_FIELD_NORM_HPP 1

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

template<typename EvalT, typename Traits, typename ScalarT>
class FieldFrobeniusNormBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  FieldFrobeniusNormBase (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell (typename Traits::EvalData d);
  void evaluateFieldsSide (typename Traits::EvalData d);

  // The parameter is always defined in terms of the
  // evaluation type native scalar type (otherwise
  // the evaluation manager does not update it).
  typedef typename EvalT::ScalarT       EScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  enum RegularizationType { NONE=1, GIVEN_VALUE, GIVEN_PARAMETER, PARAMETER_EXPONENTIAL};

  RegularizationType            regularization_type;
  ScalarT                       regularization;
  ScalarT                       printedReg;

  // Input:
  PHX::MDField<const EScalarT,Dim>    regularizationParam;
  PHX::MDField<const ScalarT> field;

  // Output:
  PHX::MDField<ScalarT> field_norm;

  bool eval_on_side;
  Albany::LocalSideSetInfo sideSet;

  std::string sideSetName;
  std::vector<PHX::DataLayout::size_type> dims;
  int numDims;
  PHX::DataLayout::size_type dimsArray[4];

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct Dim2_Tag{};
  struct Dim3_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, Dim2_Tag> Dim2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, Dim3_Tag> Dim3_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Dim2_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Dim3_Tag& tag, const int& sideSet_idx) const;

};

// Some shortcut names
template<typename EvalT, typename Traits>
using FieldFrobeniusNorm = FieldFrobeniusNormBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using FieldFrobeniusNormMesh = FieldFrobeniusNormBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using FieldFrobeniusNormParam = FieldFrobeniusNormBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // namespace PHAL

#endif // PHAL_FIELD_NORM_HPP
