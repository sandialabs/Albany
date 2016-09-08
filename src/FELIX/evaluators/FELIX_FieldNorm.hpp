//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_FIELD_NORM_HPP
#define FELIX_FIELD_NORM_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

template<typename EvalT, typename Traits, typename ScalarT>
class FieldNormBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  FieldNormBase (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  // The parameter is always defined in terms of the
  // evaluation type native scalar type (otherwise
  // the evaluation manager does not update it).
  typedef typename EvalT::ScalarT       EScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  enum RegularizationType { NONE=1, GIVEN_VALUE, GIVEN_PARAMETER, PARAMETER_EXPONENTIAL};

  PHX::MDField<EScalarT,Dim>    regularizationParam;
  RegularizationType            regularization_type;
  ScalarT                       regularization;
  ScalarT                       printedReg;

  // Input:
  PHX::MDField<ScalarT> field;

  // Output:
  PHX::MDField<ScalarT> field_norm;

  std::string sideSetName;
  std::vector<PHX::DataLayout::size_type> dims;
  int numDims;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using FieldNorm = FieldNormBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using FieldNormMesh = FieldNormBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using FieldNormParam = FieldNormBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace FELIX

#endif // FELIX_FIELD_NORM_HPP
