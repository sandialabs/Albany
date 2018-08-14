//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SETFIELD_HPP
#define SETFIELD_HPP

#include "PHAL_Dimension.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief Sets values in a field, indended for testing.
 */

template <typename EvalT, typename Traits>
class SetField : public PHX::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  SetField(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT ScalarT;

  //! The name of the field to be set.
  std::string evaluatedFieldName;

  //! The field that will be set.
  // PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> evaluatedField;
  PHX::MDField<ScalarT> evaluatedField;

  //! The dimensions of the field to be set.
  std::vector<PHX::DataLayout::size_type> evaluatedFieldDimensions;

  //! The values that will be assigned to the field
  Teuchos::ArrayRCP<ScalarT> fieldValues;
};
}  // namespace LCM

#endif
