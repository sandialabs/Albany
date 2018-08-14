//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef DENSITY_HPP
#define DENSITY_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
/**
 * \brief Evaluates density.
 */

template <typename EvalT, typename Traits>
class Density : public PHX::EvaluatorWithBaseImpl<Traits>,
                public PHX::EvaluatorDerived<EvalT, Traits>,
                public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  Density(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  ScalarT&
  getValue(const std::string& n);

 private:
  PHX::MDField<ScalarT, Cell> density;

  //! Constant value
  ScalarT constant_value;
};
}  // namespace LCM

#endif
