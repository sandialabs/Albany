//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_READSTATEFIELD_HPP
#define PHAL_READSTATEFIELD_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"

namespace PHAL {
///
/// ReadStateField
///
template <typename EvalT, typename Traits>
class ReadStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ReadStateField(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm);

  void
  evaluateFields(typename Traits::EvalData d);
};

template <typename Traits>
class ReadStateField<PHAL::AlbanyTraits::Residual, Traits>
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  ReadStateField(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& fm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  void
  readNodalState(typename Traits::EvalData d);

  void
  readElemState(typename Traits::EvalData d);

  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;

  Teuchos::RCP<PHX::FieldTag> read_state_op{Teuchos::null};
  PHX::MDField<ScalarT>       field;
  std::string                 field_name;
  std::string                 state_name;
  std::string                 field_type;
};

}  // Namespace PHAL

#endif  // PHAL_READSTATEFIELD_HPP
