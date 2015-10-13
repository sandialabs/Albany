//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP
#define PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

namespace PHAL
{
/** \brief SaveSideSetStatField

*/

template<typename EvalT, typename Traits>
class SaveSideSetStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  SaveSideSetStateField (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData workset);
};

// =========================== SPECIALIZATION ========================= //

template<typename Traits>
class SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>
                    : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Residual, Traits>
{
public:

  SaveSideSetStateField (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

  Teuchos::RCP<PHX::FieldTag> savestate_operation;
  PHX::MDField<ScalarT>       field;

  bool        isVectorField;

  std::string sideSetName;
  std::string fieldName;
  std::string stateName;
};

} // Namespace PHAL

#endif // PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP
