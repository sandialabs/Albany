//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP
#define PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_Layouts.hpp"

namespace PHAL
{
/** \brief SaveSideSetStatField

*/

// Default impl is a do-nothing one; states are only saved with the Residual
// evaluation type, which is impl-ed as a specialization of this class
template<typename EvalT, typename Traits>
class SaveSideSetStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  SaveSideSetStateField (const Teuchos::ParameterList&,
                         const Teuchos::RCP<Albany::Layouts>&) {}

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData) {}
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

  void saveElemState (typename Traits::EvalData d);
  void saveNodeState (typename Traits::EvalData d);

  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;
  using MeshScalarT = typename PHAL::AlbanyTraits::Residual::MeshScalarT;
  using FRT = Albany::FieldRankType;
  using FL  = Albany::FieldLocation;

  Teuchos::RCP<PHX::FieldTag> savestate_operation;
  PHX::MDField<const ScalarT>       field;
  // For a covariant field V (i.e., a Gradient), save tangents*V.
  // WARNING: we actually save the first sideDim components of tangents*V.
  //          This is ok for xy side sets, but not for general sidesets.
  // TODO: fix this by registering the stk field as a numDims vector field,
  //       rather than a numSideDims one.
  PHX::MDField<const MeshScalarT>   tangents;
  PHX::MDField<const MeshScalarT>   w_measure;

  std::string sideSetName;
  std::string fieldName;
  std::string stateName;

  FRT rank;
  FL  loc;

  int numQPs;
  int numNodes;

  MDFieldMemoizer<Traits> memoizer;
};

} // Namespace PHAL

#endif // PHAL_SAVE_SIDE_SET_STATE_FIELD_HPP
