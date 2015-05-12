//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_SAVE_SIDE_SET_STATE_FIELD_HPP
#define FELIX_SAVE_SIDE_SET_STATE_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

namespace FELIX
{
/** \brief SaveSideSetStatField

*/

template<typename EvalT, typename Traits>
class SaveSideSetStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  SaveSideSetStateField (const Teuchos::ParameterList& p,
                         const Albany::MeshSpecsStruct& meshSpecs);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData workset);

  typedef typename EvalT::ScalarT ScalarT;
};


template<typename Traits>
class SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>
                    : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Residual, Traits>
{
public:

  SaveSideSetStateField (const Teuchos::ParameterList& p,
                         const Albany::MeshSpecsStruct& meshSpecs);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

  Teuchos::RCP<PHX::FieldTag> savestate_operation;
  PHX::MDField<ScalarT>       field;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;

  std::string sideSetName;
  std::string cellFieldName;
  std::string sideStateName;

  int sideDims;
  int numSideNodes;
};

} // Namespace FELIX

#endif // FELIX_SAVE_SIDE_SET_STATE_FIELD_HPP
