//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_LOAD_SIDE_SET_STATE_FIELD_HPP
#define FELIX_LOAD_SIDE_SET_STATE_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

namespace FELIX
{
/** \brief LoadSideSetStatField

*/

template<typename EvalT, typename Traits>
class LoadSideSetStateField<EvalT, Traits>
                    : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  LoadSideSetStateField (const Teuchos::ParameterList& p,
                         const Albany::MeshSpecsStruct& meshSpecs);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  PHX::MDField<ScalarT>              field;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;

  std::string sideSetName;
  std::string cellFieldName;
  std::string sideStateName;

  int sideDims;
  int numSideNodes;
};

} // Namespace FELIX

#endif // FELIX_LOAD_SIDE_SET_STATE_FIELD_HPP
