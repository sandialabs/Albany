//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SAVESTATEFIELD_HPP
#define PHAL_SAVESTATEFIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief SaveStateField

*/

template<typename EvalT, typename Traits>
class SaveStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  SaveStateField(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
};


template<typename Traits>
class SaveStateField<PHAL::AlbanyTraits::Residual, Traits>
                    : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Residual, Traits>  {

public:

  SaveStateField(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void saveElemState (typename Traits::EvalData d);
  void saveNodeState (typename Traits::EvalData d);

  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

  Teuchos::RCP<PHX::FieldTag> savestate_operation;
  PHX::MDField<const ScalarT> field;
  std::string fieldName;
  std::string stateName;

  bool nodalState;
};

} // Namespace PHAL

#endif // PHAL_SAVESTATEFIELD_HPP
