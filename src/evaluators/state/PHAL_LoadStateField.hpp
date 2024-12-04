//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LOAD_STATE_FIELD_HPP
#define PHAL_LOAD_STATE_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {
/** \brief LoadStateField

*/

template<typename EvalT, typename Traits, typename ScalarType>
class LoadStateFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>  {
public:
  
  LoadStateFieldBase(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  using ExecutionSpace = typename PHX::Device::execution_space;

  PHX::MDField<ScalarType> field;
  std::string fieldName;
  std::string stateName;

  MDFieldMemoizer<Traits> memoizer;
};

template<typename EvalT, typename Traits>
class LoadStateField : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  LoadStateField(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  using ExecutionSpace = typename PHX::Device::execution_space;

  PHX::MDField<ParamScalarT> field;
  std::string fieldName;
  std::string stateName;

  MDFieldMemoizer<Traits> memoizer;
};

// Shortcut names
template<typename EvalT, typename Traits>
using LoadStateFieldST = LoadStateFieldBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using LoadStateFieldPST = LoadStateFieldBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using LoadStateFieldMST = LoadStateFieldBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using LoadStateFieldRT = LoadStateFieldBase<EvalT,Traits,RealType>;

} // namespace PHAL

#endif // PHAL_LOAD_STATE_FIELD_HPP
