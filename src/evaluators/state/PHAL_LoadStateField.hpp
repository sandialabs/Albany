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

#include "PHAL_Utilities.hpp"

namespace PHAL {
/** \brief LoadStateField

*/

template<typename EvalT, typename Traits, typename ScalarType = RealType>
class LoadStateFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>  {
public:
  
  LoadStateFieldBase(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  PHX::MDField<ScalarType> data;
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

  PHX::MDField<ParamScalarT> data;
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

//Because LoadStateField is used with Sacado::mpl (in PHAL_FactoryTraits), when we enable the follow lines, we get the error: no type named ‘ParamScalarT’ in ‘struct Sacado::mpl::arg<-1>.
//For this reason we left the original implementation of LoadStateField
// LB: The LoadStateField version still requires EvalT to have a ParamScalarT type defined inside, so how can LoadStateField be ok? I guess inside Sacado::mpl, LoadStateField is not
//     really instantiated for EvalT = Sacado::mpl::arg<-1>...

// template<typename EvalT, typename Traits>
// using LoadStateField = LoadStateFieldBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // namespace PHAL

#endif // PHAL_LOAD_STATE_FIELD_HPP
