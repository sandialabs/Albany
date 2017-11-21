//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LOADSTATEFIELD_HPP
#define PHAL_LOADSTATEFIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief LoadStateField

*/

template<typename EvalT, typename Traits, typename ScalarT>
class LoadStateFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  LoadStateFieldBase(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:


  PHX::MDField<ScalarT> data;
  std::string fieldName;
  std::string stateName;
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
};


template<typename EvalT, typename Traits>
using LoadStateFieldST = LoadStateFieldBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using LoadStateFieldPST = LoadStateFieldBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using LoadStateFieldRT = LoadStateFieldBase<EvalT,Traits,RealType>;

//Because LoadStateField is used with Sacado::mpl (in PHAL_Albany_Traits), when we enable the follow lines, we get the error: no type named ‘ParamScalarT’ in ‘struct Sacado::mpl::arg<-1>.
//For this reason we left the original implementation of LoadStateField

//template<typename EvalT, typename Traits>
//using LoadStateField = LoadStateFieldBase<EvalT,Traits,typename EvalT::ScalarT>;

}

#endif
