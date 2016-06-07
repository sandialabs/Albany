//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AMP_TIME_HPP
#define AMP_TIME_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"

namespace AMP {
///
/// \brief evaluates the current time and delta time
///
template<typename EvalT, typename Traits>
class Time : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{

  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    Time(Teuchos::ParameterList& p);

    void postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

    ScalarT& getValue(const std::string &n);

  private:

    //! Constant value
    PHX::MDField<ScalarT,Dummy> time;
    PHX::MDField<ScalarT,Dummy> deltaTime;
    ScalarT timeValue;

    bool enableTransient;
    std::string timeName;
};

}

#endif
