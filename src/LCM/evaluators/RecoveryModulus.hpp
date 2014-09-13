//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef RECOVERY_MOUDULS_HPP
#define RECOVERY_MOUDULS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

namespace LCM {

  ///
  /// \brief Evaluates recovery modulus, either as a constant, or temperature dependent
  ///
  template<typename EvalT, typename Traits>
  class RecoveryModulus :
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
  
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    RecoveryModulus(Teuchos::ParameterList& p);
  
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);
  
    void evaluateFields(typename Traits::EvalData d);
  
    ScalarT& getValue(const std::string &n);

  private:

    std::size_t numQPs;
    std::size_t numDims;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<ScalarT,Cell,QuadPoint> recoveryModulus;

    //! Is recoveryModulus constant, or random field
    bool is_constant;

    //! Constant value
    ScalarT constant_value;

    //! Optional dependence on Temperature Rd = c1 * exp(c2/Temperature)
    PHX::MDField<ScalarT,Cell,QuadPoint> Temperature;
    bool isThermoElastic;
    ScalarT c1;
    ScalarT c2;
    RealType refTemp;

    //! Exponential random field
    Teuchos::RCP< Stokhos::KL::ExponentialRandomField<MeshScalarT> > exp_rf_kl;

    //! Values of the random variables
    Teuchos::Array<ScalarT> rv;
  };
}

#endif
