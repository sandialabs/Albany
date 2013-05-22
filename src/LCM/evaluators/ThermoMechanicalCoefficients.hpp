//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ThermoMechanical_Coefficients_hpp)
#define LCM_ThermoMechanical_Coefficients_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief
  ///
  /// This evaluator computes the hydrogen concentration at trapped site
  /// through conservation of hydrogen atom
  ///
  template<typename EvalT, typename Traits>
  class ThermoMechanicalCoefficients : public PHX::EvaluatorWithBaseImpl<Traits>,
                                       public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    ThermoMechanicalCoefficients(Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Input: temperature
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> temperature_;

    ///
    /// Input: thermal conductivity
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> thermal_cond_;

    ///
    /// Output: deformation gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> def_grad_;

    ///
    /// Output: thermal transient coefficient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> thermal_transient_coeff_;

    ///
    /// Output: thermal Diffusivity
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> thermal_diffusivity_;

    ///
    /// Output: stress
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress_;

    ///
    /// Output: stress
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> source_;

    ///
    /// Number of integration points
    ///
    std::size_t num_pts_;

    ///
    /// Number of spatial dimesions
    ///
    std::size_t num_dims_;

    ///
    /// Thermal Constants
    ///
    RealType heat_capacity_, density_, transient_coeff_;
    RealType ref_temperature_, expansion_coeff_;

    ///
    /// Mechanics flag
    ///
    bool have_mech_;
  };
}

#endif
