//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Damage_Coefficients_hpp)
#define LCM_Damage_Coefficients_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM
{
/// \brief
///
/// This evaluator computes the coefficients for the damage equation
///
template<typename EvalT, typename Traits>
class DamageCoefficients: public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  ///
  /// Constructor
  ///
  DamageCoefficients(Teuchos::ParameterList& p,
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
  PHX::MDField<ScalarT, Cell, QuadPoint> damage_;

  ///
  /// Input: tiem step
  ///
  PHX::MDField<ScalarT, Dummy> delta_time_;

  ///
  /// Optional: deformation gradient
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> def_grad_;

  ///
  /// Output: damage transient coefficient
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> damage_transient_coeff_;

  ///
  /// Output: damage Diffusivity
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> damage_diffusivity_;

  ///
  /// Output: damage dot
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> damage_dot_;

  ///
  /// Number of integration points
  ///
  int num_pts_;

  ///
  /// Number of spatial dimesions
  ///
  int num_dims_;

  ///
  /// Damage Constants
  ///
  RealType diffusivity_coeff_, transient_coeff_;

  ///
  /// Mechanics flag
  ///
  bool have_mech_;

  ///
  /// String for the damage variable name
  ///
  std::string damage_name_;

};
}

#endif
