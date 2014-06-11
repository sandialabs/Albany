//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_FirstPK_hpp)
#define LCM_FirstPK_hpp

#include <Phalanx_ConfigDefs.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "Albany_Layouts.hpp"

namespace LCM
{
///
/// \brief First Piola-Kirchhoff Stress
///
/// This evaluator computes the first PK stress from the deformation gradient
/// and Cauchy stress, and optionally volume averages the pressure
///
template<typename EvalT, typename Traits>
class FirstPK:
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  ///
  FirstPK(Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

private:

  ///
  /// Input: Cauchy Stress
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress_;

  ///
  /// Input: Deformation Gradient
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> def_grad_;

  ///
  /// Output: First PK stress
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> first_pk_stress_;

  ///
  /// Optional
  /// Input: Pore Pressure
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> pore_pressure_;

  ///
  /// Optional
  /// Input: Biot Coefficient
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> biot_coeff_;

  ///
  /// Optional
  /// Input: Stabilized Pressure
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> stab_pressure_;

  ///
  /// Number of integration points
  ///
  std::size_t num_pts_;

  ///
  /// Number of spatial dimensions
  ///
  std::size_t num_dims_;

  ///
  /// Pore Pressure flag
  ///
  bool have_pore_pressure_;

  ///
  /// Stabilized Pressure flag
  ///
  bool have_stab_pressure_;

  ///
  /// Small Strain flag
  ///
  bool small_strain_;

};
}

#endif
