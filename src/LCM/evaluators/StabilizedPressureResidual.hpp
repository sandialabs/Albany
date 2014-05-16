//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_StabilizedPressure_Residual_hpp)
#define LCM_StabilizedPressure_Residual_hpp

#include <Phalanx_ConfigDefs.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "Albany_Layouts.hpp"

namespace LCM
{
///
/// \brief StabilizedPressure Residual
///
/// This evaluator computes the residual
/// for the equal order pressure stabilization 
///
template<typename EvalT, typename Traits>
class StabilizedPressureResidual:
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  ///
  StabilizedPressureResidual(Teuchos::ParameterList& p,
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
  /// Input: Shear Modulus
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> shear_modulus_;

  ///
  /// Input: Bulk Modulus
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> bulk_modulus_;

  ///
  /// Input: Deformation Gradient
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> def_grad_;

  ///
  /// Input: Cauchy Stress
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress_;

  ///
  /// Input: Pressure
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> pressure_;

  ///
  /// Input: Pressure
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> pressure_grad_;

  ///
  /// Input: Weighted Basis Function Gradients
  ///
  PHX::MDField<MeshScalarT, Cell, Node, QuadPoint, Dim> w_grad_bf_;

  ///
  /// Input: Weighted Basis Functions
  ///
  PHX::MDField<MeshScalarT, Cell, Node, QuadPoint> w_bf_;

  ///
  /// Input: Weighted Basis Functions
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> h_;

  ///
  /// Output: Residual Forces
  ///
  PHX::MDField<ScalarT, Cell, Node> residual_;

  ///
  /// Number of element nodes
  ///
  std::size_t num_nodes_;

  ///
  /// Number of integration points
  ///
  std::size_t num_pts_;

  ///
  /// Number of spatial dimensions
  ///
  std::size_t num_dims_;

  ///
  /// Small strain flag
  ///
  bool small_strain_;

  ///
  /// Stabilization parameter
  ///
  RealType alpha_;
};
}

#endif
