//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Mechanics_Residual_hpp)
#define LCM_Mechanics_Residual_hpp

#include <Phalanx_ConfigDefs.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "Albany_Layouts.hpp"

namespace LCM
{
///
/// \brief Mechanics Residual
///
/// This evaluator computes the residual due to the balance
/// of linear momentum for infinitesimal and finite deformation,
/// with or without dynamics
///
template<typename EvalT, typename Traits>
class MechanicsResidual:
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  ///
  MechanicsResidual(Teuchos::ParameterList& p,
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
  /// Input: Weighted Basis Function Gradients
  ///
  PHX::MDField<MeshScalarT, Cell, Node, QuadPoint, Dim> w_grad_bf_;

  ///
  /// Input: Weighted Basis Functions
  ///
  PHX::MDField<MeshScalarT, Cell, Node, QuadPoint> w_bf_;

  ///
  /// Input: body force vector
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> body_force_;

  ///
  /// Input: acceleration
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> acceleration_;

  ///
  /// Output: Residual Forces
  ///
  PHX::MDField<ScalarT, Cell, Node, Dim> residual_;

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
  /// Body force flag
  ///
  bool have_body_force_;

  ///
  /// Density
  ///
  RealType density_;

  ///
  /// Dynamics flag
  ///
  bool enable_dynamics_;
};
}

#endif
