//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Mechanics_Residual_hpp)
#define LCM_Mechanics_Residual_hpp

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "AAdapt_RC_Field.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
///
/// \brief Mechanics Residual
///
/// This evaluator computes the residual due to the balance
/// of linear momentum for infinitesimal and finite deformation,
/// with or without dynamics
///
template <typename EvalT, typename Traits>
class MechanicsResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  ///
  /// Constructor
  ///
  MechanicsResidual(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
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
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> stress_;

  ///
  /// Input: Weighted Basis Function Gradients
  ///
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> w_grad_bf_;

  ///
  /// Input: Weighted Basis Functions
  ///
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> w_bf_;

  ///
  /// Input: body force vector
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> body_force_;

  ///
  /// Input: acceleration
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> acceleration_;

  ///
  /// Input: mass contribution to residual/Jacobian (if not using AD to compute
  /// mass matrix)
  ///
  PHX::MDField<const ScalarT, Cell, Node, Dim> mass_;

  ///
  /// Output: Residual Forces
  ///
  PHX::MDField<ScalarT, Cell, Node, Dim> residual_;

  ///
  /// Number of element nodes
  ///
  int num_nodes_;

  ///
  /// Number of integration points
  ///
  int num_pts_;

  ///
  /// Number of spatial dimensions
  ///
  int num_dims_;

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

  ///
  /// Flag to mark if using mass from AnalyticMassResidual evaluator
  ///
  bool use_analytic_mass_;

  ///
  /// Input, if RCU.
  ///
  AAdapt::rc::Field<2> def_grad_rc_;

 public:  // Kokkos
  struct residual_Tag
  {
  };
  struct residual_haveBodyForce_Tag
  {
  };
  struct residual_haveBodyForce_and_dynamic_Tag
  {
  };
  struct residual_have_dynamic_Tag
  {
  };

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy<ExecutionSpace, residual_Tag> residual_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, residual_haveBodyForce_Tag>
      residual_haveBodyForce_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, residual_haveBodyForce_and_dynamic_Tag>
          residual_haveBodyForce_and_dynamic_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, residual_have_dynamic_Tag>
      residual_have_dynamic_Policy;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_haveBodyForce_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_haveBodyForce_and_dynamic_Tag& tag, const int& i)
      const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const residual_have_dynamic_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void
  compute_Stress(const int cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  compute_BodyForce(const int cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  compute_Acceleration(const int cell) const;
};
}  // namespace LCM

#endif
