//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef KINEMATICS_HPP
#define KINEMATICS_HPP
//#ifndef DEFGRAD_HPP
//#define DEFGRAD_HPP

#include "AAdapt_RC_Field.hpp"
#include "Albany_Layouts.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/// \brief Kinematics Evaluator
///
///  This evaluator computes kinematics quantities i.e.
///  Deformation Gradient
///  (optional) Velocity Gradient
///  (optional) Strain
///
template <typename EvalT, typename Traits>
class Kinematics : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  Kinematics(
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
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  //! Input: displacement gradient
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> grad_u_;

  //! Input: integration weights
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> weights_;

  //! Output: deformation gradient
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> def_grad_;

  //! Output: determinant of the deformation gradient
  PHX::MDField<ScalarT, Cell, QuadPoint> j_;

  //! Output: velocity gradient
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> vel_grad_;

  //! Output: strain
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> strain_;

  //! number of integration points
  int num_pts_;

  //! number of spatial dimensions
  int num_dims_;

  //! flag to compute the weighted average of J
  bool weighted_average_;

  //! stabilization parameter for the weighted average
  ScalarT alpha_;

  //! flag to compute the velocity Gradient
  bool needs_vel_grad_;

  //! flag to compute the strain
  bool needs_strain_;

  ///! Input, if RCU.
  AAdapt::rc::Field<2> def_grad_rc_;
  // For debugging.
  PHX::MDField<const ScalarT, Cell, Vertex, Dim> u_;
  bool
  check_det(typename Traits::EvalData d, int cell, int pt);
};
}  // namespace LCM
#endif
