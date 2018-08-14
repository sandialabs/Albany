//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Electrostatic_Residual_hpp)
#define LCM_Electrostatic_Residual_hpp

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "Albany_Layouts.hpp"

namespace LCM {
///
/// \brief Electrostatic Residual
///
template <typename EvalT, typename Traits>
class ElectrostaticResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  ///
  ElectrostaticResidual(
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
  /// Input: Electric Displacement
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> edisp_;

  ///
  /// Input: Weighted Basis Function Gradients
  ///
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> w_grad_bf_;

  ///
  /// Output: Residual Charges
  ///
  PHX::MDField<ScalarT, Cell, Node> residual_;

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
};
}  // namespace LCM

#endif
