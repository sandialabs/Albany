//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Gradient_Element_Length_hpp)
#define LCM_Gradient_Element_Length_hpp

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>

#include "Albany_Layouts.hpp"

namespace LCM {
/// \brief
///
/// Compute element length in the direction of the solution gradient
/// (cf. Tezduyar and Park CMAME 1986).
///
template <typename EvalT, typename Traits>
class GradientElementLength : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  GradientElementLength(
      const Teuchos::ParameterList&        p,
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

  ///
  /// Input: unit scalar gradient
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> unit_grad_;

  ///
  /// Input: basis function gradients
  ///
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> grad_bf_;

  ///
  /// Output: element length
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> element_length_;

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
