//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CM_Residual_Fine_hpp)
#define LCM_CM_Residual_Fine_hpp

#include <Phalanx_config.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "Albany_Layouts.hpp"

namespace LCM
{
///
/// \brief Concurrent Multiscale Residual Fine
///
/// Residual for coarse scale for concurrent multiscale
///
template<typename EvalT, typename Traits>
class CMResidualFine:
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  ///
  CMResidualFine(
      Teuchos::ParameterList & p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits> & vm);

  ///
  /// Implementation of mechanics
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

};

}

#endif // LCM_CM_Residual_Fine_hpp
