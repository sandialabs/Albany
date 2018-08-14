//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_NodePointVecInterpolation_hpp)
#define LCM_NodePointVecInterpolation_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {

/// Finite Element Interpolation Evaluator
/// This evaluator interpolates nodal DOFVec values an arbitrary
/// point within the cell.
template <typename EvalT, typename Traits>
class NodePointVecInterpolation : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  NodePointVecInterpolation(
      Teuchos::ParameterList const&        p,
      Teuchos::RCP<Albany::Layouts> const& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT, Cell, Node, VecDim> nodal_value_;

  //! Basis Functions
  PHX::MDField<const RealType, Cell, Node, Point> basis_fn_;

  // Output:
  //! Value at a point
  PHX::MDField<ScalarT, Cell, Point, VecDim> point_value_;

  int number_nodes_;

  int number_points_;

  int dimension_;
};

//! Specialization for Jacobian evaluation taking advantage of known sparsity
template <typename Traits>
class NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  NodePointVecInterpolation(
      Teuchos::ParameterList const&        p,
      Teuchos::RCP<Albany::Layouts> const& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT, Cell, Node, VecDim> nodal_value_;

  //! Basis Functions
  PHX::MDField<const RealType, Cell, Node> basis_fn_;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT, Cell, Point> point_value_;

  int number_nodes_;

  int number_points_;

  int dimension_;

  int offset_;
};

}  // namespace LCM

#endif  // LCM_NodePointVecInterpolation_hpp
