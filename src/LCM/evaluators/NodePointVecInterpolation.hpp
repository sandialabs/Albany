//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_NodePointVecInterpolation_hpp)
#define LCM_NodePointVecInterpolation_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {

/// Finite Element Interpolation Evaluator
/// This evaluator interpolates nodal DOFVec values an arbitrary
/// point within the cell.

template<typename EvalT, typename Traits>
class NodePointVecInterpolation
: public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  NodePointVecInterpolation(
      Teuchos::ParameterList const & p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  void postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits> & vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT, Cell, Node, VecDim>
  nodal_value;

  //! Basis Functions
  PHX::MDField<RealType, Cell, Node, Point> shape_fn;

  // Output:
  //! Value at a points
  PHX::MDField<ScalarT, Cell, Point, VecDim> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t vecDim;
};

//! Specialization for Jacobian evaluation taking advantage of known sparsity
template<typename Traits>
class NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>
: public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Jacobian, Traits>
{

public:

  NodePointVecInterpolation(const Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT, Cell, Node, VecDim> nodal_value;
  //! Basis Functions
  PHX::MDField<RealType, Cell, Node, Point> shape_fn;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT, Cell, Point, VecDim> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t vecDim;
  std::size_t offset;
};

}

#endif // LCM_NodePointVecInterpolation_hpp
