//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_CELL_TO_SIDE_HPP
#define PHAL_DOF_CELL_TO_SIDE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {
/** \brief Finite Element CellToSide Evaluator

    This evaluator creates a field defined cell-side wise from a cell wise field

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFCellToSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFCellToSideBase (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  std::string                     sideSetName;
  std::vector<int>                dims;

  Kokkos::View<int**, PHX::Device> sideNodes;

  Albany::LocalSideSetInfo sideSet;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT> val_cell;

  // Output:
  //! Values on side
  PHX::MDField<ScalarT> val_side;

  // TODO: Need to update collapsed layouts and add kernels for remaining layout types when removing non-collapsed layouts
  enum LayoutType
  {
    CELL_SCALAR = 1,
    CELL_SCALAR_SIDESET,
    CELL_VECTOR,
    CELL_TENSOR,
    NODE_SCALAR,
    NODE_SCALAR_SIDESET,
    NODE_VECTOR,
    NODE_VECTOR_SIDESET,
    NODE_TENSOR,
    VERTEX_VECTOR,
    VERTEX_VECTOR_SIDESET
  };

  LayoutType layout;

  MDFieldMemoizer<Traits> memoizer;

  int dimsArray[5];

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct CellScalarSideset_Tag{};
  struct NodeScalarSideset_Tag{};
  struct NodeVectorSideset_Tag{};
  struct VertexVectorSideset_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, CellScalarSideset_Tag> CellScalarSideset_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, NodeScalarSideset_Tag> NodeScalarSideset_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, NodeVectorSideset_Tag> NodeVectorSideset_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, VertexVectorSideset_Tag> VertexVectorSideset_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const CellScalarSideset_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const NodeScalarSideset_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const NodeVectorSideset_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const VertexVectorSideset_Tag& tag, const int& sideSet_idx) const;

};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFCellToSide = DOFCellToSideBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFCellToSideMesh = DOFCellToSideBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFCellToSideParam = DOFCellToSideBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // DOF_CELL_TO_SIDE_HPP
