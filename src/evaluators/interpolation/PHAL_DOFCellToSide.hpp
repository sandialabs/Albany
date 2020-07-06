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

  Kokkos::DynRankView<int, PHX::Device> sideNodes;

  Albany::SideStructViews sideSet;

  Teuchos::RCP<shards::CellTopology> cellType;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT> val_cell;

  // Output:
  //! Values on side
  PHX::MDField<ScalarT> val_side;

  enum LayoutType
  {
    CELL_SCALAR = 1,
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

  Kokkos::DynRankView<int, PHX::Device> dimsView;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct DOFCellToSide_CellScalar_Tag{};
  struct DOFCellToSide_CellVector_Tag{};
  struct DOFCellToSide_CellTensor_Tag{};
  struct DOFCellToSide_NodeScalar_Tag{};
  struct DOFCellToSide_NodeScalarSideset_Tag{};
  struct DOFCellToSide_NodeVector_Tag{};
  struct DOFCellToSide_NodeVectorSideset_Tag{};
  struct DOFCellToSide_NodeTensor_Tag{};
  struct DOFCellToSide_VertexVector_Tag{};
  struct DOFCellToSide_VertexVectorSideset_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_CellScalar_Tag> DOFCellToSide_CellScalar_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_CellVector_Tag> DOFCellToSide_CellVector_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_CellTensor_Tag> DOFCellToSide_CellTensor_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_NodeScalar_Tag> DOFCellToSide_NodeScalar_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_NodeScalarSideset_Tag> DOFCellToSide_NodeScalarSideset_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_NodeVector_Tag> DOFCellToSide_NodeVector_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_NodeVectorSideset_Tag> DOFCellToSide_NodeVectorSideset_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_NodeTensor_Tag> DOFCellToSide_NodeTensor_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_VertexVector_Tag> DOFCellToSide_VertexVector_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFCellToSide_VertexVectorSideset_Tag> DOFCellToSide_VertexVectorSideset_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_CellScalar_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_CellVector_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_CellTensor_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_NodeScalar_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_NodeScalarSideset_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_NodeVector_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_NodeVectorSideset_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_NodeTensor_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_VertexVector_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFCellToSide_VertexVectorSideset_Tag& tag, const int& sideSet_idx) const;

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
