//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_NODES_TO_CELL_INTERPOLATION_HPP
#define PHAL_NODES_TO_CELL_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_SacadoTypes.hpp"

#include "Intrepid2_CellTools.hpp"

namespace PHAL
{
/** \brief Average from Qp to Cell

    This evaluator averages the quadrature points values to
    obtain a single value for the whole cell

*/

template<typename EvalT, typename Traits, typename ScalarT>
class NodesToCellInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  NodesToCellInterpolationBase (const Teuchos::ParameterList& p,
                                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;

  int numNodes;
  int numQPs;
  int vecDim;

  bool isVectorField;

  enum {ValueAtCellBarycenter, CellAverage} interpolationType;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;
  Kokkos::DynRankView<RealType, PHX::Device>                       basis_at_barycenter;

  MDFieldMemoizer<Traits> memoizer;

  // Input:
  PHX::MDField<const ScalarT>                       field_node;
  PHX::MDField<const RealType,Cell,Node,QuadPoint>  BF;
  PHX::MDField<const MeshScalarT,Cell,Node>         w_measure;

  // Output:
  PHX::MDField<OutputScalarT>                       field_cell;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct Cell_Average_Vector_Field_Tag{};
  struct Cell_Average_Scalar_Field_Tag{};
  struct Cell_Barycenter_Vector_Field_Tag{};
  struct Cell_Barycenter_Scalar_Field_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Average_Vector_Field_Tag> Cell_Average_Vector_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Average_Scalar_Field_Tag> Cell_Average_Scalar_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Barycenter_Vector_Field_Tag> Cell_Barycenter_Vector_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Barycenter_Scalar_Field_Tag> Cell_Barycenter_Scalar_Field_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Average_Vector_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Average_Scalar_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Barycenter_Vector_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Barycenter_Scalar_Field_Tag& tag, const int& cell) const;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using NodesToCellInterpolation = NodesToCellInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using NodesToCellInterpolationMesh = NodesToCellInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using NodesToCellInterpolationParam = NodesToCellInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_NODES_TO_CELL_INTERPOLATION_HPP
