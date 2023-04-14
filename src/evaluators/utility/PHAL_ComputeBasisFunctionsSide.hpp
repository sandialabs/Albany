//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_COMPUTE_BASIS_FUNCTIONS_SIDE_HPP
#define PHAL_COMPUTE_BASIS_FUNCTIONS_SIDE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
template<typename EvalT, typename Traits>
class ComputeBasisFunctionsSide : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  ComputeBasisFunctionsSide(const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  unsigned int numSides, numSideNodes, numSideQPs, numCellDims, numSideDims, numNodes, effectiveCoordDim;
  MDFieldMemoizer<Traits> memoizer;

  //! The side set where to compute the Basis Functions
  std::string sideSetName;

  // Input:
  //! Coordinate vector at side's vertices
  PHX::MDField<const MeshScalarT>                 sideCoordVec;
  PHX::MDField<const MeshScalarT,Cell,Vertex,Dim> coordVec;

  // Temporary Kokkos Views
  Kokkos::DynRankView<RealType, PHX::Device> val_at_cub_points;
  Kokkos::DynRankView<RealType, PHX::Device> grad_at_cub_points;
  Kokkos::DynRankView<RealType, PHX::Device> cub_weights;
  Kokkos::DynRankView<RealType, PHX::Device> cub_points;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> normals_view;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubature;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;

  // Output:
  //! Basis Functions and other quantities at quadrature points
  PHX::MDField<MeshScalarT>                               metric_det;
  PHX::MDField<MeshScalarT>                               tangents;
  PHX::MDField<MeshScalarT>                               metric;
  PHX::MDField<MeshScalarT>                               w_measure;
  PHX::MDField<MeshScalarT>                               inv_metric;
  PHX::MDField<MeshScalarT>                               BF;
  PHX::MDField<MeshScalarT>                               GradBF;
  PHX::MDField<MeshScalarT,Side,QuadPoint,Dim>            normals;

  int currentSide;

  Teuchos::RCP<shards::CellTopology> cellType;
  bool compute_normals;

  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct ComputeBasisFunctionsSide_Tag{};
  struct ScatterCoordVec_Tag{};
  struct GatherNormals_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, ComputeBasisFunctionsSide_Tag> ComputeBasisFunctionsSide_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterCoordVec_Tag> ScatterCoordVec_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, GatherNormals_Tag> GatherNormals_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ComputeBasisFunctionsSide_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterCoordVec_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const GatherNormals_Tag& tag, const int& sideSet_idx) const;

};

} // Namespace PHAL

#endif // PHAL_COMPUTE_BASIS_FUNCTIONS_SIDE_HPP
