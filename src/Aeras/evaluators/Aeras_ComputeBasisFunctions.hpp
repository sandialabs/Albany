//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_COMPUTEBASISFUNCTIONS_HPP
#define AERAS_COMPUTEBASISFUNCTIONS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"
#include "Aeras_EvaluatorUtilities.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace Albany { class StateManager; }

namespace Aeras {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ComputeBasisFunctions : public PHX::EvaluatorWithBaseImpl<Traits>,
 			 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  const int spatialDimension;
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numVertices, numDims, numNodes, numQPs;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > cubature;
  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Temporary FieldContainers
  //PHX::MDField<RealType,Node,QuadPoint>    val_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    val_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    grad_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    D2_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    refPoints;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    refWeights;

  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>   sphere_coord; 
  PHX::MDField<MeshScalarT,Cell,Node> lambda_nodal;
  PHX::MDField<MeshScalarT,Cell,Node> theta_nodal;
  PHX::MDField<MeshScalarT,Cell,QuadPoint>     jacobian_det; 
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian_inv;
  PHX::MDField<MeshScalarT,Cell,Node,Dim,Dim> jacobian_inv_node;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim,Dim> GradGradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim,Dim> wGradGradBF;
         
  const double earthRadius;
  void div_check(const int spatialDim, const int numelements) const;
  void spherical_divergence(Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> &,
                            const Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> &,
                            const int e,
                            const double rrearth=1) const;
  void initialize_grad(Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> &) const;

  MDFieldMemoizer<Traits> memoizer_;

  // Kokkos
/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  Kokkos::View<RealType*, PHX::Device> refWeights_CUDA;
  Kokkos::View<RealType**, PHX::Device> val_at_cub_points_CUDA;
  Kokkos::View<RealType***, PHX::Device> grad_at_cub_points_CUDA;

  Kokkos::View<MeshScalarT**,  PHX::Device>  phi;
  Kokkos::View<MeshScalarT***, PHX::Device>  dphi;
  Kokkos::View<MeshScalarT*,   PHX::Device>  norm;
  Kokkos::View<MeshScalarT*,   PHX::Device>  sinL;
  Kokkos::View<MeshScalarT*,   PHX::Device>  cosL;
  Kokkos::View<MeshScalarT*,   PHX::Device>  sinT;
  Kokkos::View<MeshScalarT*,   PHX::Device>  cosT;
  Kokkos::View<MeshScalarT***, PHX::Device>  D1;
  Kokkos::View<MeshScalarT***, PHX::Device>  D2;
  Kokkos::View<MeshScalarT***, PHX::Device>  D3;


  double pi;
  double DIST_THRESHOLD;

  int numelements;
  int spatialDim;
  int basisDim;

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct ComputeBasisFunctions_Tag{};
  struct ComputeBasisFunctions_basisDim_Tag{};
  struct ComputeBasisFunctions_no_Jacobian_Tag{};
  struct ComputeBasisFunctions_no_Jacobian_basisDim_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, ComputeBasisFunctions_Tag> ComputeBasisFunctions_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ComputeBasisFunctions_basisDim_Tag> ComputeBasisFunctions_basisDim_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ComputeBasisFunctions_no_Jacobian_Tag> ComputeBasisFunctions_no_Jacobian_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ComputeBasisFunctions_no_Jacobian_basisDim_Tag> ComputeBasisFunctions_no_Jacobian_basisDim_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ComputeBasisFunctions_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ComputeBasisFunctions_basisDim_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ComputeBasisFunctions_no_Jacobian_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ComputeBasisFunctions_no_Jacobian_basisDim_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void compute_jacobian (const int cell) const;

  KOKKOS_INLINE_FUNCTION
  void compute_jacobian_inv (const int cell) const;

  KOKKOS_INLINE_FUNCTION
  void compute_jacobian_det (const int cell) const;

  KOKKOS_INLINE_FUNCTION
  void computeCellMeasure (const int cell) const;

  KOKKOS_INLINE_FUNCTION
  void compute_BF (const int cell) const;

  KOKKOS_INLINE_FUNCTION
  void compute_wBF (const int cell) const;  

  KOKKOS_INLINE_FUNCTION
  void compute_GradBF (const int cell) const;

  KOKKOS_INLINE_FUNCTION
  void compute_wGradBF (const int cell) const;


#endif*/
};
}

#endif
