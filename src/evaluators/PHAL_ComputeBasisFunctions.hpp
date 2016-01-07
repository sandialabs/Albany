//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_COMPUTEBASISFUNCTIONS_HPP
#define PHAL_COMPUTEBASISFUNCTIONS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
class ComputeBasisFunctions : public PHX::EvaluatorWithBaseImpl<Traits>,
 			 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numVertices, numDims, numNodes, numQPs;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<shards::CellTopology> cellType;
#if defined ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > cubature;
  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> > > intrepidBasis;

  // Temporary FieldContainers
  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> val_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> grad_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> refPoints;
  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> refWeights;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT,PHX::Layout,PHX::Device> jacobian;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT,PHX::Layout,PHX::Device> jacobian_inv;
#else
  Teuchos::RCP<Intrepid2::Cubature<RealType> > cubature;
  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer<RealType> > > intrepidBasis;

  Intrepid2::FieldContainer<RealType> val_at_cub_points;
  Intrepid2::FieldContainer<RealType> grad_at_cub_points;
  Intrepid2::FieldContainer<RealType> refPoints;
  Intrepid2::FieldContainer<RealType> refWeights;
  Intrepid2::FieldContainer<MeshScalarT> jacobian;
  Intrepid2::FieldContainer<MeshScalarT> jacobian_inv;
#endif

  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det; 
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
};
/*#else // ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
class ComputeBasisFunctions : public PHX::EvaluatorWithBaseImpl<Traits>,
 			 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  typedef typename PHX::Device execution_space;
  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numVertices, numDims, numNodes, numQPs;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > cubature;
  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> > > intrepidBasis;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Temporary FieldContainers
  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> val_at_cub_points;
  Kokkos::View <RealType**, PHX::Device> val_at_cub_points_CUDA;
  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> grad_at_cub_points;
  Kokkos::View <RealType***, PHX::Device> grad_at_cub_points_CUDA;

  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> refPoints;
  Kokkos::View <RealType**, PHX::Device> refPoints_CUDA; 
  Intrepid2::FieldContainer_Kokkos<RealType,PHX::Layout,PHX::Device> refWeights;
  Kokkos::View <RealType*, PHX::Device> refWeights_CUDA;
//  Intrepid2::FieldContainer_Kokkos<MeshScalarT,PHX::Layout,PHX::Device> jacobian;
  PHX::MDField <MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian; 
  //Intrepid2::FieldContainer_Kokkos<MeshScalarT,PHX::Layout,PHX::Device> jacobian_inv;
  PHX::MDField <MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian_inv;
  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det; 
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
};
#endif // ALBANY_KOKKOS_UNDER_DEVELOPMENT
*/
}

#endif
