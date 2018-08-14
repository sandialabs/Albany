//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_L2_PROJECTION_RESIDUAL_HPP
#define SURFACE_L2_PROJECTION_RESIDUAL_HPP

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Project a discrete scalar at integration point to
    an element-wise linear field.

**/

template <typename EvalT, typename Traits>
class SurfaceL2ProjectionResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                                    public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  SurfaceL2ProjectionResidual(
      const Teuchos::ParameterList&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Length scale parameter for localization zone
  RealType thickness;
  //! Numerical integration rule
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature;
  //! Finite element basis for the midplane
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  //! Scalar Gradient for H1 projection (not yet implemented)
  // PHX::MDField<ScalarT,Cell,QuadPoint,Dim> scalarGrad;
  //! Scalar Gradient Operator for H1 projection (not yet implemented)
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> surface_Grad_BF;
  //! Reference configuration dual basis
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim, Dim> refDualBasis;
  //! Reference configuration normal
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> refNormal;
  //! Reference configuration area
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> refArea;
  //! Cauchy Stress
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> Cauchy_stress_;
  //! Determinant of deformation gradient
  PHX::MDField<const ScalarT, Cell, QuadPoint> detF_;
  //! Porjected hydrostatic Kirchhoff stress
  PHX::MDField<const ScalarT, Cell, QuadPoint> projected_tau_;

  //! Reference Cell Views
  Kokkos::DynRankView<RealType, PHX::Device> refValues;
  Kokkos::DynRankView<RealType, PHX::Device> refGrads;
  Kokkos::DynRankView<RealType, PHX::Device> refPoints;
  Kokkos::DynRankView<RealType, PHX::Device> refWeights;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> projection_residual_;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numPlaneNodes;
  unsigned int numPlaneDims;
};
}  // namespace LCM

#endif
