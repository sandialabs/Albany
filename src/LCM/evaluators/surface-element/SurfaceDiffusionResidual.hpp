//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_DIFFUSION_RESIDUAL_HPP
#define SURFACE_DIFFUSION_RESIDUAL_HPP

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Compute the residual forces on a surface

**/

template <typename EvalT, typename Traits>
class SurfaceDiffusionResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  SurfaceDiffusionResidual(
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
  ScalarT thickness;
  //! Numerical integration rule
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature;
  //! Finite element basis for the midplane
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  //! Scalar Gradient
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> scalarGrad;
  //! Scalar Jump
  PHX::MDField<const ScalarT, Cell, QuadPoint> scalarJump;
  //! Current configuration basis
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> currentBasis;
  //! Reference configuration dual basis
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> refDualBasis;
  //! Reference configuration normal
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> refNormal;
  //! Reference configuration area
  PHX::MDField<const ScalarT, Cell, QuadPoint> refArea;

  //! Reference Cell Views
  Kokkos::DynRankView<RealType, PHX::Device> refValues;
  Kokkos::DynRankView<RealType, PHX::Device> refGrads;
  Kokkos::DynRankView<RealType, PHX::Device> refPoints;
  Kokkos::DynRankView<RealType, PHX::Device> refWeights;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> scalarResidual;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numPlaneNodes;
  unsigned int numPlaneDims;
};
}  // namespace LCM

#endif
