//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_TL_PORO_MASS_RESIDUAL_HPP
#define SURFACE_TL_PORO_MASS_RESIDUAL_HPP

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Compute the balance of mass residual on the surface

**/

template <typename EvalT, typename Traits>
class SurfaceTLPoroMassResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  SurfaceTLPoroMassResidual(
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
  //! Scalar Gradient Operator
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> surface_Grad_BF;
  //! Scalar Jump
  PHX::MDField<const ScalarT, Cell, QuadPoint> scalarJump;
  //! Reference configuration dual basis
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim, Dim> refDualBasis;
  //! Reference configuration normal
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> refNormal;
  //! Reference configuration area
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> refArea;
  //! Determinant of the surface deformation gradient
  PHX::MDField<const ScalarT, Cell, QuadPoint> J;
  //! Pore Pressure at the 2D integration point location
  PHX::MDField<const ScalarT, Cell, QuadPoint> porePressure;
  //! Nodal Pore Pressure at the 2D integration point location
  PHX::MDField<const ScalarT, Cell, Node> nodalPorePressure;
  //! Biot Coefficient at the 2D integration point location
  PHX::MDField<const ScalarT, Cell, QuadPoint> biotCoefficient;
  //! Biot Modulus at the 2D integration point location
  PHX::MDField<const ScalarT, Cell, QuadPoint> biotModulus;
  //! Permeability at the 2D integration point location
  PHX::MDField<const ScalarT, Cell, QuadPoint> kcPermeability;
  //! Deformation Gradient
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> defGrad;
  //! Time
  PHX::MDField<const ScalarT, Dummy> deltaTime;

  //! Data from previous time step
  std::string porePressureName, JName;

  //! Reference Cell Views
  Kokkos::DynRankView<RealType, PHX::Device> refValues;
  Kokkos::DynRankView<RealType, PHX::Device> refGrads;
  Kokkos::DynRankView<RealType, PHX::Device> refPoints;
  Kokkos::DynRankView<RealType, PHX::Device> refWeights;

  // Work space FCs
  Kokkos::DynRankView<ScalarT, PHX::Device> F_inv;
  Kokkos::DynRankView<ScalarT, PHX::Device> F_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> C;
  Kokkos::DynRankView<ScalarT, PHX::Device> Cinv;
  Kokkos::DynRankView<ScalarT, PHX::Device> JF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> KJF_invT;
  Kokkos::DynRankView<ScalarT, PHX::Device> Kref;

  // Temporary Views
  Kokkos::DynRankView<ScalarT, PHX::Device> flux;

  // Output:
  PHX::MDField<ScalarT, Cell, Node> poroMassResidual;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numPlaneNodes;
  unsigned int numPlaneDims;

  bool haveMech;
};
}  // namespace LCM

#endif
