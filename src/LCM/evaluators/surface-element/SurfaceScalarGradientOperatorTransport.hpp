//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_SCALAR_GRADIENT_OPERATOR_TRANSPORT_HPP
#define SURFACE_SCALAR_GRADIENT_OPERATOR_TRANSPORT_HPP

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Construct a scalar gradient operator for the surface element.

**/

template <typename EvalT, typename Traits>
class SurfaceScalarGradientOperatorTransport
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  SurfaceScalarGradientOperatorTransport(
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
  /// Length scale parameter for localization zone
  RealType thickness;

  /// Numerical integration rule
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature;

  /// for the parallel gradient term
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  // nodal value used to construct in-plan gradient
  PHX::MDField<const ScalarT, Cell, Node> val_node;

  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim, Dim> refDualBasis;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim>      refNormal;

  //! Reference Cell Views
  Kokkos::DynRankView<RealType, PHX::Device> refValues;
  Kokkos::DynRankView<RealType, PHX::Device> refGrads;
  Kokkos::DynRankView<RealType, PHX::Device> refPoints;
  Kokkos::DynRankView<RealType, PHX::Device> refWeights;

  // Output:
  PHX::MDField<MeshScalarT, Cell, Node, QuadPoint, Dim> surface_Grad_BF;
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim>           grad_val_qp;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numPlaneNodes;
  unsigned int numPlaneDims;
};
}  // namespace LCM

#endif
