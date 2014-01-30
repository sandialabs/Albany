//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef  SURFACE_L2_PROJECTION_RESIDUAL_HPP
#define  SURFACE_L2_PROJECTION_RESIDUAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Project a discrete scalar at integration point to
    an element-wise linear field.

**/

template<typename EvalT, typename Traits>
class SurfaceL2ProjectionResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

	SurfaceL2ProjectionResidual(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;


  // Input:
  //! Length scale parameter for localization zone
  RealType thickness;
  //! Numerical integration rule
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  //! Finite element basis for the midplane
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  //! Scalar Gradient for H1 projection (not yet implemented)
  //PHX::MDField<ScalarT,Cell,QuadPoint,Dim> scalarGrad;
 //! Scalar Gradient Operator for H1 projection (not yet implemented)
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> surface_Grad_BF;
  //! Reference configuration dual basis
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim, Dim> refDualBasis;
  //! Reference configuration normal
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> refNormal;
  //! Reference configuration area
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> refArea;
  //! Cauchy Stress
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim> Cauchy_stress_;
  //! Determinant of deformation gradient
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> detF_;
  //! Porjected hydrostatic Kirchhoff stress
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> projected_tau_;

//  // weight times basis function value at integration point
//  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  //! Reference Cell FieldContainers
  Intrepid::FieldContainer<RealType> refValues;
  Intrepid::FieldContainer<RealType> refGrads;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;


  // Output:
  PHX::MDField<ScalarT,Cell,Node> projection_residual_;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numPlaneNodes;
  unsigned int numPlaneDims;

};
}

#endif
