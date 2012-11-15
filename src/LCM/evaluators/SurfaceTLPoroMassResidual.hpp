//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef SURFACE_TL_PORO_MASS_RESIDUAL_HPP
#define SURFACE_TL_PORO_MASS_RESIDUAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Compute the residual forces on a surface

**/

template<typename EvalT, typename Traits>
class SurfaceTLPoroMassResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  SurfaceTLPoroMassResidual(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Length scale parameter for localization zone
  ScalarT thickness;
  //! Numerical integration rule
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  //! Finite element basis for the midplane
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  //! Scalar Gradient
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> scalarGrad;
  //! Scalar Jump
   PHX::MDField<ScalarT,Cell,QuadPoint> scalarJump;
  //! Current configuration basis
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim> currentBasis;
  //! Reference configuration dual basis
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim> refDualBasis;
  //! Reference configuration normal
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> refNormal;
  //! Reference configuration area
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> refArea;
  //! Determinant of the surface deformation gradient
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  //! Pore Pressure at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> porePressure;
  //! Biot Coefficient at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> biotCoefficient;
  //! Biot Modulus at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> biotModulus;

//  // weight times basis function value at integration point
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  //Data from previous time step
   std::string porePressureName, JName;

  //! Reference Cell FieldContainers
  Intrepid::FieldContainer<RealType> refValues;
  Intrepid::FieldContainer<RealType> refGrads;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> poroMassResidual;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numPlaneNodes;
  unsigned int numPlaneDims;
};
}

#endif
