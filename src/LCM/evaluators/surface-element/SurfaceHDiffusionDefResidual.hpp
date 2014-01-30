//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef  SURFACE_H_DIFFUSION_DEF_RESIDUAL_HPP
#define SURFACE_H_DIFFUSION_DEF_RESIDUAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

    Compute the balance of mass residual on the surface

**/

template<typename EvalT, typename Traits>
class SurfaceHDiffusionDefResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

	SurfaceHDiffusionDefResidual(const Teuchos::ParameterList& p,
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
 //! Scalar Gradient Operator
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> surface_Grad_BF;
  //! Scalar Jump
   PHX::MDField<ScalarT,Cell,QuadPoint> scalarJump;
  //! Reference configuration dual basis
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim, Dim> refDualBasis;
  //! Reference configuration normal
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> refNormal;
  //! Reference configuration area
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> refArea;
  //! Determinant of the surface deformation gradient
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  //! Pore Pressure at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> transport_;
  //! Nodal Pore Pressure at the 2D integration point location
  PHX::MDField<ScalarT,Cell,Node> nodal_transport_;

  //! diffusion coefficient at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> dL_;
  //! effective diffusion constant at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> eff_diff_;
  //! strain rate factor at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> strain_rate_factor_;
  //! Convection-like term with hydrostatic stress at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> convection_coefficient_;
  //! Hydrostatic stress gradient at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint, Dim> hydro_stress_gradient_;
  //! Equvialent plastic strain at the 2D integration point location
  PHX::MDField<ScalarT,Cell,QuadPoint> eqps_;
  //! Elelement length parameter for stabilization procedure
  PHX::MDField<ScalarT,Cell,QuadPoint> element_length_;


  //! Deformation Gradient
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim> defGrad;

//  // weight times basis function value at integration point
//  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  //Data from previous time step
   std::string transportName, JName, CLGradName, eqpsName;

   // Time
   PHX::MDField<ScalarT,Dummy> deltaTime;

  //! Reference Cell FieldContainers
  Intrepid::FieldContainer<RealType> refValues;
  Intrepid::FieldContainer<RealType> refGrads;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;



  Intrepid::FieldContainer<ScalarT> artificalDL;
  Intrepid::FieldContainer<ScalarT> stabilizedDL;

  Intrepid::FieldContainer<ScalarT> pterm;

  // Temporary FieldContainers
  Intrepid::FieldContainer<ScalarT> flux;

  ScalarT trialPbar;

  // Stabilization Parameter
  RealType stab_param_;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> transport_residual_;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numPlaneNodes;
  unsigned int numPlaneDims;

  bool haveMech;
};
}

#endif
