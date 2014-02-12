//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_VECTOR_RESIDUAL_HPP
#define SURFACE_VECTOR_RESIDUAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

#include "Albany_Layouts.hpp"

namespace LCM
{
  /** \brief

   Compute the residual forces on a surface

   **/

  template<typename EvalT, typename Traits>
  class SurfaceVectorResidual: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
  {

  public:

    SurfaceVectorResidual(Teuchos::ParameterList& p,
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
    //! First PK Stress
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    //! Current configuration basis
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> currentBasis;
    //! Reference configuration dual basis
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> refDualBasis;
    //! Reference configuration normal
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> refNormal;
    //! Reference configuration area
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> refArea;

    //! Reference Cell FieldContainers
    Intrepid::FieldContainer<RealType> refValues;
    Intrepid::FieldContainer<RealType> refGrads;
    Intrepid::FieldContainer<RealType> refPoints;
    Intrepid::FieldContainer<RealType> refWeights;

    ///
    /// Optional Cohesive Traction
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim> traction_;
    
    // Output:
    PHX::MDField<ScalarT, Cell, Node, Dim> force;

    unsigned int worksetSize;
    unsigned int numNodes;
    unsigned int numQPs;
    unsigned int numDims;
    unsigned int numPlaneNodes;
    unsigned int numPlaneDims;

    ///
    /// Cohesive Flag
    ///
    bool use_cohesive_traction_;

    ///
    /// Membrane Forces Flag
    ///
    bool compute_membrane_forces_;
  };
}

#endif
