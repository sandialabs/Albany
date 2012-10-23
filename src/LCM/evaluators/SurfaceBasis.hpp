//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACE_BASIS_HPP
#define SURFACE_BASIS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace LCM {
  /** \brief Surface Basis Evaluator

      This evaluator computes bases for surface elements

  */

  template<typename EvalT, typename Traits>
  class SurfaceBasis : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef Intrepid::FieldContainer<ScalarT> FC;

    SurfaceBasis(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

    ///
    /// Takes the reference coordinates and computes the midplane
    /// \param refCoords
    /// \param midplaneCoords
    ///
    void computeReferenceMidplaneCoords(const PHX::MDField<MeshScalarT,Cell,Vertex,Dim> refCoords,
                                        FC & midplaneCoords);

    ///
    /// Takes the current coordinates and computes the midplane
    /// \param currentCoords
    /// \param midplaneCoords
    ///
    void computeCurrentMidplaneCoords(const PHX::MDField<ScalarT,Cell,Vertex,Dim> currentCoords,
                                      FC & midplaneCoords);

    ///
    /// Computes current configuration Bases from the midplane
    /// \param midplaneCoords
    /// \param basis
    ///
    void computeBaseVectors(const FC & midplaneCoords, 
                            PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> basis);

    ///
    /// Computes the Dual from the midplane and current bases
    /// \param midplaneCoords
    /// \param basis
    /// \param normals
    /// \param dualBasis
    ///
    void computeDualBaseVectors(const FC & midplaneCoords, 
                                const PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> basis, 
                                PHX::MDField<ScalarT,Cell,QuadPoint,Dim> normal, 
                                PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> dualBasis);

    ///
    /// Computes the jacobian mapping - da/dA
    /// \param basis
    /// \param dualBasis
    /// \param area
    ///
    void computeJacobian(const PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> basis,
                         const PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> dualBasis,
                         PHX::MDField<ScalarT,Cell,QuadPoint> area);

  private:
    unsigned int  numDims, numNodes, numQPs, numPlaneNodes, numPlaneDims;

    bool needCurrentBasis;

    // Input:
    //! Cordinates in the reference configuration
    PHX::MDField<MeshScalarT,Cell,Vertex,Dim> referenceCoords;
    //! Numerical integration rule
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
    //! Finite element basis for the midplane
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;

    // Surface Ref Bases FieldContainers
    Intrepid::FieldContainer<ScalarT> midplaneCoords;

    // Output:
    //! Reference basis
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> refBasis;
    //! Reference integration area
    PHX::MDField<ScalarT,Cell,QuadPoint> refArea;
    //! Reference dual basis
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> refDualBasis;
    //! Reference normal
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> refNormal;

    // if we need to compute the current bases (for mechanics)
    //! Coordinates in the current configuration
    PHX::MDField<ScalarT,Cell,Vertex,Dim> currentCoords;
    //! Current basis
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> currentBasis;

    //! Reference Cell FieldContainers
    Intrepid::FieldContainer<RealType> refValues;
    Intrepid::FieldContainer<RealType> refGrads;
    Intrepid::FieldContainer<RealType> refPoints;
    Intrepid::FieldContainer<RealType> refWeights;
  };
}

#endif
