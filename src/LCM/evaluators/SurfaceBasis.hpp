/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef SURFACEBASIS_HPP
#define SURFACEBASIS_HPP

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
  typedef Intrepid::FieldContainer<ScalarT> FC;

  SurfaceBasis(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ///
  /// Takes the current coordinates and computes the midplane
  /// \param currentCoords
  /// \param midplaneCoords
  ///
  void computeMidplaneCoords(const PHX::MDField<ScalarT,Cell,Vertex,Dim> coords,
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

  int  numDims, numNodes, numQPs, numPlaneNodes, numPlaneDims;

  bool needCurrentBasis;

  // Input:
  //! Cordinates in the reference configuration
  PHX::MDField<ScalarT,Cell,Vertex,Dim> referenceCoords;
  //! Numerical integration rule
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  //! Finite element basis for the midplane
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  //! Finite element basis for the midplane
  Teuchos::RCP<shards::CellTopology> cellType;

  // Reference Cell FieldContainers
  Intrepid::FieldContainer<RealType> refValues;
  Intrepid::FieldContainer<RealType> refGrads;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;

  // Surface Ref Bases FieldContainers
  Intrepid::FieldContainer<ScalarT> midplaneCoords;
  //
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> basis;

  // Output:
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

};
}

#endif
