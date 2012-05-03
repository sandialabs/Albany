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


#ifndef LOCALIZATION_HPP
#define LOCALIZATION_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace LCM {
/** \brief Localization Element Evaluator

    This evaluator computes relevant things for the Localization element.

*/

template<typename EvalT, typename Traits>
class Localization : public PHX::EvaluatorWithBaseImpl<Traits>,
		     public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef Intrepid::FieldContainer<ScalarT> FC;

  Localization(const Teuchos::ParameterList& p);

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
  /// \param bases
  ///
  void computeBaseVectors(const FC & midplaneCoords, FC & bases);

  ///
  /// Computes the Dual from the midplane and current bases
  /// \param midplaneCoords
  /// \param bases
  /// \param normals
  /// \param dualBases
  ///
  void computeDualBaseVectors(const FC & midplaneCoords, const FC & bases, FC & normals, FC & dualBases);

  ///
  /// Computes the jacobian mapping - da/dA
  /// \param bases
  /// \param dualBases
  /// \param area
  /// \param jacobian
  ///
  void computeJacobian(const FC & bases, const FC & dualBases, FC & area, FC & jacobian);

  ///
  /// Computes the gap or jump
  /// \param coords
  /// \param gap
  ///
  void computeGap(const PHX::MDField<ScalarT,Cell,Vertex,Dim> coords, 
                  FC & gap);

  ///
  /// Computes the deformation gradient
  /// \param thickness h parameter
  /// \param bases
  /// \param dualBases
  /// \param refNormal
  /// \param gap
  /// \param defGrad deformation gradient
  /// \param J determinant of the deformation gradient
  ///
  void computeDeformationGradient(const ScalarT thickness, const FC & bases, const FC & dualBases, const FC & refNormal, const FC & gap,
                                  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad, FC & J);

  ///
  /// Computes the Cauchy stress
  /// \param defGrad - deformation gradient
  /// \param J - determinant of the deformation gradient
  /// \param mu - shear modulus
  /// \param kappa - bulk modulus
  /// \param stress
  ///
  void computeStress(const PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad, const FC & J, const PHX::MDField<ScalarT,Cell,QuadPoint> mu,
                     const PHX::MDField<ScalarT,Cell,QuadPoint> kappa, PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress);

private:

  int  numDims, numNodes, numQPs, numPlaneNodes, numPlaneDims;

  // Input:
  //! Cordinates in the reference configuration
  PHX::MDField<ScalarT,Cell,Vertex,Dim> referenceCoords;
  //! Coordinates in the current configuration
  PHX::MDField<ScalarT,Cell,Vertex,Dim> currentCoords;
  //! Numerical integration rule
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  //! Finite element basis for the midplane
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  //! Finite element basis for the midplane
  Teuchos::RCP<shards::CellTopology> cellType;
  //! Length scale parameter for localization zone
  ScalarT thickness;
  //! Shear Modulus
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;
  //! Bulk Modulus
  PHX::MDField<ScalarT,Cell,QuadPoint> kappa;

  // Reference Cell FieldContainers
  Intrepid::FieldContainer<RealType> refValues;
  Intrepid::FieldContainer<RealType> refGrads;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;

  // Localization Element FieldContainers
  Intrepid::FieldContainer<ScalarT> midplaneCoords;
  Intrepid::FieldContainer<ScalarT> bases;
  Intrepid::FieldContainer<ScalarT> dualBases;
  Intrepid::FieldContainer<ScalarT> refJacobian;
  Intrepid::FieldContainer<ScalarT> refNormal;
  Intrepid::FieldContainer<ScalarT> refArea;
  Intrepid::FieldContainer<ScalarT> gap;
  Intrepid::FieldContainer<ScalarT> J;

  // Output:
  //! the 3D deformation gradient at integration points
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad;
  //! the 3D Cauchy stress
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;
};
}

#endif
