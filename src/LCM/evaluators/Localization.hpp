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
  ///
  void computeDeformationGradient(const FC & bases);

private:

  int  numDims, numNodes, numQPs, numPlaneNodes;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<ScalarT,Cell,Vertex,Dim> referenceCoords;
  PHX::MDField<ScalarT,Cell,Vertex,Dim> currentCoords;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Reference Cell FieldContainers
  Intrepid::FieldContainer<RealType> refValues;
  Intrepid::FieldContainer<RealType> refGrads;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;

  // Localization Element FieldContainers
  Intrepid::FieldContainer<ScalarT> midplaneCoords;
  Intrepid::FieldContainer<ScalarT> bases;
  Intrepid::FieldContainer<ScalarT> dualBases;
  Intrepid::FieldContainer<ScalarT> jacobian;
  Intrepid::FieldContainer<ScalarT> normals;
  Intrepid::FieldContainer<ScalarT> area;
  Intrepid::FieldContainer<ScalarT> gap;

  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad;
};
}

#endif
