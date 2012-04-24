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
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef Intrepid::FieldContainer<ScalarT> FC;

  Localization(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  void computeMidplaneCoords(PHX::MDField<ScalarT,Cell,Vertex,Dim> coordVec,
                             FC & midplaneCoords);
  void baseVectors(const FC & midplnaeCoords, FC & bases);
  void dualBaseVectors(const FC & midplaneCoords, FC & normals, FC & dualBases);
  void computeJacobian(const FC & bases, const FC & dualBases, FC & area, FC & jacobian);

private:

  int  numVertices, numDims, numNodes, numQPs, numPlaneNodes;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType> val_at_cub_points;
  Intrepid::FieldContainer<RealType> grad_at_cub_points;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;
  //Intrepid::FieldContainer<MeshScalarT> jacobian;
  Intrepid::FieldContainer<MeshScalarT> jacobian_inv;
  Intrepid::FieldContainer<MeshScalarT> jacobian_det;
  Intrepid::FieldContainer<MeshScalarT> weighted_measure;

  // new stuff
  Intrepid::FieldContainer<ScalarT> midplaneCoords;
  Intrepid::FieldContainer<ScalarT> bases;
  Intrepid::FieldContainer<ScalarT> dualBases;
  Intrepid::FieldContainer<ScalarT> jacobian;
  Intrepid::FieldContainer<ScalarT> normals;

  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
};
}

#endif
