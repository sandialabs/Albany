//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_COMPUTEBASISFUNCTIONS_HPP
#define AERAS_COMPUTEBASISFUNCTIONS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace Aeras {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ComputeBasisFunctions : public PHX::EvaluatorWithBaseImpl<Traits>,
 			 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  const int spatialDimension;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numVertices, numDims, numNodes, numQPs;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType>    val_at_cub_points;
  Intrepid::FieldContainer<RealType>    grad_at_cub_points;
  Intrepid::FieldContainer<RealType>    refPoints;
  Intrepid::FieldContainer<RealType>    refWeights;

  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>   sphere_coord; 
  PHX::MDField<MeshScalarT,Cell,QuadPoint>     jacobian_det; 
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian_inv;
  PHX::MDField<MeshScalarT,Cell,Node,Dim,Dim> jacobian_inv_node;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  const double earthRadius;
  void div_check(const int spatialDim, const int numelements) const;
  void spherical_divergence(Intrepid::FieldContainer<MeshScalarT> &,
                            const Intrepid::FieldContainer<MeshScalarT> &,
                            const int e,
                            const double rrearth=1) const;
  void initialize_grad(Intrepid::FieldContainer<RealType> &) const;

};
}

#endif
