//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_COMPUTE_BASIS_FUNCTIONS_SIDE_HPP
#define PHAL_COMPUTE_BASIS_FUNCTIONS_SIDE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
template<typename EvalT, typename Traits>
class ComputeBasisFunctionsSide : public PHX::EvaluatorWithBaseImpl<Traits>,
       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComputeBasisFunctionsSide(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  int numSides, numSideNodes, numSideQPs, cellDims, sideDims;

  //! The side set where to compute the Basis Functions
  std::string sideSetName;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType>    val_at_cub_points;
  Intrepid::FieldContainer<RealType>    grad_at_cub_points;
  Intrepid::FieldContainer<RealType>    cub_weights;
  Intrepid::FieldContainer<MeshScalarT> tangents;
  Intrepid::FieldContainer<MeshScalarT> metric;

  // Output:
  //! Basis Functions and other quantities at quadrature points
  PHX::MDField<RealType,Cell,Side,QuadPoint>              metric_det;
  PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>           w_measure;
  PHX::MDField<RealType,Cell,Side,Node,QuadPoint>         BF;
  PHX::MDField<MeshScalarT,Cell,Side,Node,QuadPoint,Dim>  GradBF;
  PHX::MDField<MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>   inv_metric;

  std::vector<std::vector<int> > sideNodes;
};

} // Namespace PHAL

#endif // PHAL_COMPUTE_BASIS_FUNCTIONS_SIDE_HPP
