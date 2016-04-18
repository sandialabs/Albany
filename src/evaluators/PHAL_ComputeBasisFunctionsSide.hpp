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

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

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
  //! Coordinate vector at side's vertices
  PHX::MDField<MeshScalarT,Cell,Side,Vertex,Dim> coordVec;

  // Temporary FieldContainers
  Intrepid2::FieldContainer_Kokkos<RealType,    PHX::Layout, PHX::Device> val_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType,    PHX::Layout, PHX::Device> grad_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType,    PHX::Layout, PHX::Device> cub_weights;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> tangents;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>  metric;

  // Output:
  //! Basis Functions and other quantities at quadrature points
  PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>           metric_det;
  PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>           w_measure;
  PHX::MDField<MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>   inv_metric;
  PHX::MDField<RealType,Cell,Side,Node,QuadPoint>         BF;
  PHX::MDField<MeshScalarT,Cell,Side,Node,QuadPoint,Dim>  GradBF;

  std::vector<std::vector<int> > sideNodes;
};

} // Namespace PHAL

#endif // PHAL_COMPUTE_BASIS_FUNCTIONS_SIDE_HPP
