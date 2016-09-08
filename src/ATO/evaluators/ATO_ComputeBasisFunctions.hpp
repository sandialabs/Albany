//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_COMPUTEBASISFUNCTIONS_HPP
#define PHAL_COMPUTEBASISFUNCTIONS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

#include <Cogent_Integrator.hpp>

namespace ATO {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
template<typename EvalT, typename Traits>
class ComputeBasisFunctions : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComputeBasisFunctions(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl,
                        const Albany::MeshSpecsStruct* meshSpecs);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  int numCells, numVertices, numDims, numNodes, numQPs, numTopos;

  std::string elementBlockName;
  Teuchos::Array<std::string> topoNames;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Cogent::Integrator> cubature;

  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> topoVals;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> coordVals;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> val_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> grad_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refPoints;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> weights;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobian;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobian_inv;

  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det; 
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
};
}

#endif
