//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESCONTRAVARIENTMETRICTENSOR_HPP
#define FELIX_STOKESCONTRAVARIENTMETRICTENSOR_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesContravarientMetricTensor : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  StokesContravarientMetricTensor(const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numDims, numQPs;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Temporary FieldContainers
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refPoints;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refWeights;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobian;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobian_inv;

  // Output:
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> Gc;
};
}

#endif
