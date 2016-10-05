//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_CONTRAVARIANTTARGETMETRICTENSOR_HPP
#define PHAL_CONTRAVARIANTTARGETMETRICTENSOR_HPP

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
class ContravariantTargetMetricTensor : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ContravariantTargetMetricTensor(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  int  numDims, numQPs, containerSize;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<ScalarT, Cell, Node, Dim> solnVec;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubature;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;

  // Temporary FieldContainers
  Kokkos::DynRankView<RealType, PHX::Device> refPoints;
  Kokkos::DynRankView<RealType, PHX::Device> refWeights;
  Kokkos::DynRankView<ScalarT, PHX::Device> jacobian;
  Kokkos::DynRankView<ScalarT, PHX::Device> jacobian_inv;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> Gc;

};
}

#endif
