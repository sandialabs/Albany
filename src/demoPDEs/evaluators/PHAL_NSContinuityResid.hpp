//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_NSCONTINUITYRESID_HPP
#define PHAL_NSCONTINUITYRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class NSContinuityResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  NSContinuityResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim,Dim> VGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint> rho;
  
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint> TauM;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> Rm;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> CResidual;

  unsigned int numCells, numQPs, numDims, numNodes;
  Kokkos::DynRankView<ScalarT, PHX::Device> divergence;
  bool havePSPG;
};
}

#endif
