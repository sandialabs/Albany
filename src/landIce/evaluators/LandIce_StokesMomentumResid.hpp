//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKESMOMENTUMRESID_HPP
#define LANDICE_STOKESMOMENTUMRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesMomentumResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  StokesMomentumResid(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim,Dim> VGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> V;
  PHX::MDField<const ScalarT,Cell,QuadPoint> P;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> force;
  PHX::MDField<const ScalarT,Cell,QuadPoint> TauM;
  PHX::MDField<const ScalarT,Cell,QuadPoint> mu;
  PHX::MDField<const ScalarT,Cell,QuadPoint> muLandIce;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> MResidual;

  unsigned int numQPs, numDims, numNodes;
 
};
}

#endif
