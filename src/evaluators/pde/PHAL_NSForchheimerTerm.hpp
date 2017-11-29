//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_NSFORCHHEIMERTERM_HPP
#define PHAL_NSFORCHHEIMERTERM_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
namespace PHAL {

template<typename EvalT, typename Traits>
class NSForchheimerTerm : public PHX::EvaluatorWithBaseImpl<Traits>,
	     public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  NSForchheimerTerm(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> V;
  PHX::MDField<const ScalarT,Cell,QuadPoint> rho;
  PHX::MDField<const ScalarT,Cell,QuadPoint> phi;
  PHX::MDField<const ScalarT,Cell,QuadPoint> K;
  PHX::MDField<const ScalarT,Cell,QuadPoint> F;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> ForchTerm;

  unsigned int numQPs, numDims, numNodes, numCells;
  bool enableTransient;
  bool haveHeat;

  Kokkos::DynRankView<ScalarT, PHX::Device> normV;
 
};
}

#endif
