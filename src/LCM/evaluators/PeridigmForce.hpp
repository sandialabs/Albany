//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PERIDIGMFORCE_HPP
#define PERIDIGMFORCE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#ifdef ALBANY_PERIDIGM
#include <Peridigm.hpp>
#include <Peridigm_AlbanyDiscretization.hpp>
#endif

namespace LCM {
/** \brief Evaluates nodal forces through a code coupling with the Peridigm peridynamics code.
*/

template<typename EvalT, typename Traits>
class PeridigmForce : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  PeridigmForce(Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dataLayout);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  RealType density;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> volume;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> referenceCoordinates;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> currentCoordinates;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> force;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> residual;

  unsigned int numQPs;
  unsigned int numDims;
};
}

#endif
