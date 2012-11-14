//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOONEYRIVLIN_HPP
#define MOONEYRIVLIN_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief compressible Mooney-Rivlin stress response

    This evaluator computes stress based on a coupled Mooney-Rivlin
    Helmholtz potential

*/

template<typename EvalT, typename Traits>
class MooneyRivlin : public PHX::EvaluatorWithBaseImpl<Traits>,
		   public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  MooneyRivlin(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  RealType c1;
  RealType c2;
  RealType c;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // scratch space FCs
  Intrepid::FieldContainer<ScalarT> FT;
};
}

#endif





