//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QPT_LOCATION_HPP
#define QPT_LOCATION_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief

    Obtain position vector for the integration points.


**/

template<typename EvalT, typename Traits>
class QptLocation : public PHX::EvaluatorWithBaseImpl<Traits>,
	       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  QptLocation(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint, Dim> GradBF;


  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gptLocation;

  unsigned int worksetSize;
  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int numDims;
};
}

#endif
