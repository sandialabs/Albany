//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_NSBODYFORCE_HPP
#define PHAL_NSBODYFORCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class NSBodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  NSBodyForce(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:  
  PHX::MDField<ScalarT,Cell,QuadPoint> T;
  PHX::MDField<ScalarT,Cell,QuadPoint> rho;
  PHX::MDField<ScalarT,Cell,QuadPoint> beta;
  Teuchos::Array<double> gravity;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> force;

   //Radom field types
  enum BFTYPE {NONE, CONSTANT, BOUSSINESQ};
  BFTYPE bf_type;

  unsigned int numQPs, numDims;

  bool haveHeat;
 
};
}

#endif
