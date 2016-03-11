//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_EPSILONL1L2_HPP
#define FELIX_EPSILONL1L2_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class EpsilonL1L2 : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>,
		    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  EpsilonL1L2(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n); 

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarT homotopyParam;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> epsilonXX;
  PHX::MDField<ScalarT,Cell,QuadPoint> epsilonYY;
  PHX::MDField<ScalarT,Cell,QuadPoint> epsilonXY;
  PHX::MDField<ScalarT,Cell,QuadPoint> epsilonB;

  unsigned int numQPs, numDims, numNodes;
  
};
}

#endif
