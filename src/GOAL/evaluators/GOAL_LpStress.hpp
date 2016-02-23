//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_LPSTRESS_HPP
#define GOAL_LPSTRESS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace GOAL {

template<typename EvalT, typename Traits>
class LpStress :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

  public:

    LpStress(
        const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    int numDims;
    int numNodes;
    int numQPs;

    // input:
    int order;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weight;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress;

    // Output:
    PHX::MDField<ScalarT,Cell> lpStress;
};

}

#endif
