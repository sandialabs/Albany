//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_SCATTERQOI_HPP
#define GOAL_SCATTERQOI_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace GOAL {

/****************************************************************************/
template<typename EvalT, typename Traits>
class ScatterQoI :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{
  public:

    ScatterQoI(
        const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);
};


/****************************************************************************
  Jacobian Specialization
 ***************************************************************************/
template<typename Traits>
class ScatterQoI<PHAL::AlbanyTraits::Jacobian, Traits> :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Jacobian, Traits>
{
  public:

    ScatterQoI(
        const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

    int numNodes;

    // Input:
    PHX::MDField<ScalarT, Cell> qoi;

    // Output:
    Teuchos::RCP<PHX::FieldTag> operation;

};

}

#endif
