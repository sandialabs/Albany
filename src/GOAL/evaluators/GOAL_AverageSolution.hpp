//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_AVERAGESOLUTION_HPP
#define GOAL_AVERAGESOLUTION_HPP

#include <Phalanx_config.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "Albany_Layouts.hpp"
#include "AAdapt_RC_Field.hpp"

namespace GOAL {

template<typename EvalT, typename Traits>
class AverageSolution:
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  AverageSolution(
      Teuchos::ParameterList& p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  /* input */
  PHX::MDField<MeshScalarT, Cell, Node, QuadPoint> wBF;

  /* output */
  PHX::MDField<ScalarT, Cell, Node, Dim> avgSolution;

  int numNodes;
  int numQPs;
  int numDims;

};

}

#endif
