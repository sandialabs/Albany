//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_GEOPOTENTIAL_HPP
#define AERAS_XZHYDROSTATIC_GEOPOTENTIAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Geopotential (phi) for XZHydrostatic atmospheric model

    This evaluator computes the Geopotential for the XZHydrostatic model
    of atmospheric dynamics.

*/
template<typename EvalT, typename Traits>
class XZHydrostatic_GeoPotential : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_GeoPotential(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input
                     
  //why not Eta??? it is in XZHydrostaticProblem
                     
  PHX::MDField<ScalarT,Cell,Node,Level> density;
  PHX::MDField<ScalarT,Cell,Node,Level> Pi;
                     
  PHX::MDField<ScalarT,Cell,Node> PhiSurf;
                     
  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level> Phi;

  const int numNodes;
  const int numLevels;

  ScalarT Phi0;
};
}

#endif
