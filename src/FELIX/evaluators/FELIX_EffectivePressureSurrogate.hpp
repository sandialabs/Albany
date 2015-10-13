//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_EFFECTIVE_PRESSURE_SURROGATE_HPP
#define FELIX_EFFECTIVE_PRESSURE_SURROGATE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits>
class EffectivePressureSurrogate : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  EffectivePressureSurrogate (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<ScalarT,Cell,Side,QuadPoint> H;
  PHX::MDField<ScalarT,Cell,Side,QuadPoint> z_s;

  // Output:
  PHX::MDField<ScalarT,Cell,Side,QuadPoint> N;

  std::string basalSideName;

  int numSideQPs;

  double alpha;
  double rho_i;
  double rho_w;
  double g;
};

} // Namespace FELIX

#endif // FELIX_EFFECTIVE_PRESSURE_SURROGATE_HPP
