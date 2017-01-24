//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_EFFECTIVE_PRESSURE_HPP
#define FELIX_EFFECTIVE_PRESSURE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Effective Pressure Evaluator

    This evaluator evaluates the effective pressure at the basal side
*/

template<typename EvalT, typename Traits, bool OnSide, bool Surrogate>
class EffectivePressure : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  typedef typename std::conditional<Surrogate,ParamScalarT,ScalarT>::type  HydroScalarT;

  EffectivePressure (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  void evaluateFieldsSide (typename Traits::EvalData workset);
  void evaluateFieldsCell (typename Traits::EvalData workset);

  // Input:
  PHX::MDField<ParamScalarT>  H;
  PHX::MDField<ParamScalarT>  z_s;
  PHX::MDField<HydroScalarT>  phi;

  // Output:
  PHX::MDField<HydroScalarT>  N;

  int numNodes;

  double rho_i;
  double rho_w;
  double g;

  std::string basalSideName; // Needed if OnSide=true

  // Parameters needed for Stokes alone case
  bool   regularized;
  PHX::MDField<ScalarT,Dim> alphaParam;
  PHX::MDField<ScalarT,Dim> regularizationParam;
  ScalarT printedAlpha;
};

} // Namespace FELIX

#endif // FELIX_EFFECTIVE_PRESSURE_HPP
