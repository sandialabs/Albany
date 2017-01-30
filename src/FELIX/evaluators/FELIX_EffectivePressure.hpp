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

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes>
class EffectivePressure;

// Partial specialization for Hydrology only problem
template<typename EvalT, typename Traits>
class EffectivePressure<EvalT,Traits,false,true> : public PHX::EvaluatorWithBaseImpl<Traits>,
                                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  EffectivePressure (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const ParamScalarT>  H;
  PHX::MDField<const ParamScalarT>  z_s;
  PHX::MDField<const ParamScalarT>  phi;
  PHX::MDField<const ScalarT,Dim> alphaParam;
  PHX::MDField<const ScalarT,Dim> regularizationParam;

  // Output:
  PHX::MDField<ParamScalarT>  N;

  std::string basalSideName;

  int numNodes;

  bool   regularized;
  double rho_i;
  double rho_w;
  double g;

  ScalarT printedAlpha;
};

// Partial specialization: coupled StokesFOHydrology problem
template<typename EvalT, typename Traits>
class EffectivePressure<EvalT,Traits,true,true> : public PHX::EvaluatorWithBaseImpl<Traits>,
                                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  EffectivePressure (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const ParamScalarT>  H;
  PHX::MDField<const ParamScalarT>  z_s;
  PHX::MDField<const ScalarT>       phi;

  // Output:
  PHX::MDField<ScalarT>       N;

  std::string basalSideName;

  int numNodes;

  double rho_i;
  double rho_w;
  double g;
};

// Partial specialization: Hydrology problem
template<typename EvalT, typename Traits>
class EffectivePressure<EvalT,Traits,true,false> : public PHX::EvaluatorWithBaseImpl<Traits>,
                                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT      ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  EffectivePressure (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const ParamScalarT>  H;
  PHX::MDField<const ParamScalarT>  z_s;
  PHX::MDField<const ScalarT>       phi;

  // Output:
  PHX::MDField<ScalarT>       N;

  int numNodes;

  double rho_i;
  double rho_w;
  double g;
};

} // Namespace FELIX

#endif // FELIX_EFFECTIVE_PRESSURE_HPP
