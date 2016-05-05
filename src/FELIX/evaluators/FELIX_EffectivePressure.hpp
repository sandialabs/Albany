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

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits>
class EffectivePressure : public PHX::EvaluatorWithBaseImpl<Traits>,
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

  ScalarT& getValue(const std::string &name);

private:

  // Input:
  PHX::MDField<ParamScalarT> H;
  PHX::MDField<ParamScalarT> z_s;
  PHX::MDField<ScalarT>      phi;

  // Output:
  PHX::MDField<ScalarT>      N;

  std::string basalSideName;

  int numNodes;

  bool   regularized;
  bool   surrogate;
  bool   stokes;
  double rho_i;
  double rho_w;
  double g;


  PHX::MDField<ScalarT,Dim> alphaField;
  ScalarT printedAlpha;
  ScalarT dummyParam;
};

} // Namespace FELIX

#endif // FELIX_EFFECTIVE_PRESSURE_HPP
