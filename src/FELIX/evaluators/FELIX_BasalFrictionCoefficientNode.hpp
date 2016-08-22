//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_BASAL_FRICTION_COEFFICIENT_NODE_HPP
#define FELIX_BASAL_FRICTION_COEFFICIENT_NODE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Basal friction coefficient evaluator

    This evaluator computes the friction coefficient beta for basal natural BC
*/
template<typename EvalT, typename Traits>
class BasalFrictionCoefficientNode : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  BasalFrictionCoefficientNode (const Teuchos::ParameterList& p,
                                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  // Coefficients for computing beta (if not given)
  PHX::MDField<ScalarT,Dim> muParam;              // Coulomb friction coefficient
  PHX::MDField<ScalarT,Dim> lambdaParam;          // Bed bumps avg length divided by bed bumps avg slope (for REGULARIZED_COULOMB only)
  PHX::MDField<ScalarT,Dim> powerParam;           // Exponent (for POWER_LAW and REGULARIZED COULOMB only)

  ScalarT printedMu;
  ScalarT printedLambda;
  ScalarT printedQ;

  double A;               // Constant value for the flowFactorA field (for REGULARIZED_COULOMB only)

  // Input:
  PHX::MDField<ParamScalarT>  u_norm;
  PHX::MDField<ParamScalarT>  N;

  // Output:
  PHX::MDField<ScalarT>       beta;

  int numNodes;

  bool logParameters;
  bool regularize;
};

} // Namespace FELIX

#endif // FELIX_BASAL_FRICTION_COEFFICIENT_NODE_HPP
