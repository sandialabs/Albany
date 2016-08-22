//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_BASAL_FRICTION_COEFFICIENT_HPP
#define FELIX_BASAL_FRICTION_COEFFICIENT_HPP 1

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

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes>
class BasalFrictionCoefficient : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  typedef typename std::conditional<IsStokes,ScalarT,ParamScalarT>::type     IceScalarT;
  typedef typename std::conditional<IsHydrology,ScalarT,ParamScalarT>::type  HydroScalarT;

  BasalFrictionCoefficient (const Teuchos::ParameterList& p,
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

  double beta_given_val;  // Constant value (for CONSTANT only)
  double A;               // Constant value for the flowFactorA field (for REGULARIZED_COULOMB only)

  // Input:
  PHX::MDField<ParamScalarT>  beta_given_field;
  PHX::MDField<IceScalarT>    u_norm;
  PHX::MDField<HydroScalarT>  N;
  PHX::MDField<MeshScalarT>   coordVec;

  // Output:
  PHX::MDField<ScalarT>       beta;

  std::string                 basalSideName;  // Only if IsStokes=true

  bool use_stereographic_map;

  double x_0;
  double y_0;
  double R2;

  int numNodes;
  int numQPs;

  bool logParameters;
  bool regularize;

  enum BETA_TYPE {GIVEN_CONSTANT, EXP_GIVEN_FIELD, GIVEN_FIELD, POWER_LAW, REGULARIZED_COULOMB};
  BETA_TYPE beta_type;
};

} // Namespace FELIX

#endif // FELIX_BASAL_FRICTION_COEFFICIENT_HPP
