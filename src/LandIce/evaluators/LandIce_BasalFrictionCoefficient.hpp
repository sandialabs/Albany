//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_BASAL_FRICTION_COEFFICIENT_HPP
#define LANDICE_BASAL_FRICTION_COEFFICIENT_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Basal friction coefficient evaluator

    This evaluator computes the friction coefficient beta for basal natural BC
*/

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes, bool ThermoCoupled>
class BasalFrictionCoefficient : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

  typedef typename std::conditional<IsStokes,ScalarT,ParamScalarT>::type       IceScalarT;
  typedef typename std::conditional<IsHydrology,ScalarT,ParamScalarT>::type    HydroScalarT;
  typedef typename std::conditional<ThermoCoupled,ScalarT,ParamScalarT>::type  TempScalarT;

  BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  void evaluateFieldsSide (typename Traits::EvalData d, ScalarT mu, ScalarT lambda, ScalarT power);
  void evaluateFieldsCell (typename Traits::EvalData d, ScalarT mu, ScalarT lambda, ScalarT power);

  // Coefficients for computing beta (if not given)
  PHX::MDField<const ScalarT,Dim> muParam;              // Coulomb friction coefficient
  PHX::MDField<const ScalarT,Dim> lambdaParam;          // Bed bumps avg length divided by bed bumps avg slope (for REGULARIZED_COULOMB only)
  PHX::MDField<const ScalarT,Dim> powerParam;           // Exponent (for POWER_LAW and REGULARIZED COULOMB only)

  ScalarT printedMu;
  ScalarT printedLambda;
  ScalarT printedQ;

  double beta_given_val;  // Constant value (for CONSTANT only)

  // Input:
  PHX::MDField<const ParamScalarT>      beta_given_field;
  PHX::MDField<const RealType>          BF;
  PHX::MDField<const IceScalarT>        u_norm;
  PHX::MDField<const ParamScalarT>      lambdaField;
  PHX::MDField<const HydroScalarT>      N;
  PHX::MDField<const MeshScalarT>       coordVec;

  PHX::MDField<const TempScalarT>       ice_softness;

  PHX::MDField<const ParamScalarT>      bed_topo_field;
  PHX::MDField<const ParamScalarT>      thickness_field;

  // Output:
  PHX::MDField<ScalarT>       beta;

  std::string                 basalSideName;  // Only if IsStokes=true

  bool use_stereographic_map, zero_on_floating;

  double x_0;
  double y_0;
  double R2;

  double rho_i, rho_w;

  int numNodes;
  int numQPs;

  bool logParameters;
  bool distributedLambda;
  bool nodal;

  enum BETA_TYPE {GIVEN_CONSTANT, GIVEN_FIELD, EXP_GIVEN_FIELD, GAL_PROJ_EXP_GIVEN_FIELD, POWER_LAW, REGULARIZED_COULOMB};
  BETA_TYPE beta_type;

  PHAL::MDFieldMemoizer<Traits> memoizer;
};

} // Namespace LandIce

#endif // LANDICE_BASAL_FRICTION_COEFFICIENT_HPP
