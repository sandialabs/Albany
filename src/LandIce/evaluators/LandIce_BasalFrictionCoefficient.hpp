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
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Utilities.hpp"
#include "PHAL_Dimension.hpp"

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
  double n;                                             // [adim] exponent of Glen's law
  PHX::MDField<const ScalarT,Dim> muPowerLaw;           // [yr^q m^{-q}], friction coefficient of the power Law with exponent q
  PHX::MDField<const ScalarT,Dim> muCoulomb;            // [adim], Coulomb friction coefficient
  PHX::MDField<const ScalarT,Dim> lambdaParam;          // [km],  Bed bumps avg length divided by bed bumps avg slope (for REGULARIZED_COULOMB only)
  PHX::MDField<const ScalarT,Dim> powerParam;           // [adim], Exponent (for POWER_LAW and REGULARIZED COULOMB only)

  ScalarT printedMu;
  ScalarT printedLambda;
  ScalarT printedQ;

  double given_val;  // Constant value (for CONSTANT only)

  // Input:
  PHX::MDField<const ParamScalarT>      given_field;     // [KPa yr m^{-1}]
  PHX::MDField<const RealType>          BF;
  PHX::MDField<const IceScalarT>        u_norm;          // [m yr^{-1}]
  PHX::MDField<const ParamScalarT>      lambdaField;     // [km],  q is the power in the Regularized Coulomb Friction and n is the Glen's law exponent
  PHX::MDField<const HydroScalarT>      N;               // [kPa]
  PHX::MDField<const MeshScalarT>       coordVec;        // [km]

  PHX::MDField<const TempScalarT>       ice_softness;    // [(kPa)^{-n} (kyr)^{-1}]

  PHX::MDField<const ParamScalarT>      bed_topo_field;  // [km]
  PHX::MDField<const ParamScalarT>      thickness_field; // [km]

  // Output:
  PHX::MDField<ScalarT>       beta;     // [kPa yr m^{-1}]

  std::string                 basalSideName;  // Only if IsStokes=true

  bool use_stereographic_map, zero_on_floating;

  double x_0;             // [km]
  double y_0;             // [km]
  double R2;              // [km]

  double rho_i, rho_w;    // [kg m^{-3}]

  int numNodes;
  int numQPs;

  bool logParameters;
  bool distributedLambda; // [km]
  bool nodal;

  enum BETA_TYPE {GIVEN_CONSTANT, GIVEN_FIELD, EXP_GIVEN_FIELD, GAL_PROJ_EXP_GIVEN_FIELD, POWER_LAW, REGULARIZED_COULOMB};
  BETA_TYPE beta_type;

  PHAL::MDFieldMemoizer<Traits> memoizer;
};

} // Namespace LandIce

#endif // LANDICE_BASAL_FRICTION_COEFFICIENT_HPP
