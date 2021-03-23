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

template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
class BasalFrictionCoefficient : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::MeshScalarT   MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;

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
  PHX::MDField<const RealType>          given_field;        // [KPa yr m^{-1}]
  PHX::MDField<const ParamScalarT>      given_field_param;  // [KPa yr m^{-1}]
  PHX::MDField<const RealType>          BF;
  PHX::MDField<const VelocityST>        u_norm;          // [m yr^{-1}]
  PHX::MDField<const ParamScalarT>      lambdaField;     // [km],  q is the power in the Regularized Coulomb Friction and n is the Glen's law exponent
  PHX::MDField<const ParamScalarT>      muPowerLawField; // [yr^q m^{-q}], friction coefficient of the power Law with exponent q
  PHX::MDField<const ParamScalarT>      muCoulombField;  // [adim], Coulomb friction coefficient
  PHX::MDField<const EffPressureST>     N;               // [kPa]
  PHX::MDField<const MeshScalarT>       coordVec;        // [km]

  PHX::MDField<const TemperatureST>     ice_softness;    // [(kPa)^{-n} (kyr)^{-1}]

  PHX::MDField<const RealType>          bed_topo_field;  // [km]
  PHX::MDField<const MeshScalarT>       bed_topo_field_mst;  // [km]
  PHX::MDField<const RealType>          thickness_field; // [km]
  PHX::MDField<const ParamScalarT>      thickness_param_field; // [km]

  // Output:
  PHX::MDField<ScalarT>       beta;     // [kPa yr m^{-1}]

  std::string                 basalSideName;  // Only if is_side_equation=true

  bool use_stereographic_map, zero_on_floating, interpolate_then_exponentiate;

  double x_0;             // [km]
  double y_0;             // [km]
  double R2;              // [km]

  double rho_i, rho_w;    // [kg m^{-3}]

  ParamScalarT mu;
  ParamScalarT lambda;
  ParamScalarT power;

  unsigned int numNodes;
  unsigned int numQPs;
  unsigned int dim;

  bool useCollapsedSidesets;

  bool logParameters;
  bool distributedLambda;
  bool distributedMu;
  bool nodal;
  bool is_side_equation;
  bool is_thickness_param;
  bool is_given_field_param;

  enum BETA_TYPE {GIVEN_CONSTANT, GIVEN_FIELD, EXP_GIVEN_FIELD, POWER_LAW, REGULARIZED_COULOMB};
  BETA_TYPE beta_type;

  PHAL::MDFieldMemoizer<Traits> memoizer;

  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct GivenFieldParam_Tag{};
  struct GivenField_Tag{};
  struct PowerLaw_DistributedMu_Tag{};
  struct PowerLaw_Tag{};
  struct RegularizedCoulomb_DistributedLambda_DistributedMu_Tag{};
  struct RegularizedCoulomb_DistributedLambda_Tag{};
  struct RegularizedCoulomb_DistributedMu_Tag{};
  struct RegularizedCoulomb_Tag{};
  struct ExpGivenFieldParam_Nodal_Tag{};
  struct ExpGivenFieldParam_Tag{};
  struct ExpGivenField_Nodal_Tag{};
  struct ExpGivenField_Tag{};
  struct StereographicMapCorrection_Tag{};

  struct Side_GivenFieldParam_Tag{};
  struct Side_GivenField_Tag{};
  struct Side_PowerLaw_DistributedMu_Tag{};
  struct Side_PowerLaw_Tag{};
  struct Side_RegularizedCoulomb_DistributedLambda_DistributedMu_Tag{};
  struct Side_RegularizedCoulomb_DistributedLambda_Tag{};
  struct Side_RegularizedCoulomb_DistributedMu_Tag{};
  struct Side_RegularizedCoulomb_Tag{};
  struct Side_ExpGivenFieldParam_Nodal_Tag{};
  struct Side_ExpGivenFieldParam_Tag{};
  struct Side_ExpGivenField_Nodal_Tag{};
  struct Side_ExpGivenField_Tag{};
  struct Side_ZeroOnFloatingParam_Tag{};
  struct Side_ZeroOnFloating_Tag{};
  struct Side_StereographicMapCorrection_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,GivenFieldParam_Tag> GivenFieldParam_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,GivenField_Tag> GivenField_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PowerLaw_DistributedMu_Tag> PowerLaw_DistributedMu_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PowerLaw_Tag> PowerLaw_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,RegularizedCoulomb_DistributedLambda_DistributedMu_Tag> RegularizedCoulomb_DistributedLambda_DistributedMu_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,RegularizedCoulomb_DistributedLambda_Tag> RegularizedCoulomb_DistributedLambda_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,RegularizedCoulomb_DistributedMu_Tag> RegularizedCoulomb_DistributedMu_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,RegularizedCoulomb_Tag> RegularizedCoulomb_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,ExpGivenFieldParam_Nodal_Tag> ExpGivenFieldParam_Nodal_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,ExpGivenFieldParam_Tag> ExpGivenFieldParam_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,ExpGivenField_Nodal_Tag> ExpGivenField_Nodal_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,ExpGivenField_Tag> ExpGivenField_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,StereographicMapCorrection_Tag> StereographicMapCorrection_Policy;

  typedef Kokkos::RangePolicy<ExecutionSpace,Side_GivenFieldParam_Tag> Side_GivenFieldParam_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_GivenField_Tag> Side_GivenField_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_PowerLaw_DistributedMu_Tag> Side_PowerLaw_DistributedMu_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_PowerLaw_Tag> Side_PowerLaw_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_RegularizedCoulomb_DistributedLambda_DistributedMu_Tag> Side_RegularizedCoulomb_DistributedLambda_DistributedMu_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_RegularizedCoulomb_DistributedLambda_Tag> Side_RegularizedCoulomb_DistributedLambda_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_RegularizedCoulomb_DistributedMu_Tag> Side_RegularizedCoulomb_DistributedMu_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_RegularizedCoulomb_Tag> Side_RegularizedCoulomb_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_ExpGivenFieldParam_Nodal_Tag> Side_ExpGivenFieldParam_Nodal_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_ExpGivenFieldParam_Tag> Side_ExpGivenFieldParam_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_ExpGivenField_Nodal_Tag> Side_ExpGivenField_Nodal_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_ExpGivenField_Tag> Side_ExpGivenField_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_ZeroOnFloatingParam_Tag> Side_ZeroOnFloatingParam_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_ZeroOnFloating_Tag> Side_ZeroOnFloating_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Side_StereographicMapCorrection_Tag> Side_StereographicMapCorrection_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const GivenFieldParam_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const GivenField_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PowerLaw_DistributedMu_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PowerLaw_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const RegularizedCoulomb_DistributedLambda_DistributedMu_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const RegularizedCoulomb_DistributedLambda_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const RegularizedCoulomb_DistributedMu_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const RegularizedCoulomb_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const ExpGivenFieldParam_Nodal_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const ExpGivenFieldParam_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const ExpGivenField_Nodal_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const ExpGivenField_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const StereographicMapCorrection_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_GivenFieldParam_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_GivenField_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_PowerLaw_DistributedMu_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_PowerLaw_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_RegularizedCoulomb_DistributedLambda_DistributedMu_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_RegularizedCoulomb_DistributedLambda_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_RegularizedCoulomb_DistributedMu_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_RegularizedCoulomb_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_ExpGivenFieldParam_Nodal_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_ExpGivenFieldParam_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_ExpGivenField_Nodal_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_ExpGivenField_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_ZeroOnFloatingParam_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_ZeroOnFloating_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Side_StereographicMapCorrection_Tag& tag, const int& i) const;

};

} // Namespace LandIce

#endif // LANDICE_BASAL_FRICTION_COEFFICIENT_HPP
