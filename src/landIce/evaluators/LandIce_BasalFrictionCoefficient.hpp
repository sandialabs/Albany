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

  // Coefficients for computing beta (if not given)
  double n;                                             // [adim] exponent of Glen's law
  PHX::MDField<const ScalarT,Dim> muParam;           // [yr^q m^{-q}], friction coefficient of the power Law with exponent q
  PHX::MDField<const ScalarT,Dim> lambdaParam;          // [km],  Bed bumps avg length divided by bed bumps avg slope (for REGULARIZED_COULOMB only)
  PHX::MDField<const ScalarT,Dim> powerParam;           // [adim], Exponent (for POWER_LAW and REGULARIZED COULOMB only)

  ScalarT printedMu;
  ScalarT printedLambda;
  ScalarT printedQ;

  double beta_val;   // beta value [kPa yr/m] (for CONSTANT only)
  double N_val; // effective pressure value [kPa]

  // Input:
  PHX::MDField<const RealType>          BF;
  PHX::MDField<const VelocityST>        u_norm;          // [m yr^{-1}]
  PHX::MDField<const ParamScalarT>      lambdaField;     // [km], characteristic length
  PHX::MDField<const ParamScalarT>      muField;         // [yr^q m^{-q}] or [adim], Power Law with exponent q, Coulomb Friction
  PHX::MDField<const EffPressureST>     N;               // [kPa]
  PHX::MDField<const MeshScalarT>       coordVec;        // [km]

  PHX::MDField<const TemperatureST>     ice_softness;    // [(kPa)^{-n} (kyr)^{-1}]

  PHX::MDField<const MeshScalarT>       bed_topo_field;  // [km]
  PHX::MDField<const MeshScalarT>       thickness_field; // [km]

  // Output:
  PHX::MDField<ScalarT>       beta;     // [kPa yr m^{-1}]

  std::string                 basalSideName;  // Only if is_side_equation=true

  bool use_stereographic_map, zero_on_floating, zero_N_on_floating_at_nodes;

  double x_0;             // [km]
  double y_0;             // [km]
  double R2;              // [km]

  double rho_i, rho_w;    // [kg m^{-3}]
  double g;               // [m s^{-2}]

  bool use_pressurized_bed;
  double overburden_fraction;  // [adim]
  double pressure_smoothing_length_scale; //[km]

  ParamScalarT mu;
  ParamScalarT lambda;
  ParamScalarT power;

  int numNodes;
  int numQPs;
  int dim;
  int worksetSize;

  bool logParameters;
  bool nodal;
  bool is_side_equation;
  bool is_power_parameter;
  enum class BETA_TYPE {CONSTANT, FIELD, POWER_LAW, REGULARIZED_COULOMB};
  enum class FIELD_TYPE {CONSTANT, FIELD, EXPONENT_OF_FIELD, EXPONENT_OF_FIELD_AT_NODES};
  enum class EFFECTIVE_PRESSURE_TYPE {CONSTANT, FIELD, HYDROSTATIC, HYDROSTATIC_AT_NODES};
  BETA_TYPE beta_type;
  EFFECTIVE_PRESSURE_TYPE effectivePressure_type;
  FIELD_TYPE mu_type, lambda_type;

  PHAL::MDFieldMemoizer<Traits> memoizer;

  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct BasalFrictionCoefficient_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,BasalFrictionCoefficient_Tag> BasalFrictionCoefficient_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const BasalFrictionCoefficient_Tag& tag, const int& i) const;

};

} // Namespace LandIce

#endif // LANDICE_BASAL_FRICTION_COEFFICIENT_HPP
