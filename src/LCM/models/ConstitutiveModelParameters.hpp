//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveModelParameters_hpp)
#define LCM_ConstitutiveModelParameters_hpp

#include "Albany_config.h"

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_ParameterList.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_KL_ExponentialRandomField.hpp"
#endif
#include "Albany_Layouts.hpp"
#include "Teuchos_Array.hpp"

namespace LCM {
///
/// \brief Evaluates a selecltion of Constitutive Model Parameters
/// Either as a constant or a truncated KL expansion.
///
template <typename EvalT, typename Traits>
class ConstitutiveModelParameters
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>,
      public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  ///
  /// Constructor
  ///
  ConstitutiveModelParameters(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

  ///
  /// Sacado method to access parameter values
  ///
  ScalarT&
  getValue(const std::string& n);

  ///
  /// Helper method to parse a parameter
  ///
  void
  parseParameters(
      const std::string&      n,
      Teuchos::ParameterList& pl,
      Teuchos::RCP<ParamLib>  paramLib);

 private:
  ///
  /// Number of integration points
  ///
  int num_pts_;

  ///
  /// Number of spatial dimensions
  ///
  int num_dims_;

  ///
  /// spatial locations of integration points
  ///
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coord_vec_;

  ///
  /// Constitutive Model Parameters
  ///
  /// Elastic Moduli
  PHX::MDField<ScalarT, Cell, QuadPoint> elastic_mod_;
  PHX::MDField<ScalarT, Cell, QuadPoint> poissons_ratio_;
  PHX::MDField<ScalarT, Cell, QuadPoint> bulk_mod_;
  PHX::MDField<ScalarT, Cell, QuadPoint> shear_mod_;
  PHX::MDField<ScalarT, Cell, QuadPoint> c11_;
  PHX::MDField<ScalarT, Cell, QuadPoint> c12_;
  PHX::MDField<ScalarT, Cell, QuadPoint> c44_;
  /// Plasticity Parameters
  PHX::MDField<ScalarT, Cell, QuadPoint> yield_strength_;
  PHX::MDField<ScalarT, Cell, QuadPoint> hardening_mod_;
  PHX::MDField<ScalarT, Cell, QuadPoint> recovery_mod_;
  PHX::MDField<ScalarT, Cell, QuadPoint> flow_coeff_;
  PHX::MDField<ScalarT, Cell, QuadPoint> flow_exp_;
  ///  Concentration parameters
  PHX::MDField<ScalarT, Cell, QuadPoint> conc_eq_param_;
  PHX::MDField<ScalarT, Cell, QuadPoint> diff_coeff_;
  ///  Thermal parameters
  PHX::MDField<ScalarT, Cell, QuadPoint> thermal_cond_;
  ///  ACE parameters
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_density_;
  PHX::MDField<ScalarT, Cell, QuadPoint> water_density_;
  PHX::MDField<ScalarT, Cell, QuadPoint> sediment_density_;
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_heat_capacity_;
  PHX::MDField<ScalarT, Cell, QuadPoint> water_heat_capacity_;
  PHX::MDField<ScalarT, Cell, QuadPoint> sediment_heat_capacity_;
  PHX::MDField<ScalarT, Cell, QuadPoint> max_ice_saturation_;
  PHX::MDField<ScalarT, Cell, QuadPoint> init_ice_saturation_;
  PHX::MDField<ScalarT, Cell, QuadPoint> ice_thermal_conductivity_;
  PHX::MDField<ScalarT, Cell, QuadPoint> water_thermal_conductivity_;
  PHX::MDField<ScalarT, Cell, QuadPoint> sediment_thermal_conductivity_;

  ///
  /// map of strings to specify parameter names to MDFields
  ///
  std::map<std::string, PHX::MDField<ScalarT, Cell, QuadPoint>> field_map_;

  ///
  /// map of flags to specify if a parameter is constant
  ///
  std::map<std::string, bool> is_constant_map_;

  ///
  /// map of strings to ScalarTs to specify constant values
  ///
  std::map<std::string, ScalarT> constant_value_map_;

  ///
  /// Optional dependence on Temperature
  ///
  bool                                         have_temperature_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> temperature_;
  std::map<std::string, std::string>           temp_type_map_;
  std::map<std::string, RealType>              dparam_dtemp_map_;
  std::map<std::string, RealType>              ref_temp_map_;
  std::map<std::string, RealType>              pre_exp_map_;
  std::map<std::string, RealType>              exp_param_map_;

#ifdef ALBANY_STOKHOS
  //! map of strings to exponential random fields
  std::map<
      std::string,
      Teuchos::RCP<Stokhos::KL::ExponentialRandomField<RealType>>>
      exp_rf_kl_map_;
#endif

  //! map of strings to Arrays of values of the random variables
  std::map<std::string, Teuchos::Array<ScalarT>> rv_map_;

  //! storing the DataLayouts
  const Teuchos::RCP<Albany::Layouts>& dl_;

  ScalarT dummy;

  // Kokkos kernels

 public:
  struct is_constant_map_Tag
  {
  };
  struct no_const_map_Tag
  {
  };
  struct is_const_map_have_temperature_Linear_Tag
  {
  };
  struct is_const_map_have_temperature_Arrhenius_Tag
  {
  };
  struct no_const_map_have_temperature_Linear_Tag
  {
  };
  struct no_const_map_have_temperature_Arrhenius_Tag
  {
  };

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy<ExecutionSpace, is_constant_map_Tag>
      is_const_map_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, no_const_map_Tag>
      no_const_map_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, is_const_map_have_temperature_Linear_Tag>
          is_const_map_have_temperature_Linear_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, is_const_map_have_temperature_Arrhenius_Tag>
          is_const_map_have_temperature_Arrhenius_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, no_const_map_have_temperature_Linear_Tag>
          no_const_map_have_temperature_Linear_Policy;
  typedef Kokkos::
      RangePolicy<ExecutionSpace, no_const_map_have_temperature_Arrhenius_Tag>
          no_const_map_have_temperature_Arrhenius_Policy;

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const is_constant_map_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const no_const_map_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const is_const_map_have_temperature_Linear_Tag& tag, const int& i)
      const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(
      const is_const_map_have_temperature_Arrhenius_Tag& tag,
      const int&                                         i) const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const no_const_map_have_temperature_Linear_Tag& tag, const int& i)
      const;
  KOKKOS_INLINE_FUNCTION
  void
  operator()(
      const no_const_map_have_temperature_Arrhenius_Tag& tag,
      const int&                                         i) const;

  KOKKOS_INLINE_FUNCTION
  void
  compute_second_constMap(const int cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  compute_second_no_constMap(const int cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  compute_temperature_Linear(const int cell) const;
  KOKKOS_INLINE_FUNCTION
  void
  compute_temperature_Arrhenius(const int cell) const;
};
}  // namespace LCM

#endif
