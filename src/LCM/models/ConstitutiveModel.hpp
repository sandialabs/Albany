//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveModel_hpp)
#define LCM_ConstitutiveModel_hpp

#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
class ParallelKernel;

///
/// Constitutive Model Base Class
///
template <typename EvalT, typename Traits>
class ConstitutiveModel
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  using Workset     = typename Traits::EvalData;

  using FieldMap = std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>>;
  using DepFieldMap =
      std::map<std::string, Teuchos::RCP<PHX::MDField<const ScalarT>>>;
  using DataLayoutMap = std::map<std::string, Teuchos::RCP<PHX::DataLayout>>;

  ///
  /// Constructor
  ///
  ConstitutiveModel(
      Teuchos::ParameterList*              p,
      Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~ConstitutiveModel() { return; };

  ///
  /// No copy constructor
  ///
  ConstitutiveModel(ConstitutiveModel const&) = delete;

  ///
  /// No copy assignment
  ///
  ConstitutiveModel&
  operator=(ConstitutiveModel const&) = delete;

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      Workset     workset,
      DepFieldMap dep_fields,
      FieldMap    eval_fields) = 0;

  virtual void
  computeStateParallel(
      Workset     workset,
      DepFieldMap dep_fields,
      FieldMap    eval_fields) = 0;

  ///
  /// Optional Method to volume average the pressure
  ///
  void
  computeVolumeAverage(
      Workset     workset,
      DepFieldMap dep_fields,
      FieldMap    eval_fields);

  ///
  /// Accessors and mutators
  ///
  int
  getNumDimensions()
  {
    return num_dims_;
  }

  int
  getNumCubaturePoints()
  {
    return num_pts_;
  }

  int
  getNumStateVariables()
  {
    return num_state_variables_;
  }

  ///
  /// state variable registration helpers
  ///
  std::string
  getStateVarName(int state_var)
  {
    return state_var_names_[state_var];
  }

  Teuchos::RCP<PHX::DataLayout>
  getStateVarLayout(int state_var)
  {
    return state_var_layouts_[state_var];
  }

  std::string
  getStateVarInitType(int state_var)
  {
    return state_var_init_types_[state_var];
  }

  double
  getStateVarInitValue(int state_var)
  {
    return state_var_init_values_[state_var];
  }

  bool
  getStateVarOldStateFlag(int state_var)
  {
    return state_var_old_state_flags_[state_var];
  }

  bool
  getStateVarOutputFlag(int state_var)
  {
    return state_var_output_flags_[state_var];
  }

  void
  addStateVar(
      std::string const&            name,
      Teuchos::RCP<PHX::DataLayout> layout,
      std::string const&            init_type,
      double                        init_value,
      bool                          old_state_flag,
      bool                          output_flag)
  {
    ++num_state_variables_;
    state_var_names_.push_back(name);
    state_var_layouts_.push_back(layout);
    state_var_init_types_.push_back(init_type);
    state_var_init_values_.push_back(init_value);
    state_var_old_state_flags_.push_back(old_state_flag);
    state_var_output_flags_.push_back(output_flag);
  }

  ///
  /// Deal with fields
  ///

  Teuchos::RCP<std::map<std::string, std::string>>
  getFieldNameMap()
  {
    return field_name_map_;
  }

  DataLayoutMap
  getDependentFieldMap()
  {
    return dep_field_map_;
  }

  DataLayoutMap
  getEvaluatedFieldMap()
  {
    return eval_field_map_;
  }

  void
  setDependentField(
      std::string const&                   field_name,
      Teuchos::RCP<PHX::DataLayout> const& field)
  {
    dep_field_map_.insert(std::make_pair(field_name, field));
  }

  void
  setDependentFieldFromNameMap(
      std::string const&                   name_key,
      Teuchos::RCP<PHX::DataLayout> const& field)
  {
    std::string const name = (*field_name_map_)[name_key];
    setDependentField(name, field);
  }

  void
  setEvaluatedField(
      std::string const&                   field_name,
      Teuchos::RCP<PHX::DataLayout> const& field)
  {
    eval_field_map_.insert(std::make_pair(field_name, field));
  }

  void
  setEvaluatedFieldFromNameMap(
      std::string const&                   name_key,
      Teuchos::RCP<PHX::DataLayout> const& field)
  {
    std::string const name = (*field_name_map_)[name_key];
    setEvaluatedField(name, field);
  }

  ///
  /// Integration point location flag
  ///
  bool
  getIntegrationPointLocationFlag()
  {
    return need_integration_pt_locations_;
  }

  ///
  /// Integration point location set method
  ///
  void
  setCoordVecField(
      PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coord_vec)
  {
    coord_vec_ = coord_vec;
  }

  ///
  /// set the Temperature field
  ///
  void
  setTemperatureField(PHX::MDField<const ScalarT, Cell, QuadPoint> temperature)
  {
    temperature_ = temperature;
  }

  ///
  /// set the damage field
  ///
  void
  setDamageField(PHX::MDField<const ScalarT, Cell, QuadPoint> damage)
  {
    damage_ = damage;
  }

  ///
  /// set the total concentration
  ///
  void
  setTotalConcentrationField(
      PHX::MDField<const ScalarT, Cell, QuadPoint> total_concentration)
  {
    total_concentration_ = total_concentration;
  }

  ///
  /// set the total bubble density
  ///
  void
  setTotalBubbleDensityField(
      PHX::MDField<const ScalarT, Cell, QuadPoint> total_bubble_density)
  {
    total_bubble_density_ = total_bubble_density;
  }

  ///
  /// set the bubble volume fraction
  ///
  void
  setBubbleVolumeFractionField(
      PHX::MDField<const ScalarT, Cell, QuadPoint> bubble_volume_fraction)
  {
    bubble_volume_fraction_ = bubble_volume_fraction;
  }

  ///
  /// set the Weights field
  ///
  void
  setWeightsField(PHX::MDField<const MeshScalarT, Cell, QuadPoint> weights)
  {
    weights_ = weights;
  }

  ///
  /// set the J field
  ///
  void
  setJField(PHX::MDField<const ScalarT, Cell, QuadPoint> j)
  {
    j_ = j;
  }

 protected:
  friend class ParallelKernel<EvalT, Traits>;

  ///
  /// Number of dimensions
  ///
  int num_dims_{0};

  ///
  /// Number of integration points
  ///
  int num_pts_{0};

  std::vector<std::string> state_var_names_;

  std::vector<Teuchos::RCP<PHX::DataLayout>> state_var_layouts_;

  std::vector<std::string> state_var_init_types_;

  std::vector<double> state_var_init_values_;

  std::vector<bool> state_var_old_state_flags_;

  std::vector<bool> state_var_output_flags_;

  ///
  /// Map of field names
  ///
  Teuchos::RCP<std::map<std::string, std::string>> field_name_map_;

  DataLayoutMap dep_field_map_;

  DataLayoutMap eval_field_map_;

  ///
  /// Number of State Variables
  ///
  int num_state_variables_{0};

  ///
  /// flag for integration point locations
  ///
  bool need_integration_pt_locations_{false};

  ///
  /// flag that the energy needs to be computed
  ///
  bool compute_energy_{false};

  ///
  /// flag that the tangent needs to be computed
  ///
  bool compute_tangent_{false};

  ///
  /// Bool for temperature
  ///
  bool have_temperature_{false};

  ///
  /// Bool for damage
  ///
  bool have_damage_{false};

  ///
  /// Bool for total concentration
  ///
  bool have_total_concentration_{false};

  ///
  /// Bool for total bubble density
  ///
  bool have_total_bubble_density_{false};

  ///
  /// Bool for bubble_volume_fraction
  ///
  bool have_bubble_volume_fraction_{false};

  ///
  /// optional integration point locations field
  ///
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coord_vec_;

  ///
  /// optional temperature field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> temperature_;

  ///
  /// Optional total concentration field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> total_concentration_;

  ///
  /// Optional total (He) bubble density field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> total_bubble_density_;

  ///
  /// Optional bubble volume fraction field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> bubble_volume_fraction_;

  ///
  /// optional damage field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> damage_;

  ///
  /// optional weights field
  ///
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> weights_;

  ///
  /// optional J field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> j_;

  ///
  /// Thermal Expansion Coefficient
  ///
  RealType expansion_coeff_{0.0};

  ///
  /// Reference Temperature
  ///
  RealType ref_temperature_{0.0};

  ///
  /// Heat Capacity
  ///
  RealType heat_capacity_{1.0};

  ///
  /// Density
  ///
  RealType density_{1.0};

  ///
  /// Latent Heat
  ///
  RealType latent_heat_{1.0};

  ///
  /// Ice and Water Saturation
  ///
  RealType ice_saturation_{1.0};
  RealType water_saturation_{0.0};
};

}  // namespace LCM

#endif
