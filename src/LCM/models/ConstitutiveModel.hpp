//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveModel_hpp)
#define LCM_ConstitutiveModel_hpp

#include "AbstractModel.hpp"

namespace LCM
{

///
/// Constitutive Model Base Class
///
template<typename EvalT, typename Traits>
class ConstitutiveModel : public AbstractModel<EvalT, Traits>
{
public:

  using ScalarT = typename AbstractModel<EvalT, Traits>::ScalarT;
  using MeshScalarT = typename AbstractModel<EvalT, Traits>::MeshScalarT;
  using Workset = typename AbstractModel<EvalT, Traits>::Workset;

  using FieldMap = typename AbstractModel<EvalT, Traits>::FieldMap;
  using DataLayoutMap = typename AbstractModel<EvalT, Traits>::DataLayoutMap;

  ///
  /// Constructor
  ///
  ConstitutiveModel(
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  ///
  /// Virtual Destructor
  ///
  virtual
  ~ConstitutiveModel()
  {
    return;
  };

  ///
  /// No copy constructor
  ///
  ConstitutiveModel(ConstitutiveModel const &) = delete;

  ///
  /// No copy assignment
  ///
  ConstitutiveModel &
  operator=(ConstitutiveModel const &) = delete;

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(
      Workset workset,
      FieldMap dep_fields,
      FieldMap eval_fields) = 0;

  virtual
  void
  computeStateParallel(
      Workset workset,
      FieldMap dep_fields,
      FieldMap eval_fields) = 0;

  ///
  /// Optional Method to volume average the pressure
  ///
  void
  computeVolumeAverage(
      Workset workset,
      FieldMap dep_fields,
      FieldMap eval_fields);

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
  setCoordVecField(PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> coord_vec)
  {
    coord_vec_ = coord_vec;
  }

  ///
  /// set the Temperature field
  ///
  void
  setTemperatureField(PHX::MDField<ScalarT, Cell, QuadPoint> temperature)
  {
    temperature_ = temperature;
  }

  ///
  /// set the damage field
  ///
  void
  setDamageField(PHX::MDField<ScalarT, Cell, QuadPoint> damage)
  {
    damage_ = damage;
  }

  ///
  /// set the total concentration
  ///
  void
  setTotalConcentrationField(PHX::MDField<ScalarT, Cell, QuadPoint> total_concentration)
  {
    total_concentration_ = total_concentration;
  }

  ///
  /// set the total bubble density
  ///
  void
  setTotalBubbleDensityField(PHX::MDField<ScalarT, Cell, QuadPoint> total_bubble_density)
  {
    total_bubble_density_ = total_bubble_density;
  }

  ///
  /// set the bubble volume fraction
  ///
  void
  setBubbleVolumeFractionField(PHX::MDField<ScalarT, Cell, QuadPoint> bubble_volume_fraction)
  {
    bubble_volume_fraction_ = bubble_volume_fraction;
  }

  ///
  /// set the Weights field
  ///
  void
  setWeightsField(PHX::MDField<MeshScalarT, Cell, QuadPoint> weights)
  {
    weights_ = weights;
  }

  ///
  /// set the J field
  ///
  void
  setJField(PHX::MDField<ScalarT, Cell, QuadPoint> j)
  {
    j_ = j;
  }

protected:

  ///
  /// flag for integration point locations
  ///
  bool
  need_integration_pt_locations_{false};

  ///
  /// flag that the energy needs to be computed
  ///
  bool
  compute_energy_{false};

  ///
  /// flag that the tangent needs to be computed
  ///
  bool
  compute_tangent_{false};

  ///
  /// Bool for temperature
  ///
  bool
  have_temperature_{false};

  ///
  /// Bool for damage
  ///
  bool
  have_damage_{false};

  ///
  /// Bool for total concentration
  ///
  bool
  have_total_concentration_{false};

  ///
  /// Bool for total bubble density
  ///
  bool
  have_total_bubble_density_{false};

  ///
  /// Bool for bubble_volume_fraction
  ///
  bool
  have_bubble_volume_fraction_{false};

  ///
  /// optional integration point locations field
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim>
  coord_vec_;

  ///
  /// optional temperature field
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint>
  temperature_;

  ///
  /// Optional total concentration field
  ///
  PHX::MDField<ScalarT,Cell,QuadPoint>
  total_concentration_;

  ///
  /// Optional total (He) bubble density field
  ///
  PHX::MDField<ScalarT,Cell,QuadPoint>
  total_bubble_density_;

  ///
  /// Optional bubble volume fraction field
  ///
  PHX::MDField<ScalarT,Cell,QuadPoint>
  bubble_volume_fraction_;

  ///
  /// optional damage field
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint>
  damage_;

  ///
  /// optional weights field
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint>
  weights_;

  ///
  /// optional J field
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint>
  j_;

  ///
  /// Thermal Expansion Coefficient
  ///
  RealType
  expansion_coeff_{0.0};

  ///
  /// Reference Temperature
  ///
  RealType
  ref_temperature_{0.0};

  ///
  /// Heat Capacity
  ///
  RealType
  heat_capacity_{1.0};

  ///
  /// Density
  ///
  RealType
  density_{1.0};
};

} // namespace LCM

#endif
