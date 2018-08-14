//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ParallelConstitutiveModel_hpp)
#define LCM_ParallelConstitutiveModel_hpp

#include <functional>
#include <memory>
#include "ConstitutiveModel.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"

namespace LCM {

template <typename S>
using FieldMap = std::map<std::string, Teuchos::RCP<PHX::MDField<S>>>;

template <typename EvalT, typename Traits>
struct ParallelKernel
{
  using DataLayoutMap = std::map<std::string, Teuchos::RCP<PHX::DataLayout>>;
  using NameMap       = std::map<std::string, std::string>;

  using ScalarT          = typename EvalT::ScalarT;
  using MeshScalarT      = typename EvalT::MeshScalarT;
  using Workset          = typename Traits::EvalData;
  using ScalarField      = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<const ScalarT>;

 protected:
  ParallelKernel(ConstitutiveModel<EvalT, Traits>& model)
      : model_(model),
        field_name_map_(*model.field_name_map_),
        dep_field_map_(model.dep_field_map_),
        eval_field_map_(model.eval_field_map_),
        num_dims_(model.num_dims_),
        num_pts_(model.num_pts_),
        need_integration_pt_locations_(model.need_integration_pt_locations_),
        compute_energy_(model.compute_energy_),
        compute_tangent_(model.compute_tangent_),
        have_temperature_(model.have_temperature_),
        have_damage_(model.have_damage_),
        have_total_concentration_(model.have_total_concentration_),
        have_total_bubble_density_(model.have_total_bubble_density_),
        have_bubble_volume_fraction_(model.have_bubble_volume_fraction_),
        coord_vec_(model.coord_vec_),
        temperature_(model.temperature_),
        total_concentration_(model.total_concentration_),
        total_bubble_density_(model.total_bubble_density_),
        bubble_volume_fraction_(model.bubble_volume_fraction_),
        damage_(model.damage_),
        weights_(model.weights_),
        j_(model.j_),
        expansion_coeff_(model.expansion_coeff_),
        ref_temperature_(model.ref_temperature_),
        heat_capacity_(model.heat_capacity_),
        density_(model.density_),
        latent_heat_(model.latent_heat_)
  {
  }

  void
  setDependentField(
      std::string const&                   field_name,
      Teuchos::RCP<PHX::DataLayout> const& field)
  {
    model_.setDependentField(field_name, field);
  }

  void
  setEvaluatedField(
      std::string const&                   field_name,
      Teuchos::RCP<PHX::DataLayout> const& field)
  {
    model_.setEvaluatedField(field_name, field);
  }

  void
  addStateVariable(
      std::string const&            name,
      Teuchos::RCP<PHX::DataLayout> layout,
      std::string const&            init_type,
      double                        init_value,
      bool                          old_state_flag,
      bool                          output_flag)
  {
    model_.addStateVar(
        name, layout, init_type, init_value, old_state_flag, output_flag);
  }

  void
  extractEvaluatedFieldArray(
      std::string const&                      field_name,
      std::size_t                             num,
      std::vector<Teuchos::RCP<ScalarField>>& state,
      std::vector<Albany::MDArray*>&          old_state,
      FieldMap<ScalarT>&                      eval_fields,
      Workset&                                workset);

  void
  extractEvaluatedFieldArray(
      std::string const&                      field_name,
      std::size_t                             num,
      std::vector<Teuchos::RCP<ScalarField>>& state,
      FieldMap<ScalarT>&                      eval_fields);

  ConstitutiveModel<EvalT, Traits>& model_;

  ///
  /// Map of field names
  ///
  NameMap& field_name_map_;

  DataLayoutMap& dep_field_map_;

  DataLayoutMap& eval_field_map_;

  int num_dims_;

  int num_pts_;

  bool need_integration_pt_locations_;

  bool compute_energy_;

  bool compute_tangent_;

  bool have_temperature_;

  bool have_damage_;

  bool have_total_concentration_;

  bool have_total_bubble_density_;

  bool have_bubble_volume_fraction_;

  /// optional integration point locations field
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coord_vec_;

  /// optional temperature field
  PHX::MDField<const ScalarT, Cell, QuadPoint> temperature_;

  /// Optional total concentration field
  PHX::MDField<const ScalarT, Cell, QuadPoint> total_concentration_;

  /// Optional total (He) bubble density field
  PHX::MDField<const ScalarT, Cell, QuadPoint> total_bubble_density_;

  /// Optional (He) bubble volume fraction field
  PHX::MDField<const ScalarT, Cell, QuadPoint> bubble_volume_fraction_;

  /// optional scalar damage field
  PHX::MDField<const ScalarT, Cell, QuadPoint> damage_;

  /// optional integration weights field
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> weights_;

  ///< optional J field
  PHX::MDField<const ScalarT, Cell, QuadPoint> j_;

  ///< Thermal Expansion Coefficient
  RealType expansion_coeff_;

  ///< Reference Temperature
  RealType ref_temperature_;

  RealType heat_capacity_;

  RealType density_;

  RealType latent_heat_;

  /// Flag indicating failure in model calculation
  Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test_{
      Teuchos::null};
};

template <typename EvalT, typename Traits, typename Kernel>
class ParallelConstitutiveModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using ScalarT     = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  using EvalKernel  = Kernel;

  using ConstitutiveModel<EvalT, Traits>::num_pts_;

  ParallelConstitutiveModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~ParallelConstitutiveModel() = default;

  void
  computeState(
      typename Traits::EvalData workset,
      FieldMap<const ScalarT>   dep_fields,
      FieldMap<ScalarT>         eval_fields) final;

  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      FieldMap<const ScalarT>   dep_fields,
      FieldMap<ScalarT>         eval_fields) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

 protected:
  std::unique_ptr<EvalKernel> kernel_;
};

}  // namespace LCM

#endif
