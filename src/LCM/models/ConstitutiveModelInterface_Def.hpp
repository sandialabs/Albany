//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

#include "AAAModel.hpp"
#include "ACEice.hpp"
#include "ACEpermafrost.hpp"
#include "AnisotropicDamageModel.hpp"
#include "AnisotropicHyperelasticDamageModel.hpp"
#include "AnisotropicViscoplasticModel.hpp"
#include "CapExplicitModel.hpp"
#include "CapImplicitModel.hpp"
#include "CreepModel.hpp"
#include "CrystalPlasticityModel.hpp"
#include "DruckerPragerModel.hpp"
#include "ElasticCrystalModel.hpp"
#include "ElasticDamageModel.hpp"
#include "ElastoViscoplasticModel.hpp"
#include "FM_AlbanyInterface.hpp"
#include "GursonHMRModel.hpp"
#include "GursonModel.hpp"
#include "HyperelasticDamageModel.hpp"
#include "J2FiberModel.hpp"
#include "J2HMCModel.hpp"
#include "J2MiniSolver.hpp"
#include "J2Model.hpp"
#include "LinearElasticModel.hpp"
#include "LinearHMCModel.hpp"
#include "LinearPiezoModel.hpp"
#include "MooneyRivlinModel.hpp"
#include "NeohookeanModel.hpp"
#include "NewtonianFluidModel.hpp"
#include "OrtizPandolfiModel.hpp"
#include "ParallelNeohookeanModel.hpp"
#include "RIHMRModel.hpp"
#include "StVenantKirchhoffModel.hpp"
#include "TvergaardHutchinsonModel.hpp"
#include "ViscoElasticModel.hpp"

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
ConstitutiveModelInterface<EvalT, Traits>::ConstitutiveModelInterface(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : have_temperature_(false),
      have_damage_(false),
      have_total_concentration_(false),
      have_total_bubble_density_(false),
      have_bubble_volume_fraction_(false),
      volume_average_pressure_(p.get<bool>("Volume Average Pressure", false))
{
  Teuchos::ParameterList* plist =
      p.get<Teuchos::ParameterList*>("Material Parameters");
  plist->set<bool>("Volume Average Pressure", volume_average_pressure_);
  this->initializeModel(plist, dl);

  // construct the dependent fields
  auto dependent_map = model_->getDependentFieldMap();
  for (auto& pair : dependent_map) {
    auto temp_field =
        Teuchos::rcp(new PHX::MDField<const ScalarT>(pair.first, pair.second));
    dep_fields_map_.insert(std::make_pair(pair.first, temp_field));
  }

  // register dependent fields
  typename decltype(dep_fields_map_)::iterator it;
  for (it = dep_fields_map_.begin(); it != dep_fields_map_.end(); ++it) {
    this->addDependentField(*(it->second));
  }

  // optionally deal with integration point locations
  if (model_->getIntegrationPointLocationFlag()) {
    coord_vec_ = decltype(coord_vec_)("Coord Vec", dl->qp_vector);
    this->addDependentField(coord_vec_);
  }

  // optionally deal with temperature
  if (p.isType<std::string>("Temperature Name")) {
    have_temperature_ = true;
    temperature_      = decltype(temperature_)(
        p.get<std::string>("Temperature Name"), dl->qp_scalar);
    this->addDependentField(temperature_);
  }

  // optionally deal with damage
  if (p.isType<std::string>("Damage Name")) {
    have_damage_ = true;
    damage_ =
        decltype(damage_)(p.get<std::string>("Damage Name"), dl->qp_scalar);
    this->addDependentField(damage_);
  }

  // optionally deal with total concentration
  if (p.isType<std::string>("Total Concentration Name")) {
    have_total_concentration_ = true;
    total_concentration_      = decltype(total_concentration_)(
        p.get<std::string>("Total Concentration Name"), dl->qp_scalar);
    this->addDependentField(total_concentration_);
  }

  // optionally deal with total bubble density
  if (p.isType<std::string>("Total Bubble Density Name")) {
    have_total_bubble_density_ = true;
    total_bubble_density_      = decltype(total_bubble_density_)(
        p.get<std::string>("Total Bubble Density Name"), dl->qp_scalar);
    this->addDependentField(total_bubble_density_);
  }

  // optionally deal with bubble volume fraction
  if (p.isType<std::string>("Bubble Volume Fraction Name")) {
    have_bubble_volume_fraction_ = true;
    bubble_volume_fraction_      = decltype(bubble_volume_fraction_)(
        p.get<std::string>("Bubble Volume Fraction Name"), dl->qp_scalar);
    this->addDependentField(bubble_volume_fraction_);
  }

  // optional volume averaging needs integration weights and J
  if (volume_average_pressure_) {
    weights_ =
        decltype(weights_)(p.get<std::string>("Weights Name"), dl->qp_scalar);
    this->addDependentField(weights_);

    j_ = decltype(j_)(p.get<std::string>("J Name"), dl->qp_scalar);
    this->addDependentField(j_);
  }

  // construct the evaluated fields
  auto eval_map = model_->getEvaluatedFieldMap();
  for (auto& pair : eval_map) {
    auto temp_field =
        Teuchos::rcp(new PHX::MDField<ScalarT>(pair.first, pair.second));
    eval_fields_map_.insert(std::make_pair(pair.first, temp_field));
  }

  // register evaluated fields
  for (auto& pair : eval_fields_map_) {
    this->addEvaluatedField(*(pair.second));
  }

  this->setName("ConstitutiveModelInterface" + PHX::typeAsString<EvalT>());
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelInterface<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      dep_fields_map_.size() == 0,
      std::logic_error,
      "something is wrong in the LCM::CMI");
  TEUCHOS_TEST_FOR_EXCEPTION(
      eval_fields_map_.size() == 0,
      std::logic_error,
      "something is wrong in the LCM::CMI");
  // dependent fields
  typename decltype(dep_fields_map_)::iterator it;
  for (it = dep_fields_map_.begin(); it != dep_fields_map_.end(); ++it) {
    this->utils.setFieldData(*(it->second), fm);
  }

  // optionally deal with integration point locations
  if (model_->getIntegrationPointLocationFlag()) {
    this->utils.setFieldData(coord_vec_, fm);
    model_->setCoordVecField(coord_vec_);
  }

  // optionally deal with temperature
  if (have_temperature_) {
    this->utils.setFieldData(temperature_, fm);
    model_->setTemperatureField(temperature_);
  }

  // optionally deal with damage
  if (have_damage_) {
    this->utils.setFieldData(damage_, fm);
    model_->setDamageField(damage_);
  }

  // optionally deal with total concentration
  if (have_total_concentration_) {
    this->utils.setFieldData(total_concentration_, fm);
    model_->setTotalConcentrationField(total_concentration_);
  }

  // optionally deal with total bubble density
  if (have_total_bubble_density_) {
    this->utils.setFieldData(total_bubble_density_, fm);
    model_->setTotalBubbleDensityField(total_bubble_density_);
  }

  // optionally deal with bubble volume fraction
  if (have_bubble_volume_fraction_) {
    this->utils.setFieldData(bubble_volume_fraction_, fm);
    model_->setBubbleVolumeFractionField(bubble_volume_fraction_);
  }

  // optionally deal with damage
  if (have_damage_) {
    this->utils.setFieldData(damage_, fm);
    model_->setDamageField(damage_);
  }

  // optionally deal with damage
  if (have_damage_) {
    this->utils.setFieldData(damage_, fm);
    model_->setDamageField(damage_);
  }

  // optionally deal with volume averaging
  if (volume_average_pressure_) {
    this->utils.setFieldData(weights_, fm);
    model_->setWeightsField(weights_);
    this->utils.setFieldData(j_, fm);
    model_->setJField(j_);
  }

  // evaluated fields
  for (auto& pair : eval_fields_map_) {
    this->utils.setFieldData(*(pair.second), fm);
  }
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelInterface<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  model_->computeState(workset, dep_fields_map_, eval_fields_map_);
  if (volume_average_pressure_) {
    model_->computeVolumeAverage(workset, dep_fields_map_, eval_fields_map_);
  }
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelInterface<EvalT, Traits>::fillStateVariableStruct(
    int state_var)
{
  sv_struct_.name               = model_->getStateVarName(state_var);
  sv_struct_.data_layout        = model_->getStateVarLayout(state_var);
  sv_struct_.init_type          = model_->getStateVarInitType(state_var);
  sv_struct_.init_value         = model_->getStateVarInitValue(state_var);
  sv_struct_.register_old_state = model_->getStateVarOldStateFlag(state_var);
  sv_struct_.output_to_exodus   = model_->getStateVarOutputFlag(state_var);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelInterface<EvalT, Traits>::initializeModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string model_name =
      p->sublist("Material Model").get<std::string>("Model Name");

  std::string const error_msg = "Undefined material model name";

  Teuchos::RCP<ConstitutiveModel<EvalT, Traits>> model = Teuchos::null;

  using Teuchos::rcp;

  if (model_name == "") {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, error_msg);
  } else if (model_name == "Neohookean") {
    model = rcp(new NeohookeanModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Parallel Neohookean") {
    model = rcp(new ParallelNeohookeanModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Creep") {
    model = rcp(new CreepModel<EvalT, Traits>(p, dl));
  } else if (model_name == "J2") {
    model = rcp(new J2Model<EvalT, Traits>(p, dl));
  } else if (model_name == "Newtonian Fluid") {
    model = rcp(new NewtonianFluidModel<EvalT, Traits>(p, dl));
  } else if (model_name == "CrystalPlasticity") {
    model = rcp(new CrystalPlasticityModel<EvalT, Traits>(p, dl));
  } else if (model_name == "ElasticCrystal") {
    model = rcp(new ElasticCrystalModel<EvalT, Traits>(p, dl));
  } else if (model_name == "ViscoElastic") {
    model = rcp(new ViscoElasticModel<EvalT, Traits>(p, dl));
  } else if (model_name == "AHD") {
    model = rcp(new AnisotropicHyperelasticDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Gurson") {
    model = rcp(new GursonModel<EvalT, Traits>(p, dl));
  } else if (model_name == "GursonHMR") {
    model = rcp(new GursonHMRModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Mooney Rivlin") {
    model = rcp(new MooneyRivlinModel<EvalT, Traits>(p, dl));
  } else if (model_name == "RIHMR") {
    model = rcp(new RIHMRModel<EvalT, Traits>(p, dl));
  } else if (model_name == "J2Fiber") {
    model = rcp(new J2FiberModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Anisotropic Damage") {
    model = rcp(new AnisotropicDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Elastic Damage") {
    model = rcp(new ElasticDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Saint Venant Kirchhoff") {
    model = rcp(new StVenantKirchhoffModel<EvalT, Traits>(p, dl));
  } else if (model_name == "AAA") {
    model = rcp(new AAAModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Linear Elastic") {
    model = rcp(new LinearElasticModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Hyperelastic Damage") {
    model = rcp(new HyperelasticDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Cap Explicit") {
    model = rcp(new CapExplicitModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Cap Implicit") {
    model = rcp(new CapImplicitModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Drucker Prager") {
    model = rcp(new DruckerPragerModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Tvergaard Hutchinson") {
    model = rcp(new TvergaardHutchinsonModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Viscoplastic") {
    model = rcp(new AnisotropicViscoplasticModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Linear HMC") {
    model = rcp(new LinearHMCModel<EvalT, Traits>(p, dl));
  } else if (model_name == "J2 HMC") {
    model = rcp(new J2HMCModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Linear Piezoelectric") {
    model = rcp(new LinearPiezoModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Ferroic") {
    model = rcp(new FerroicDriver<EvalT, Traits>(p, dl));
  } else if (model_name == "Ortiz Pandolfi") {
    model = rcp(new OrtizPandolfiModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Elasto Viscoplastic") {
    model = rcp(new ElastoViscoplasticModel<EvalT, Traits>(p, dl));
  } else if (model_name == "J2 MiniSolver") {
    model = rcp(new J2MiniSolver<EvalT, Traits>(p, dl));
  } else if (model_name == "ACE ice") {
    model = rcp(new ACEice<EvalT, Traits>(p, dl));
  } else if (model_name == "ACE permafrost") {
    model = rcp(new ACEpermafrost<EvalT, Traits>(p, dl));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, error_msg);
  }

  this->model_ = model;
}

//------------------------------------------------------------------------------
}  // namespace LCM
