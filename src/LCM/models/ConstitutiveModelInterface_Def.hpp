//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

#include "AnisotropicDamageModel.hpp"
#include "AnisotropicHyperelasticDamageModel.hpp"
#include "ElasticDamageModel.hpp"
#include "GursonHMRModel.hpp"
#include "GursonModel.hpp"
#include "J2FiberModel.hpp"
#include "J2Model.hpp"
#include "MooneyRivlinModel.hpp"
#include "NeohookeanModel.hpp"
#include "RIHMRModel.hpp"
#include "StVenantKirchhoffModel.hpp"
#include "AAAModel.hpp"
#include "LinearElasticModel.hpp"
#include "HyperelasticDamageModel.hpp"
#include "CapExplicitModel.hpp"
#include "CapImplicitModel.hpp"
#include "DruckerPragerModel.hpp"
#include "CrystalPlasticityModel.hpp"
#include "TvergaardHutchinsonModel.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ConstitutiveModelInterface<EvalT, Traits>::
ConstitutiveModelInterface(Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl):
  have_temperature_(false),
  have_damage_(false),
  volume_average_pressure_(p.get<bool>("Volume Average Pressure", false))
{
  this->initializeModel(p.get<Teuchos::ParameterList*>("Material Parameters"),
      dl);

  // construct the dependent fields
  std::map<std::string, Teuchos::RCP<PHX::DataLayout> >
  dependent_map = model_->getDependentFieldMap();
  typename std::map<std::string, Teuchos::RCP<PHX::DataLayout> >::iterator miter;
  for (miter = dependent_map.begin();
      miter != dependent_map.end();
      ++miter) {
    Teuchos::RCP<PHX::MDField<ScalarT> > temp_field =
        Teuchos::rcp(new PHX::MDField<ScalarT>(miter->first, miter->second));
    dep_fields_map_.insert(std::make_pair(miter->first, temp_field));
  }

  // register dependent fields
  typename std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > >::iterator it;
  for (it = dep_fields_map_.begin();
      it != dep_fields_map_.end();
      ++it) {
    this->addDependentField(*(it->second));
  }

  // optionally deal with integration point locations
  if (model_->getIntegrationPointLocationFlag()) {
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> cv("Coord Vec",
        dl->qp_vector);
    coord_vec_ = cv;
    this->addDependentField(coord_vec_);
  }

  // optionally deal with temperature
  if (p.isType<std::string>("Temperature Name")) {
    have_temperature_ = true;
    PHX::MDField<ScalarT, Cell, QuadPoint> t(p.get<std::string>("Temperature Name"),
        dl->qp_scalar);
    temperature_ = t;
    this->addDependentField(temperature_);
  }

  // optionally deal with damage
  if (p.isType<std::string>("Damage Name")) {
    have_damage_ = true;
    PHX::MDField<ScalarT, Cell, QuadPoint> d(p.get<std::string>("Damage Name"),
        dl->qp_scalar);
    damage_ = d;
    this->addDependentField(damage_);
  }

  // optional volume averaging needs integration weights
  if (volume_average_pressure_) {
    PHX::MDField<ScalarT, Cell, QuadPoint> w(p.get<std::string>("Weights Name"),
        dl->qp_scalar);
    weights_ = w;
    this->addDependentField(damage_);
  }

  // construct the evaluated fields
  std::map<std::string, Teuchos::RCP<PHX::DataLayout> >
  eval_map = model_->getEvaluatedFieldMap();
  for (miter = eval_map.begin();
      miter != eval_map.end();
      ++miter) {
    Teuchos::RCP<PHX::MDField<ScalarT> > temp_field =
        Teuchos::rcp(new PHX::MDField<ScalarT>(miter->first, miter->second));
    eval_fields_map_.insert(std::make_pair(miter->first, temp_field));
  }

  // register evaluated fields
  for (it = eval_fields_map_.begin();
      it != eval_fields_map_.end();
      ++it) {
    this->addEvaluatedField(*(it->second));
  }

  this->setName("ConstitutiveModelInterface" + PHX::TypeString<EvalT>::value);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelInterface<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(dep_fields_map_.size() == 0, std::logic_error,
      "something is wrong in the LCM::CMI");
  TEUCHOS_TEST_FOR_EXCEPTION(eval_fields_map_.size() == 0, std::logic_error,
      "something is wrong in the LCM::CMI");
  // dependent fields
  typename std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > >::iterator it;
  for (it = dep_fields_map_.begin();
      it != dep_fields_map_.end();
      ++it) {
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

  // evaluated fields
  for (it = eval_fields_map_.begin();
      it != eval_fields_map_.end();
      ++it) {
    this->utils.setFieldData(*(it->second), fm);
  }
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelInterface<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  model_->computeState(workset, dep_fields_map_, eval_fields_map_);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelInterface<EvalT, Traits>::
fillStateVariableStruct(int state_var)
{
  sv_struct_.name = model_->getStateVarName(state_var);
  sv_struct_.data_layout = model_->getStateVarLayout(state_var);
  sv_struct_.init_type = model_->getStateVarInitType(state_var);
  sv_struct_.init_value = model_->getStateVarInitValue(state_var);
  sv_struct_.register_old_state = model_->getStateVarOldStateFlag(state_var);
  sv_struct_.output_to_exodus = model_->getStateVarOutputFlag(state_var);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelInterface<EvalT, Traits>::
initializeModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string model_name =
      p->sublist("Material Model").get<std::string>("Model Name");

  if (model_name == "Neohookean") {
    this->model_ = Teuchos::rcp(new LCM::NeohookeanModel<EvalT, Traits>(p, dl));
  } else if (model_name == "J2") {
    this->model_ = Teuchos::rcp(new LCM::J2Model<EvalT, Traits>(p, dl));
  } else if (model_name == "CrystalPlasticity") {
    this->model_ = Teuchos::rcp(new LCM::CrystalPlasticityModel<EvalT, Traits>(p, dl));
  } else if (model_name == "AHD") {
    this->model_ = Teuchos::rcp(
        new LCM::AnisotropicHyperelasticDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Gurson") {
    this->model_ = Teuchos::rcp(new LCM::GursonModel<EvalT, Traits>(p, dl));
  } else if (model_name == "GursonHMR") {
    this->model_ = Teuchos::rcp(new LCM::GursonHMRModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Mooney Rivlin") {
    this->model_ = Teuchos::rcp(
        new LCM::MooneyRivlinModel<EvalT, Traits>(p, dl));
  } else if (model_name == "RIHMR") {
    this->model_ = Teuchos::rcp(new LCM::RIHMRModel<EvalT, Traits>(p, dl));
  } else if (model_name == "J2Fiber") {
    this->model_ = Teuchos::rcp(new LCM::J2FiberModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Anisotropic Damage") {
    this->model_ = Teuchos::rcp(
        new LCM::AnisotropicDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Elastic Damage") {
    this->model_ = Teuchos::rcp(
        new LCM::ElasticDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Saint Venant Kirchhoff") {
    this->model_ = Teuchos::rcp(
        new LCM::StVenantKirchhoffModel<EvalT, Traits>(p, dl));
  } else if (model_name == "AAA") {
    this->model_ = Teuchos::rcp(new LCM::AAAModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Linear Elastic") {
    this->model_ = Teuchos::rcp(
        new LCM::LinearElasticModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Hyperelastic Damage") {
    this->model_ = Teuchos::rcp(
        new LCM::HyperelasticDamageModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Cap Explicit") {
    this->model_ = Teuchos::rcp(
        new LCM::CapExplicitModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Cap Implicit") {
      this->model_ = Teuchos::rcp(
        new LCM::CapImplicitModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Drucker Prager") {
      this->model_ = Teuchos::rcp(
        new LCM::DruckerPragerModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Tvergaard Hutchinson") {
      this->model_ = Teuchos::rcp(
        new LCM::TvergaardHutchinsonModel<EvalT, Traits>(p, dl));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,
        std::logic_error,
        "Undefined material model name");
  }
}

//------------------------------------------------------------------------------
}

