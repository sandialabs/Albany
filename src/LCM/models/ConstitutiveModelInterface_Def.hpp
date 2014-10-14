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
#include "CreepModel.hpp"
#include "MooneyRivlinModel.hpp"
#include "NeohookeanModel.hpp"
#include "RIHMRModel.hpp"
#include "StVenantKirchhoffModel.hpp"
#include "AAAModel.hpp"
#include "LinearElasticModel.hpp"
#include "LinearHMCModel.hpp"
#include "HyperelasticDamageModel.hpp"
#include "CapExplicitModel.hpp"
#include "CapImplicitModel.hpp"
#include "DruckerPragerModel.hpp"
#include "CrystalPlasticityModel.hpp"
#include "TvergaardHutchinsonModel.hpp"
#include "AnisotropicViscoplasticModel.hpp"
#include "OrtizPandolfiModel.hpp"

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
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Material Parameters");
  plist->set<bool>("Volume Average Pressure", volume_average_pressure_);
  this->initializeModel(plist,dl);

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
    PHX::MDField<MeshScalarT, Cell, QuadPoint> w(p.get<std::string>("Weights Name"),
        dl->qp_scalar);
    weights_ = w;
    this->addDependentField(weights_);
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

  // optionally deal with volume averaging
  if (volume_average_pressure_) {
    this->utils.setFieldData(weights_, fm);
    model_->setWeightsField(weights_);
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
  if (volume_average_pressure_) {
    model_->computeVolumeAverage(workset,dep_fields_map_, eval_fields_map_);
  }
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
  std::string
  model_name = p->sublist("Material Model").get<std::string>("Model Name");

  std::string const
  error_msg = "Undefined material model name";

  Teuchos::RCP<ConstitutiveModel<EvalT, Traits> >
  model = Teuchos::null;

  using Teuchos::rcp;

  if (model_name == "Neohookean") {
    model = rcp(new NeohookeanModel<EvalT, Traits>(p, dl));
  } else if (model_name == "Creep") {
    model = rcp(new CreepModel<EvalT, Traits>(p, dl));
  } else if (model_name == "J2") {
    model = rcp(new J2Model<EvalT, Traits>(p, dl));
  } else if (model_name == "CrystalPlasticity") {
    model = rcp(new CrystalPlasticityModel<EvalT, Traits>(p, dl));
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
  } else if (model_name == "Ortiz Pandolfi") {
    model = rcp(new OrtizPandolfiModel<EvalT, Traits>(p, dl));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, error_msg);
  }

  this->model_ = model;
}

//------------------------------------------------------------------------------
}

