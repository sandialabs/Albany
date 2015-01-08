//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <fstream>
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace AMP {

//**********************************************************************
template<typename EvalT, typename Traits>
LaserSource<EvalT, Traits>::
LaserSource(Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  coord_        (p.get<std::string>("Coordinate Name"),
                 dl->qp_vector),
  laser_source_ (p.get<std::string>("Laser Source Name"),
                 dl->qp_scalar)
{

  this->addDependentField(coord_);
  this->addEvaluatedField(laser_source_);
 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidLaserSourceParameters();

  cond_list->validateParameters(*reflist, 0,
      Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  // dummy variable used multiple times below
  std::string type; 
  
  type = cond_list->get("Laser Beam Radius Type", "Constant");
  ScalarT value_laser_beam_radius = cond_list->get("Laser Beam Radius Value", 1.0);
  init_constant_laser_beam_radius(value_laser_beam_radius,p);

  type = cond_list->get("Laser Power Type", "Constant");
  ScalarT value_laser_power = cond_list->get("Laser Power Value", 1.0);
  init_constant_laser_power(value_laser_power,p);

  type = cond_list->get("Porosity Type", "Constant");
  ScalarT value_porosity = cond_list->get("Porosity Value", 1.0);
  init_constant_porosity(value_porosity,p);

  type = cond_list->get("Particle Dia Type", "Constant");
  ScalarT value_particle_dia = cond_list->get("Particle Dia Value", 1.0);
  init_constant_particle_dia(value_particle_dia,p);

  type = cond_list->get("Powder Hemispherical Reflectivity Type", "Constant");
  ScalarT value_powder_hemispherical_reflectivity = cond_list->get("Powder Hemispherical Reflectivity Value", 1.0);
  init_constant_powder_hemispherical_reflectivity(value_powder_hemispherical_reflectivity,p);

  this->setName("LaserSource"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LaserSource<EvalT, Traits>::
init_constant_porosity(ScalarT value_porosity, Teuchos::ParameterList& p){
  porosity = value_porosity;
}

template<typename EvalT, typename Traits>
void LaserSource<EvalT, Traits>::
init_constant_particle_dia(ScalarT value_particle_dia, Teuchos::ParameterList& p){
  particle_dia = value_particle_dia;
}

template<typename EvalT, typename Traits>
void LaserSource<EvalT, Traits>::
init_constant_laser_beam_radius(ScalarT value_laser_beam_radius, Teuchos::ParameterList& p){
  laser_beam_radius = value_laser_beam_radius;
}

template<typename EvalT, typename Traits>
void LaserSource<EvalT, Traits>::
init_constant_laser_power(ScalarT value_laser_power, Teuchos::ParameterList& p){
  laser_power = value_laser_power;
}

template<typename EvalT, typename Traits>
void LaserSource<EvalT, Traits>::
init_constant_powder_hemispherical_reflectivity(ScalarT value_powder_hemispherical_reflectivity, Teuchos::ParameterList& p){
  powder_hemispherical_reflectivity = value_powder_hemispherical_reflectivity;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LaserSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coord_,fm);
  this->utils.setFieldData(laser_source_,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LaserSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // current time
  const RealType time = workset.current_time;

  // source function
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      MeshScalarT* X = &coord_(cell,qp,0);
      laser_source_(cell,qp) = laser_power + particle_dia;
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
LaserSource<EvalT, Traits>::
getValidLaserSourceParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Laser Source Params"));;
 
  valid_pl->set<std::string>("Laser Beam Radius Type", "Constant");
  valid_pl->set<double>("Laser Beam Radius Value", 1.0);

  valid_pl->set<std::string>("Laser Power Type", "Constant");
  valid_pl->set<double>("Laser Power Value", 1.0);

  valid_pl->set<std::string>("Porosity Type", "Constant");
  valid_pl->set<double>("Porosity Value", 1.0);

  valid_pl->set<std::string>("Particle Dia Type", "Constant");
  valid_pl->set<double>("Particle Dia Value", 1.0);

  valid_pl->set<std::string>("Powder Hemispherical Reflectivity Type", "Constant");
  valid_pl->set<double>("Powder Hemispherical Reflectivity Value", 1.0);

  return valid_pl;
}
//**********************************************************************

}
