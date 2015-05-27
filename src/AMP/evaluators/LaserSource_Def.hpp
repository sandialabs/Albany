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

  this->setName("LaserSource"+PHX::typeAsString<EvalT>());
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
  ScalarT pi = 3.1415926535897932;
  ScalarT LaserFlux_Max =(3.0/(pi*laser_beam_radius*laser_beam_radius))*laser_power;
  ScalarT beta = 1.5*(1.0 - porosity)/(porosity*particle_dia);

//  Parameters for the depth profile of the laser heat source:
  ScalarT lambda = 2.50;
  ScalarT a = sqrt(1.0 - powder_hemispherical_reflectivity);
  ScalarT A = (1.0 - pow(powder_hemispherical_reflectivity,2))*exp(-lambda);
  ScalarT B = 3.0 + powder_hemispherical_reflectivity*exp(-2*lambda);
  ScalarT b1 = 1 - a;
  ScalarT b2 = 1 + a;
  ScalarT c1 = b2 - powder_hemispherical_reflectivity*b1;
  ScalarT c2 = b1 - powder_hemispherical_reflectivity*b2;
  ScalarT C = b1*c2*exp(-2*a*lambda) - b2*c1*exp(2*a*lambda);
//  Following are few factors defined by the coder to be included while defining the depth profile
  ScalarT f1 = 1.0/(3.0 - 4.0*powder_hemispherical_reflectivity);
  ScalarT f2 = 2*powder_hemispherical_reflectivity*a*a/C;
  ScalarT f3 = 3.0*(1.0 - powder_hemispherical_reflectivity);
// -----------------------------------------------------------------------------------------------
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
	  MeshScalarT X = coord_(cell,qp,0);
	  MeshScalarT Y = coord_(cell,qp,1);
	  MeshScalarT Z = coord_(cell,qp,2);

//  Note:(0.0003 -Y) is because of the Y axis for the depth_profile is in the negative direction as per the Gusarov's equation.
    ScalarT depth_profile = f1*(f2*(A*(b2*exp(2.0*a*beta*(0.0003-Y))-b1*exp(-2.0*a*beta*(0.0003-Y))) - B*(c2*exp(-2.0*a*(lambda - beta*(0.0003-Y)))-c1*exp(2.0*a*(lambda-beta*(0.0003-Y))))) + f3*(exp(-beta*(0.0003-Y))+powder_hemispherical_reflectivity*exp(beta*(0.0003-Y) - 2.0*lambda)));
    MeshScalarT* XX = &coord_(cell,qp,0);
    ScalarT radius = sqrt((X - 0.00)*(X - 0.00) + (Z - 0.0)*(Z - 0.0));
     if (radius < laser_beam_radius && beta*(0.0003-Y) <= lambda)
	    laser_source_(cell,qp) =beta*LaserFlux_Max*pow((1.0-(radius*radius)/(laser_beam_radius*laser_beam_radius)),2)*depth_profile;
     else laser_source_(cell,qp) =0.0;
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
