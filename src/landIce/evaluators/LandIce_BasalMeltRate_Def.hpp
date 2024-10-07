/*
 * LandIce_BasalMeltRate_Def.hpp
 *
 *  Created on: Jun 16, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Albany_KokkosUtils.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "LandIce_BasalMeltRate.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename VelocityST>
BasalMeltRate<EvalT,Traits,VelocityST>::
BasalMeltRate(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl_basal)
 : phi               (p.get<std::string> ("Water Content Side Variable Name"),dl_basal->node_scalar)
 , beta              (p.get<std::string> ("Basal Friction Coefficient Side Variable Name"),dl_basal->node_scalar)
 , velocity          (p.get<std::string> ("Velocity Side Variable Name"),dl_basal->node_vector)
 , geoFluxHeat       (p.get<std::string> ("Geothermal Flux Side Variable Name"),dl_basal->node_scalar)
 , Enthalpy          (p.get<std::string> ("Enthalpy Side Variable Name"),dl_basal->node_scalar)
 , EnthalpyHs        (p.get<std::string> ("Enthalpy Hs Side Variable Name"),dl_basal->node_scalar)
 , enthalpyBasalFlux     (p.get<std::string> ("Basal Melt Rate Variable Name"), dl_basal->node_scalar)
 , basalVertVelocity (p.get<std::string> ("Basal Vertical Velocity Variable Name"),dl_basal->node_scalar)
{
  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> scalar_layout, vector_layout;
  if (nodal) {
    scalar_layout = dl_basal->node_scalar;
    vector_layout = dl_basal->node_vector;
  } else {
    scalar_layout = dl_basal->qp_scalar;
    vector_layout = dl_basal->qp_vector;
  }

  phi = decltype(phi)(p.get<std::string> ("Water Content Side Variable Name"),scalar_layout);
  beta = decltype(beta)(p.get<std::string> ("Basal Friction Coefficient Side Variable Name"),scalar_layout);
  velocity = decltype(velocity)(p.get<std::string> ("Velocity Side Variable Name"),vector_layout);
  geoFluxHeat = decltype(geoFluxHeat)(p.get<std::string> ("Geothermal Flux Side Variable Name"),scalar_layout);
  Enthalpy = decltype(Enthalpy)(p.get<std::string> ("Enthalpy Side Variable Name"),scalar_layout);
  EnthalpyHs = decltype(EnthalpyHs)(p.get<std::string> ("Enthalpy Hs Side Variable Name"),scalar_layout);
  enthalpyBasalFlux = decltype(enthalpyBasalFlux)(p.get<std::string> ("Basal Melt Rate Variable Name"),scalar_layout);
  basalVertVelocity = decltype(basalVertVelocity)(p.get<std::string> ("Basal Vertical Velocity Variable Name"),scalar_layout);

  //If true, the tangential velocity is the same as the horizontal velocity vector
  flat_approx = p.get<bool>("Flat Bed Approximation"); 
  if(!flat_approx) {
    normals = decltype(normals)(p.get<std::string> ("Side Normal Name"), dl_basal->qp_vector_spacedim);
  }

  this->addDependentField(phi);
  this->addDependentField(geoFluxHeat);
  this->addDependentField(velocity);
  this->addDependentField(beta);
  this->addDependentField(EnthalpyHs);
  this->addDependentField(Enthalpy);
  if(!flat_approx) {
    this->addDependentField(normals);
  }

  this->addEvaluatedField(enthalpyBasalFlux);
  this->addEvaluatedField(basalVertVelocity);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_basal->node_qp_gradient->dimensions(dims);
  numSideNodes = dims[1];
  numSideQPs   = dims[2];
  sideDim      = dims[3];
  numCellNodes = basalVertVelocity.fieldTag().dataLayout().extent(1);

  basalSideName = p.get<std::string> ("Side Set Name");

  Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_w = physics_list->get<double>("Water Density");//, 1000.0);
  rho_i = physics_list->get<double>("Ice Density");//, 910.0);
  L = physics_list->get<double>("Ice Latent Heat Of Fusion");//, 3e5);

  k_0 = physics_list->get<double>("Permeability factor");//, 0.0);
  k_i = physics_list->get<double>("Conductivity of ice");//, 1.0); //[W m^{-1} K^{-1}]
  eta_w = physics_list->get<double>("Viscosity of water");//, 0.0018);
  g = physics_list->get<double>("Gravity Acceleration");//, 9.8);
  alpha_om = physics_list->get<double>("Omega exponent alpha");//, 2.0);

  beta_p = physics_list->get<double>("Clausius-Clapeyron Coefficient");
  scyr = physics_list->get<double>("Seconds per Year");

  Teuchos::ParameterList* landice_list = p.get<Teuchos::ParameterList*>("LandIce Enthalpy");
  auto basalMelt_reg_list = landice_list->sublist("Regularization",false).sublist("Basal Melting Regularization", false);

  auto lubrication_list = landice_list->sublist("Bed Lubrication",false);
  if(lubrication_list.get<std::string>("Type") == "Dry")
    bed_lubrication = BED_LUBRICATION_TYPE::DRY;
  else if(lubrication_list.get<std::string>("Type") == "Wet")
    bed_lubrication = BED_LUBRICATION_TYPE::WET;
  else if(lubrication_list.get<std::string>("Type") == "Basal Friction Based")
    bed_lubrication = BED_LUBRICATION_TYPE::BASAL_FRICTION_BASED;
  else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Bed Lubrication Type not recognized.\n");

  if(bed_lubrication == BED_LUBRICATION_TYPE::BASAL_FRICTION_BASED)
    basal_friction_threshold = lubrication_list.get<double>("Basal Friction Threshold"); //wet if basal friction is small than threshold	  

  basalMelt_reg_alpha = basalMelt_reg_list.get<double>("alpha");
  basalMelt_reg_beta = basalMelt_reg_list.get<double>("beta");
  beta_scaling = 1000./scyr;

  this->setName("Basal Melt Rate" + PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename VelocityST>
void BasalMeltRate<EvalT,Traits,VelocityST>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename VelocityST>
KOKKOS_INLINE_FUNCTION
void BasalMeltRate<EvalT,Traits,VelocityST>::
operator() (const Basal_Melt_Rate_Tag&, const int& sideSet_idx) const {

  const unsigned int numPts = nodal ? numSideNodes : numSideQPs;

  for (unsigned int node = 0; node < numPts; ++node) {
    VelocityST lubrication; //coefficient that varies from zero (dry bed) to one (wet bed).
    switch(bed_lubrication) {
      case BED_LUBRICATION_TYPE::WET:
        lubrication = 1.0; break;
      case BED_LUBRICATION_TYPE::DRY:
        lubrication = (beta(sideSet_idx,node)>0.0) ? 0.0 : 1.0; break;
      case BED_LUBRICATION_TYPE::BASAL_FRICTION_BASED:
        lubrication = 1./(std::pow(beta(sideSet_idx,node)/basal_friction_threshold,basal_reg_coeff)+1.0); break;
    }

    ScalarT diffEnthalpy = Enthalpy(sideSet_idx,node) - EnthalpyHs(sideSet_idx,node);

    ScalarT basal_reg_scale = (diffEnthalpy > 0) ?  ScalarT(0.5 + 0.5*tanh(basalMelt_reg_alpha * diffEnthalpy)) :
                                                    ScalarT((1.0 - lubrication)*(0.5 + 0.5*tanh(basalMelt_reg_alpha * diffEnthalpy)) +
					            lubrication *(0.5 + 0.5* basalMelt_reg_alpha * diffEnthalpy));

    //mstar, [W m^{-2}] = [Pa m s^{-1}]: basal latent heat in temperate ice
    ScalarT mstar = geoFluxHeat(sideSet_idx,node);
    ScalarT tanVelSquared(0.0);
    for (unsigned int dim = 0; dim < vecDimFO; dim++)
      tanVelSquared += velocity(sideSet_idx,node,dim) * velocity(sideSet_idx,node,dim);
    if(!flat_approx) {
      ScalarT tmp(0.0);
      for (unsigned int dim = 0; dim < vecDimFO; dim++)
        tmp += -normals(sideSet_idx,node,dim)/normals(sideSet_idx,node,2)*velocity(sideSet_idx,node,dim);
      tanVelSquared += tmp*tmp;
    }

    mstar += beta_scaling * beta(sideSet_idx,node) * tanVelSquared;

    double dTdz_melting = beta_p * rho_i * g;
    mstar += k_i * dTdz_melting;

    enthalpyBasalFlux(sideSet_idx,node) =  (basal_reg_scale-1) *mstar + k_i*dTdz_melting;

    //ScalarT basal_water_flux = scyr * k_0 * (rho_w - rho_i) * g / eta_w * pow(phi(sideSet_idx,node),alpha_om); //[m yr^{-1}]
    ScalarT melting = scyr * basal_reg_scale * mstar / (L*rho_i); //[m yr^{-1}]
    basalVertVelocity(sideSet_idx,node) =  - melting /(1 - rho_w/rho_i*KU::min(phi(sideSet_idx,node),0.5));
  }

}

template<typename EvalT, typename Traits, typename VelocityST>
void BasalMeltRate<EvalT,Traits,VelocityST>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSetViews==Teuchos::null, std::runtime_error,
                              "Side set views defined in input file but not properly specified on the mesh.\n");
  if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  basal_reg_coeff = basalMelt_reg_beta*(1.0+std::log(1.0+basalMelt_reg_alpha)); // [adim]

  sideSet = workset.sideSetViews->at(basalSideName);

  Kokkos::parallel_for(Basal_Melt_Rate_Policy(0, sideSet.size), *this);
}

} //namespace LandIce
