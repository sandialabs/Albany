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
 : phi               (p.get<std::string> ("Water Content Side Variable Name"),dl_basal->node_scalar_sideset)
 , beta              (p.get<std::string> ("Basal Friction Coefficient Side Variable Name"),dl_basal->node_scalar_sideset)
 , velocity          (p.get<std::string> ("Velocity Side Variable Name"),dl_basal->node_vector_sideset)
 , geoFluxHeat       (p.get<std::string> ("Geothermal Flux Side Variable Name"),dl_basal->node_scalar_sideset)
 , Enthalpy          (p.get<std::string> ("Enthalpy Side Variable Name"),dl_basal->node_scalar_sideset)
 , EnthalpyHs        (p.get<std::string> ("Enthalpy Hs Side Variable Name"),dl_basal->node_scalar_sideset)
 , homotopy          (p.get<std::string> ("Continuation Parameter Name"),dl_basal->shared_param)
 , enthalpyBasalFlux     (p.get<std::string> ("Basal Melt Rate Variable Name"), dl_basal->node_scalar_sideset)
 , basalVertVelocity (p.get<std::string> ("Basal Vertical Velocity Variable Name"),dl_basal->node_scalar_sideset)
{
  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> scalar_layout, vector_layout;
  if (nodal) {
    scalar_layout = dl_basal->node_scalar_sideset;
    vector_layout = dl_basal->node_vector_sideset;
  } else {
    scalar_layout = dl_basal->qp_scalar_sideset;
    vector_layout = dl_basal->qp_vector_sideset;
  }

  phi = decltype(phi)(p.get<std::string> ("Water Content Side Variable Name"),scalar_layout);
  beta = decltype(beta)(p.get<std::string> ("Basal Friction Coefficient Side Variable Name"),scalar_layout);
  velocity = decltype(velocity)(p.get<std::string> ("Velocity Side Variable Name"),vector_layout);
  geoFluxHeat = decltype(geoFluxHeat)(p.get<std::string> ("Geothermal Flux Side Variable Name"),scalar_layout);
  Enthalpy = decltype(Enthalpy)(p.get<std::string> ("Enthalpy Side Variable Name"),scalar_layout);
  EnthalpyHs = decltype(EnthalpyHs)(p.get<std::string> ("Enthalpy Hs Side Variable Name"),scalar_layout);
  homotopy = decltype(homotopy)(p.get<std::string> ("Continuation Parameter Name"),dl_basal->shared_param);
  enthalpyBasalFlux = decltype(enthalpyBasalFlux)(p.get<std::string> ("Basal Melt Rate Variable Name"),scalar_layout);
  basalVertVelocity = decltype(basalVertVelocity)(p.get<std::string> ("Basal Vertical Velocity Variable Name"),scalar_layout);

  this->addDependentField(phi);
  this->addDependentField(geoFluxHeat);
  this->addDependentField(velocity);
  this->addDependentField(beta);
  this->addDependentField(EnthalpyHs);
  this->addDependentField(Enthalpy);
  this->addDependentField(homotopy);

  this->addEvaluatedField(enthalpyBasalFlux);
  this->addEvaluatedField(basalVertVelocity);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_basal->node_qp_gradient->dimensions(dims);
  numSideNodes = dims[2];
  numSideQPs   = dims[3];
  sideDim      = dims[4];
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

  a = physics_list->get<double>("Diffusivity homotopy exponent");//, -9.0);
  scyr = physics_list->get<double>("Seconds per Year");

  Teuchos::ParameterList* landice_list = p.get<Teuchos::ParameterList*>("LandIce Enthalpy");
  auto flux_reg_list = landice_list->sublist("Regularization",false).sublist("Flux Regularization", false);
  auto basalMelt_reg_list = landice_list->sublist("Regularization",false).sublist("Basal Melting Regularization", false);
  flux_reg_alpha = flux_reg_list.get<double>("alpha");
  flux_reg_beta = flux_reg_list.get<double>("beta");

  isThereWater = (landice_list->get<std::string>("Bed Lubrication") == "Wet") ? true : false;

  basalMelt_reg_alpha = basalMelt_reg_list.get<double>("alpha");
  basalMelt_reg_beta = basalMelt_reg_list.get<double>("beta");
  beta_scaling = 1000./scyr;

  this->setName("Basal Melt Rate" + PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename VelocityST>
void BasalMeltRate<EvalT,Traits,VelocityST>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename VelocityST>
KOKKOS_INLINE_FUNCTION
void BasalMeltRate<EvalT,Traits,VelocityST>::
operator() (const Basal_Melt_Rate_Tag& tag, const int& sideSet_idx) const {

  const unsigned int numPts = nodal ? numSideNodes : numSideQPs;

  for (unsigned int node = 0; node < numPts; ++node) {
    //always in presence of water on shelves (assuming that beta==0 <==> on shelves)
    bool isThereWaterHere = isThereWater || (beta(sideSet_idx,node) == 0.0);
    ScalarT diffEnthalpy = Enthalpy(sideSet_idx,node) - EnthalpyHs(sideSet_idx,node);
    ScalarT basal_reg_scale = (diffEnthalpy > 0 || !isThereWaterHere) ?  ScalarT(0.5 + 0.5*tanh(basal_reg_coeff * diffEnthalpy)) :
                                                                        ScalarT(0.5 + 0.5* basal_reg_coeff * diffEnthalpy);
                                                                  //    ScalarT(0.5 + 0.5* (0.5-0.5*std::pow(1-basal_reg_coeff * diffEnthalpy,2)));

    //mstar, [W m^{-2}] = [Pa m s^{-1}]: basal latent heat in temperate ice
    ScalarT mstar = geoFluxHeat(sideSet_idx,node);
    for (unsigned int dim = 0; dim < vecDimFO; dim++)
      mstar += beta_scaling * beta(sideSet_idx,node) * velocity(sideSet_idx,node,dim) * velocity(sideSet_idx,node,dim);

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

  hom = homotopy(0);
  basal_reg_coeff = basalMelt_reg_alpha*exp(basalMelt_reg_beta*hom); // [adim]
  flux_reg_coeff = flux_reg_alpha*exp(flux_reg_beta*hom); // [adim]

  sideSet = workset.sideSetViews->at(basalSideName);

  Kokkos::parallel_for(Basal_Melt_Rate_Policy(0, sideSet.size), *this);
}

} //namespace LandIce
