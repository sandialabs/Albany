/*
 * LandIce_BasalMeltRate_Def.hpp
 *
 *  Created on: Jun 16, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace LandIce
{


  template<typename EvalT, typename Traits, typename VelocityType>
  BasalMeltRate<EvalT,Traits,VelocityType>::
  BasalMeltRate(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl_basal):
  phi               (p.get<std::string> ("Water Content Side Variable Name"),dl_basal->node_scalar),
  geoFluxHeat       (p.get<std::string> ("Geothermal Flux Side Variable Name"),dl_basal->node_scalar),
  velocity          (p.get<std::string> ("Velocity Side Variable Name"),dl_basal->node_vector),
  beta              (p.get<std::string> ("Basal Friction Coefficient Side Variable Name"),dl_basal->node_scalar),
  EnthalpyHs        (p.get<std::string> ("Enthalpy Hs Side Variable Name"),dl_basal->node_scalar),
  Enthalpy          (p.get<std::string> ("Enthalpy Side Variable Name"),dl_basal->node_scalar),
  basal_dTdz        (p.get<std::string> ("Basal dTdz Variable Name"),dl_basal->node_scalar),
  basalMeltRate     (p.get<std::string> ("Basal Melt Rate Variable Name"),dl_basal->node_scalar),
  basalVertVelocity (p.get<std::string> ("Basal Vertical Velocity Variable Name"),dl_basal->node_scalar),
  homotopy          (p.get<std::string> ("Continuation Parameter Name"),dl_basal->shared_param)
  {
    this->addDependentField(phi);
    this->addDependentField(geoFluxHeat);
    this->addDependentField(velocity);
    this->addDependentField(beta);
    this->addDependentField(EnthalpyHs);
    this->addDependentField(Enthalpy);
    this->addDependentField(homotopy);
    this->addEvaluatedField(basal_dTdz);

    this->addEvaluatedField(basalMeltRate);
    this->addEvaluatedField(basalVertVelocity);
    this->setName("Basal Melt Rate");

    std::vector<PHX::DataLayout::size_type> dims;
    dl_basal->node_qp_gradient->dimensions(dims);
    int numSides = dims[1];
    numSideNodes = dims[2];
    sideDim      = dims[4];
    numCellNodes = basalMeltRate.fieldTag().dataLayout().dimension(1);

    basalSideName = p.get<std::string> ("Side Set Name");

    Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
    rho_w = physics_list->get<double>("Water Density");//, 1000.0);
    rho_i = physics_list->get<double>("Ice Density");//, 910.0);
    L = physics_list->get<double>("Latent heat of fusion");//, 3e5);

    k_0 = physics_list->get<double>("Permeability factor");//, 0.0);
    k_i = physics_list->get<double>("Conductivity of ice");//, 1.0); //[W m^{-1} K^{-1}]
    eta_w = physics_list->get<double>("Viscosity of water");//, 0.0018);
    g = physics_list->get<double>("Gravity Acceleration");//, 9.8);
    alpha_om = physics_list->get<double>("Omega exponent alpha");//, 2.0);

    beta_p = physics_list->get<double>("Clausius-Clapeyron Coefficient");

    a = physics_list->get<double>("Diffusivity homotopy exponent");//, -9.0);

    Teuchos::ParameterList* regularization_list = p.get<Teuchos::ParameterList*>("LandIce Enthalpy Regularization");
    auto flux_reg_list = regularization_list->sublist("Enthalpy Flux Regularization", false);
    auto basalMelt_reg_list = regularization_list->sublist("Enthalpy Basal Melting Regularization", false);
    flux_reg_alpha = flux_reg_list.get<double>("alpha");
    flux_reg_beta = flux_reg_list.get<double>("beta");
    basalMelt_reg_alpha = basalMelt_reg_list.get<double>("alpha");
    basalMelt_reg_beta = basalMelt_reg_list.get<double>("beta");
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void BasalMeltRate<EvalT,Traits,VelocityType>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(phi,fm);
    this->utils.setFieldData(geoFluxHeat,fm);
    this->utils.setFieldData(velocity,fm);
    this->utils.setFieldData(beta,fm);
    this->utils.setFieldData(EnthalpyHs,fm);
    this->utils.setFieldData(Enthalpy,fm);
    this->utils.setFieldData(basal_dTdz,fm);
    this->utils.setFieldData(homotopy,fm);
    this->utils.setFieldData(basalMeltRate,fm);
    this->utils.setFieldData(basalVertVelocity,fm);
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void BasalMeltRate<EvalT,Traits,VelocityType>::
  evaluateFields(typename Traits::EvalData d)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (d.sideSets==Teuchos::null, std::runtime_error,
                                "Side sets defined in input file but not properly specified on the mesh.\n");
    int vecDimFO = 2;
    double pi = atan(1.) * 4.;
    ScalarT hom = homotopy(0);
    const double scyr (3.1536e7);  // [s/yr];
    ScalarT phiExp; // [adim]
    ScalarT basal_reg_coeff = basalMelt_reg_alpha*exp(basalMelt_reg_beta*hom); // [adim]
    ScalarT flux_reg_coeff = flux_reg_alpha*exp(flux_reg_beta*hom); // [adim]

    if (d.sideSets->find(basalSideName) != d.sideSets->end())
    {
      const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(basalSideName);
      for (auto const& it_side : sideSet)
      {
        // Get the local data of side and cell
        const int cell = it_side.elem_LID;
        const int side = it_side.side_local_id;

        for (int node = 0; node < numSideNodes; ++node)
        {
          bool isThereWater = false;//(beta(cell,side,node)<5.0);

          ScalarT diffEnthalpy = Enthalpy(cell,side,node) - EnthalpyHs(cell,side,node);
          ScalarT basal_reg_scale = (diffEnthalpy > 0 || !isThereWater) ?  ScalarT(0.5 - atan(basal_reg_coeff * diffEnthalpy)/pi) :
                                                                 ScalarT(0.5 - basal_reg_coeff * diffEnthalpy /pi);
          ScalarT flux_reg_scale = 1;//ScalarT(0.5 - atan(flux_reg_coeff * diffEnthalpy)/pi);

          ScalarT M = geoFluxHeat(cell,side,node);
          for (int dim = 0; dim < vecDimFO; dim++)
            M += 1000./scyr * beta(cell,side,node) * velocity(cell,side,node,dim) * velocity(cell,side,node,dim);

          double dTdz_melting = beta_p * rho_i * g;
          M += 1e-3* k_i * dTdz_melting;

          phiExp = pow(phi(cell,side,node),alpha_om);
          basalMeltRate(cell,side,node) =  -flux_reg_scale * basal_reg_scale *M + 1e-3*k_i*dTdz_melting;
          basalVertVelocity(cell,side,node) =  - scyr*(1-basal_reg_scale) * M / ((1 - rho_w/rho_i*std::min(phi(cell,side,node),0.5))*L*rho_w) -  scyr  *k_0 * (rho_w - rho_i) * g / eta_w * phiExp ;
        }
      }
    } 
  }
} //namespace LandIce


