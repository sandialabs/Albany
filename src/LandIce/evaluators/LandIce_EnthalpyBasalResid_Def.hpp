/*
 * LandIce_enthalpyBasalResid_Def.hpp
 *
 *  Created on: May 31, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace LandIce
{
  template<typename EvalT, typename Traits, typename Type>
  EnthalpyBasalResid<EvalT,Traits,Type>::
  EnthalpyBasalResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  enthalpyBasalResid(p.get<std::string> ("Enthalpy Basal Residual Variable Name"), dl->node_scalar),
  homotopy    (p.get<std::string> ("Continuation Parameter Name"), dl->shared_param)
  {
    basalSideName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error, "Error! Basal side data layout not found.\n");

    Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

    BF         = decltype(BF)(p.get<std::string> ("BF Side Name"), dl_basal->node_qp_scalar);
    w_measure  = decltype(w_measure)(p.get<std::string> ("Weighted Measure Side Name"), dl_basal->qp_scalar);
    velocity   = decltype(velocity)(p.get<std::string> ("Velocity Side QP Variable Name"), dl_basal->qp_vector);
    beta       = decltype(beta)(p.get<std::string> ("Basal Friction Coefficient Side QP Variable Name"), dl_basal->qp_scalar);
    basal_dTdz = decltype(basal_dTdz)(p.get<std::string> ("Basal dTdz Side QP Variable Name"), dl_basal->qp_scalar);
    enthalpy   = decltype(enthalpy)(p.get<std::string> ("Enthalpy Side QP Variable Name"), dl_basal->qp_scalar);
    enthalpyHs = decltype(enthalpyHs)(p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl_basal->qp_scalar);
    diffEnth   = decltype(diffEnth)(p.get<std::string> ("Diff Enthalpy Variable Name"), dl->node_scalar);
    basalMeltRateQP = decltype(basalMeltRateQP)(p.get<std::string> ("Basal Melt Rate Side QP Variable Name"), dl_basal->qp_scalar);
    basalMeltRate = decltype(basalMeltRate)(p.get<std::string> ("Basal Melt Rate Side Variable Name"), dl_basal->node_scalar);
    phi        = decltype(phi)(p.get<std::string> ("Water Content Side QP Variable Name"),dl_basal->node_scalar);

    geoFlux   = decltype(geoFlux)(p.get<std::string> ("Geothermal Flux Side QP Variable Name"), dl_basal->qp_scalar);

    haveSUPG = p.isParameter("LandIce Enthalpy Stabilization") ? (p.get<Teuchos::ParameterList*>("LandIce Enthalpy Stabilization")->get<std::string>("Type") == "SUPG") : false;

    this->addDependentField(BF);
    this->addDependentField(w_measure);
    this->addDependentField(geoFlux);
    this->addDependentField(velocity);
    this->addDependentField(beta);
    this->addDependentField(basal_dTdz);
    this->addDependentField(enthalpy);
    this->addDependentField(enthalpyHs);
    this->addDependentField(diffEnth);
    this->addDependentField(homotopy);
    this->addDependentField(phi);
    this->addDependentField(basalMeltRateQP);

    this->addEvaluatedField(enthalpyBasalResid);
    this->addEvaluatedField(basalMeltRate);
 //   this->addEvaluatedField(basalMeltRateQP);
    this->setName("Enthalpy Basal Residual");

    if (haveSUPG)
    {
      GradBF         = decltype(GradBF)(p.get<std::string> ("Gradient BF Side Name"), dl_basal->node_qp_gradient);
      verticalVel    = decltype(verticalVel)(p.get<std::string>("Vertical Velocity Side QP Variable Name"), dl_basal->qp_scalar);

      this->addDependentField(verticalVel);
      this->addDependentField(GradBF);

      this->setName("Enthalpy Basal Residual SUPG");
    }

    std::vector<PHX::DataLayout::size_type> dims;
    dl_basal->node_qp_gradient->dimensions(dims);
    int numSides = dims[1];
    numSideNodes = dims[2];
    numSideQPs   = dims[3];
    numCellNodes = enthalpyBasalResid.fieldTag().dataLayout().dimension(1);

    dl->node_vector->dimensions(dims);
    vecDimFO     = std::min((int)dims[2],2);


    Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
    a = physics_list->get<double>("Diffusivity homotopy exponent");
    beta_p = physics_list->get<double>("Clausius-Clapeyron Coefficient");
    rho_i = physics_list->get<double>("Ice Density");
    rho_w = physics_list->get<double>("Water Density"); //, 1000.0);
    g     = physics_list->get<double>("Gravity Acceleration");
    L = physics_list->get<double>("Latent heat of fusion"); //, 3e5);
    k_0 = physics_list->get<double>("Permeability factor"); //, 0.0);
    k_i = physics_list->get<double>("Conductivity of ice"); //[W m^{-1} K^{-1}]
    eta_w = physics_list->get<double>("Viscosity of water"); //, 0.0018);
    alpha_om = physics_list->get<double>("Omega exponent alpha"); //, 2.0);

    // Index of the nodes on the sides in the numeration of the cell
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
    sideDim      = cellType->getDimension()-1;
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      sideNodes[side].resize(thisSideNodes);
      for (int node=0; node<thisSideNodes; ++node)
      {
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
      }
    }
  }

  template<typename EvalT, typename Traits, typename Type>
  void EnthalpyBasalResid<EvalT,Traits,Type>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(BF,fm);
    this->utils.setFieldData(w_measure,fm);
    this->utils.setFieldData(geoFlux,fm);
    this->utils.setFieldData(velocity,fm);
    this->utils.setFieldData(beta,fm);
    this->utils.setFieldData(basal_dTdz,fm);
    this->utils.setFieldData(enthalpy,fm);
    this->utils.setFieldData(enthalpyHs,fm);
    this->utils.setFieldData(diffEnth,fm);
    this->utils.setFieldData(homotopy,fm);

    this->utils.setFieldData(enthalpyBasalResid,fm);
    this->utils.setFieldData(basalMeltRate,fm);
    this->utils.setFieldData(basalMeltRateQP,fm);

    if (haveSUPG)
    {
      this->utils.setFieldData(verticalVel,fm);
      this->utils.setFieldData(GradBF,fm);
    }
  }

  template<typename EvalT, typename Traits, typename Type>
  void EnthalpyBasalResid<EvalT,Traits,Type>::
  evaluateFields(typename Traits::EvalData d)
  {
    // Zero out, to avoid leaving stuff from previous workset!
    for (int cell = 0; cell < d.numCells; ++cell)
      for (int node = 0; node < numCellNodes; ++node)
        enthalpyBasalResid(cell,node) = 0.;

    if (d.sideSets->find(basalSideName)==d.sideSets->end())
      return;

    const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(basalSideName);

    const double scyr (3.1536e7);  // [s/yr];
    double pi = atan(1.) * 4.;
    ScalarT hom = homotopy(0);

    bool isThereWater = false;

    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int node = 0; node < numSideNodes; ++node)
      {
        int cnode = sideNodes[side][node];
        enthalpyBasalResid(cell,cnode) = 0.;
  //      isThereWater =(beta(cell,side,0)<5.0);

   //     ScalarT scale = (diffEnth(cell,cnode) > 0 || !isThereWater) ?  ScalarT(0.5 - atan(alpha * diffEnth(cell,cnode))/pi) :
  //                                                             ScalarT(0.5 - alpha * diffEnth(cell,cnode) /pi);

        for (int qp = 0; qp < numSideQPs; ++qp)
        {
//          ScalarT diffEnthalpy = enthalpy(cell,side,qp)-enthalpyHs(cell,side,qp);

    //      isThereWater =(beta(cell,side,qp)<5.0);

   //       ScalarT scale = (diffEnthalpy > 0 || !isThereWater) ?  ScalarT(0.5 - atan(alpha * diffEnthalpy)/pi) :
   //                                                              ScalarT(0.5 - alpha * diffEnthalpy /pi);
   //       ScalarT M = geoFlux(cell,side,qp);

   //       for (int dim = 0; dim < vecDimFO; ++dim)
   //         M += 1000/scyr * beta(cell,side,qp) * std::pow(velocity(cell,side,qp,dim),2);

   //       double dTdz_melting =  beta_p*rho_i*g;  //[K m^{-1}]
   //       M += 1e-3* k_i * dTdz_melting;

    //      ScalarT resid_tmp = -M*scale + 1e-3*k_i*dTdz_melting;
    //      enthalpyBasalResid(cell,cnode) += resid_tmp *  BF(cell,side,node,qp) * w_measure(cell,side,qp);

          enthalpyBasalResid(cell,cnode) += basalMeltRateQP(cell,side,qp) *  BF(cell,side,node,qp) * w_measure(cell,side,qp);
  //        ScalarT phiExp = pow(phi(cell,side,qp),alpha_om);
       //   basalMeltRate(cell,side,qp) = scyr*(1-scale) * M / ((1 - rho_w/rho_i*phi(cell,side,qp))*L*rho_w) +  scyr * k_0 * (rho_w - rho_i) * g / eta_w * phiExp ;
        }
      }
    }
  }
}
