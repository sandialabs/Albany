/*
 * FELIX_enthalpyBasalResid_Def.hpp
 *
 *  Created on: May 31, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
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

    BF         = PHX::MDField<RealType,Cell,Side,Node,QuadPoint>(p.get<std::string> ("BF Side Name"), dl_basal->node_qp_scalar);
    w_measure  = PHX::MDField<MeshScalarT,Cell,Side,QuadPoint> (p.get<std::string> ("Weighted Measure Name"), dl_basal->qp_scalar);
    velocity   = PHX::MDField<Type,Cell,Side,QuadPoint,VecDim>(p.get<std::string> ("Velocity Side QP Variable Name"), dl_basal->qp_vector);
    beta       = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Basal Friction Coefficient Side QP Variable Name"), dl_basal->qp_scalar);
    basal_dTdz = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Basal dTdz Side QP Variable Name"), dl_basal->qp_scalar);
    enthalpy   = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Enthalpy Side QP Variable Name"), dl_basal->qp_scalar);
    enthalpyHs = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl_basal->qp_scalar);
    diffEnth   = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Diff Enthalpy Variable Name"), dl->node_scalar);

    geoFlux   = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Geothermal Flux Side QP Variable Name"), dl_basal->qp_scalar);

    haveSUPG = p.isParameter("FELIX Enthalpy Stabilization") ? (p.get<Teuchos::ParameterList*>("FELIX Enthalpy Stabilization")->get<std::string>("Type") == "SUPG") : false;

    this->addDependentField(BF.fieldTag());
    this->addDependentField(w_measure.fieldTag());
    this->addDependentField(geoFlux.fieldTag());
    this->addDependentField(velocity.fieldTag());
    this->addDependentField(beta.fieldTag());
    this->addDependentField(basal_dTdz.fieldTag());
    this->addDependentField(enthalpy.fieldTag());
    this->addDependentField(enthalpyHs.fieldTag());
    this->addDependentField(diffEnth.fieldTag());
    this->addDependentField(homotopy.fieldTag());

    this->addEvaluatedField(enthalpyBasalResid);
    this->setName("Enthalpy Basal Residual");

    if (haveSUPG)
    {
      enthalpyBasalResidSUPG  = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Enthalpy Basal Residual SUPG Variable Name"), dl->node_scalar);
      GradBF    		 = PHX::MDField<RealType,Cell,Side,Node,QuadPoint,Dim>(p.get<std::string> ("Gradient BF Side Name"), dl_basal->node_qp_gradient);
      verticalVel		 = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string>("Vertical Velocity Side QP Variable Name"), dl_basal->qp_scalar);

      this->addDependentField(velocity.fieldTag());
      this->addDependentField(verticalVel.fieldTag());
      this->addDependentField(GradBF.fieldTag());

      this->addEvaluatedField(enthalpyBasalResidSUPG);
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


    Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
    a = physics_list->get<double>("Diffusivity homotopy exponent");
    k_i = physics_list->get<double>("Conductivity of ice"); //[W m^{-1} K^{-1}]
    beta_p = physics_list->get<double>("Clausius-Clapeyron coefficient");
    rho_i = physics_list->get<double>("Ice Density");
    g     = physics_list->get<double>("Gravity Acceleration");

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

    if (haveSUPG)
    {
      this->utils.setFieldData(verticalVel,fm);
      this->utils.setFieldData(GradBF,fm);
      this->utils.setFieldData(enthalpyBasalResidSUPG,fm);
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

    ScalarT alpha;

    if (a == -2.0)
      alpha = pow(10.0, (a + hom*10)/8);
    else if (a == -1.0)
      alpha = pow(10.0, (a + hom*10)/4.5);
    else
      alpha = pow(10.0, a + hom*10/3);

    alpha = 1e-1*std::pow(10.0, 5*hom);

    ScalarT  robin_coeff=1e-5*std::pow(10.0, 10*hom);

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
        isThereWater =(beta(cell,side,0)<5.0);
        ScalarT scale = - atan(alpha * std::max(0.,diffEnth(cell,cnode))+
                                 alpha * double(!isThereWater)* std::min(0.,diffEnth(cell,cnode)))/pi + 0.5;

        for (int qp = 0; qp < numSideQPs; ++qp)
        {
         // isThereWater =(beta(cell,side,qp)<1.0);
          ScalarT diffEnthalpy = enthalpy(cell,side,node,qp)-enthalpyHs(cell,side,node,qp);
         // ScalarT scale = - atan(alpha * std::max(0.,diffEnthalpy)+
            //                     alpha * double(!isThereWater)* std::min(0.,diffEnthalpy))/pi + 0.5;
          ScalarT F = geoFlux(cell,side,qp);

          for (int dim = 0; dim < vecDimFO; ++dim)
            F += 1000/scyr * beta(cell,side,qp) * std::pow(velocity(cell,side,qp,dim),2);

          double dTdz_melting =  beta_p*rho_i*g;

          ScalarT resid_tmp = -F*scale + (1-scale) * 1e-3*k_i*dTdz_melting;
          ScalarT m = F+0.001*k_i*(basal_dTdz(cell,side,qp));
          ScalarT scale_m = - atan(alpha * m)/pi + 0.5;

          enthalpyBasalResid(cell,cnode) += resid_tmp *  BF(cell,side,node,qp) * w_measure(cell,side,qp);
        }
        enthalpyBasalResid(cell,cnode) += 0./robin_coeff * diffEnth(cell,cnode) + alpha / pi* isThereWater *std::min(0., diffEnth(cell,cnode));
      }
    }


    if (haveSUPG)
    {
      // Zero out, to avoid leaving stuff from previous workset!
      for (int cell = 0; cell < d.numCells; ++cell)
        for (int node = 0; node < numCellNodes; ++node)
          enthalpyBasalResidSUPG(cell,node) = 0.;

      for (auto const& iter_side : sideSet)
      {
        // Get the local data of side and cell
        const int cell = iter_side.elem_LID;
        const int side = iter_side.side_local_id;

        for (int node = 0; node < numSideNodes; ++node)
        {
          int cnode = sideNodes[side][node];
          enthalpyBasalResidSUPG(cell,cnode) = 0.;
          ScalarT scale = - atan(alpha * diffEnth(cell,cnode))/pi + 0.5;

          for (int qp = 0; qp < numSideQPs; ++qp)
          {
            ScalarT wSUPG = 0.001 / scyr * // [km^2 s^{-1}]
                (velocity(cell,side,qp,0)*GradBF(cell,side,node,qp,0) + velocity(cell,side,qp,1)*GradBF(cell,side,node,qp,1)+verticalVel(cell,side,qp) * GradBF(cell,side,node,qp,2))*w_measure(cell,side,qp);
            //     ScalarT scale = - atan(alpha * (enthalpy(cell,side,node,qp)-enthalpyHs(cell,side,node,qp)))/pi + 0.5;


            ScalarT resid_tmp = - geoFlux(cell,side,qp)*scale;
            resid_tmp += robin_coeff*std::fabs(basal_dTdz(cell,side,qp))*std::max(0.,enthalpy(cell,side,node,qp)-enthalpyHs(cell,side,node,qp));
            for (int dim = 0; dim < vecDimFO; ++dim)
              resid_tmp -= 1000/scyr * beta(cell,side,qp) * std::pow(velocity(cell,side,qp,dim),2) *scale;

            enthalpyBasalResidSUPG(cell,cnode) += resid_tmp *  wSUPG;
          }
        }
      }
    }
  }
}
