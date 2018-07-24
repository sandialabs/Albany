/*
 * LandIce_EnthalpyResid_Def.hpp
 *
 *  Created on: May 11, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace LandIce
{

  template<typename Type>
  Type distance (const Type& x0, const Type& x1, const Type& x2,
                 const Type& y0, const Type& y1, const Type& y2)
  {
    return std::sqrt(std::pow(x0-y0,2) +
                     std::pow(x1-y1,2) +
                     std::pow(x2-y2,2));
  }


  template<typename EvalT, typename Traits, typename VelocityType>
  EnthalpyResid<EvalT,Traits,VelocityType>::
  EnthalpyResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  wBF      		(p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
  wGradBF  		(p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
  Enthalpy        (p.get<std::string> ("Enthalpy QP Variable Name"), dl->qp_scalar),
  EnthalpyGrad    (p.get<std::string> ("Enthalpy Gradient QP Variable Name"), dl->qp_gradient),
  EnthalpyHs		(p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl->qp_scalar ),
  diffEnth      (p.get<std::string> ("Diff Enthalpy Variable Name"), dl->node_scalar),
  Velocity		(p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
  velGrad    (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  verticalVel		(p.get<std::string> ("Vertical Velocity QP Variable Name"),  dl->qp_scalar),
  coordVec 		(p.get<std::string> ("Coordinate Vector Name"),dl->vertices_vector),
  meltTempGrad	(p.get<std::string> ("Melting Temperature Gradient QP Variable Name"), dl->qp_gradient),
  phi			    (p.get<std::string> ("Water Content QP Variable Name"), dl->qp_scalar ),
  phiGrad		    (p.get<std::string> ("Water Content Gradient QP Variable Name"), dl->qp_gradient ),
  basalResid    (p.get<std::string>("Enthalpy Basal Residual Variable Name"), dl->node_scalar),
  Residual 		(p.get<std::string> ("Residual Variable Name"), dl->node_scalar),
  homotopy		(p.get<std::string> ("Continuation Parameter Name"), dl->shared_param)
  {
    Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
    std::vector<PHX::Device::size_type> dims;
    vector_dl->dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    vecDimFO = 2;

    if(p.isParameter("LandIce Enthalpy Stabilization")) {
      Teuchos::ParameterList* stabilization_list = p.get<Teuchos::ParameterList*>("LandIce Enthalpy Stabilization");
      const std::string& sname = stabilization_list->get<std::string>("Type");
      stabilization = (sname == "SUPG") ? STABILIZATION_TYPE::SUPG :
        (sname == "Streamline Upwind") ? STABILIZATION_TYPE::SU : STABILIZATION_TYPE::NONE;
      delta = stabilization_list->get("Parameter Delta", 0.1);
    }
    else {
      stabilization = STABILIZATION_TYPE::NONE;
      delta = 0;
    }

    haveSUPG = (stabilization == STABILIZATION_TYPE::SUPG);


    needsDiss = p.get<bool>("Needs Dissipation");
    needsBasFric = p.get<bool>("Needs Basal Friction");

    this->addDependentField(Enthalpy);
    this->addDependentField(EnthalpyGrad);
    this->addDependentField(EnthalpyHs);
    this->addDependentField(diffEnth);
    this->addDependentField(wBF);
    this->addDependentField(wGradBF);
    this->addDependentField(Velocity);
    this->addDependentField(velGrad);
    this->addDependentField(verticalVel);
    this->addDependentField(coordVec);
    this->addDependentField(meltTempGrad);
    this->addDependentField(phi);
    this->addDependentField(phiGrad);
    this->addDependentField(homotopy);
    this->addDependentField(basalResid);

    if (needsDiss)
    {
      diss = decltype(diss)(p.get<std::string> ("Dissipation QP Variable Name"),dl->qp_scalar);
      this->addDependentField(diss);
    }

    if (needsBasFric)
    {
      //basalFricHeat = decltype(basalFricHeat)(p.get<std::string> ("Basal Friction Heat QP Variable Name"),dl->node_scalar);
      //this->addDependentField(basalFricHeat);

      if(haveSUPG)
      {
        basalFricHeatSUPG = decltype(basalFricHeatSUPG)(p.get<std::string> ("Basal Friction Heat QP SUPG Variable Name"),dl->node_scalar);
        //this->addDependentField(basalFricHeatSUPG);
      }
    }

    //geoFluxHeat = decltype(geoFluxHeat)(p.get<std::string> ("Geothermal Flux Heat QP Variable Name"),dl->node_scalar);
    //this->addDependentField(geoFluxHeat);

    if(haveSUPG)
    {
      geoFluxHeatSUPG = decltype(geoFluxHeatSUPG)(p.get<std::string> ("Geothermal Flux Heat QP SUPG Variable Name"),dl->node_scalar);
      basalResidSUPG = decltype(basalResidSUPG)(p.get<std::string> ("Enthalpy Basal Residual SUPG Variable Name"),dl->node_scalar);
      //this->addDependentField(geoFluxHeatSUPG);
      this->addDependentField(basalFricHeatSUPG);
    }

    this->addEvaluatedField(Residual);
    this->setName("EnthalpyResid");

    Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
    rho_i = physics_list->get<double>("Ice Density"); //[Kg m^{-3}]
    rho_w = physics_list->get<double>("Water Density"); //[Kg m^{-3}]

    k_i = physics_list->get<double>("Conductivity of ice"); //[W m^{-1} K^{-1}]
    c_i = physics_list->get<double>("Heat capacity of ice"); //[J Kg^{-1} K^{-1}]
    K_i = k_i / (rho_i * c_i); //[m^2 s^{-1}]

    nu = physics_list->get<double>("Diffusivity temperate ice"); //[m^2 s^{-1}]

    k_0 = physics_list->get<double>("Permeability factor"); //[m^2]
    eta_w = physics_list->get<double>("Viscosity of water"); //[Pa s]
    g = physics_list->get<double>("Gravity Acceleration"); //[m s^{-2}]
    L = physics_list->get<double>("Latent heat of fusion"); //[J kg^{-1} ]
    alpha_om = physics_list->get<double>("Omega exponent alpha");

    a = physics_list->get<double>("Diffusivity homotopy exponent");

    drainage_coeff = g * rho_w * L * k_0 * (rho_w - rho_i) / eta_w; //[kg s^{-3}]

    std::cout << "Drainage: " << drainage_coeff/rho_w/L*3.1536e7 << std::endl;

    printedRegCoeff = -1.0;


    Teuchos::ParameterList* regularization_list = p.get<Teuchos::ParameterList*>("LandIce Enthalpy Regularization");
    auto flux_reg_list = regularization_list->sublist("Enthalpy Flux Regularization", false);\
    flux_reg_alpha = flux_reg_list.get<double>("alpha");
    flux_reg_beta = flux_reg_list.get<double>("beta");
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void EnthalpyResid<EvalT,Traits,VelocityType>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(Enthalpy,fm);
    this->utils.setFieldData(EnthalpyGrad,fm);
    this->utils.setFieldData(EnthalpyHs,fm);
    this->utils.setFieldData(diffEnth,fm);
    this->utils.setFieldData(wBF,fm);
    this->utils.setFieldData(wGradBF,fm);
    this->utils.setFieldData(Velocity,fm);
    this->utils.setFieldData(velGrad,fm);
    this->utils.setFieldData(verticalVel,fm);
    this->utils.setFieldData(coordVec,fm);
    this->utils.setFieldData(meltTempGrad,fm);
    this->utils.setFieldData(phi,fm);
    this->utils.setFieldData(phiGrad,fm);
    this->utils.setFieldData(homotopy,fm);
    this->utils.setFieldData(basalResid,fm);

    if (needsDiss)
      this->utils.setFieldData(diss,fm);

    if (needsBasFric)
    {
      //this->utils.setFieldData(basalFricHeat,fm);
      if(haveSUPG) {
        //this->utils.setFieldData(basalFricHeatSUPG,fm);
        this->utils.setFieldData(basalResidSUPG,fm);
      }
    }

    //this->utils.setFieldData(geoFluxHeat,fm);
    //if(haveSUPG)
      //this->utils.setFieldData(geoFluxHeatSUPG,fm);

    this->utils.setFieldData(Residual,fm);
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void EnthalpyResid<EvalT,Traits,VelocityType>::
  evaluateFields(typename Traits::EvalData d)
  {
    const double scyr (3.1536e7);  // [s/yr];
    const double drain_vel = drainage_coeff/rho_w/L*scyr; // [m/yr];
    const double pow3 = 1e3;    //[k^{-1}, k=1000
    const double powm3 = 1e-3;  //[k], k=1000
    const double powm6 = 1e-6;  //[k^2], k=1000
    const double powm9 = 1e-9;  //[k^3], k=1000
    ScalarT K;
    double pi = atan(1.) * 4.;
    ScalarT hom = homotopy(0);


    ScalarT flux_reg_coeff = flux_reg_alpha*exp(flux_reg_beta*hom); // [adim]

    if (std::fabs(printedRegCoeff - flux_reg_coeff) > 0.0001*flux_reg_coeff)
    {
      std::cout << "[Diffusivity()] alpha = " << flux_reg_coeff << " :: " <<hom << "\n";
      printedRegCoeff = flux_reg_coeff;
    }

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
      for (std::size_t node = 0; node < numNodes; ++node)
        Residual(cell,node) = 0.0;

 // double x_p = -35,z_p = 1;

    if (needsDiss)	// this term is always in regardless the bc at the base
    {
      for (std::size_t cell = 0; cell < d.numCells; ++cell)
      {
        for (std::size_t node = 0; node < numNodes; ++node)
        {
//        double x = coordVec(cell,node,0), y = coordVec(cell,node,1), z = coordVec(cell,node,2);
          for (std::size_t qp = 0; qp < numQPs; ++qp)
          {
//          ScalarT scale = 1+2000*std::max(ScalarT(0.1-std::pow(x-x_p,2)/2500 - std::pow(z-z_p,2)),ScalarT(0));
//          Residual(cell,node) -= powm3*Albany::ADValue(diss(cell,qp))*wBF(cell,node,qp);
            Residual(cell,node) -= powm3*(diss(cell,qp))*wBF(cell,node,qp);
          }
        }
      }
    }

    if (needsBasFric)
    {
      for (std::size_t cell = 0; cell < d.numCells; ++cell)
      {

        //ScalarT diffEnt=0;
        //for (std::size_t qp = 0; qp < numQPs; ++qp)
        //  diffEnt += Enthalpy(cell,qp) - EnthalpyHs(cell,qp);
        //diffEnt /= numQPs;

        //ScalarT scale = - atan(alpha * diffEnt)/pi + 0.5;
        //scale = Albany::ADValue(scale);
        for (std::size_t node = 0; node < numNodes; ++node)
        {
          // Modify here if you want to impose different basal BC. NB: in case of temperate ice, we disregard the extra boundary term related to the gradient of the T_m. You might want to reconsider this in the future
          //Residual(cell,node) -= powm6*( basalFricHeat(cell,node) + geoFluxHeat(cell,node) ) * scale;  //go to zero in temperate region
          Residual(cell,node) += powm6*basalResid(cell,node);  //go to zero in temperate region
        }
      }
    }

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
      for (std::size_t node = 0; node < numNodes; ++node)
      {
//      ScalarT scale = 0.5 - atan(flux_reg_coeff * diffEnth(cell,node))/pi;
        for (std::size_t qp = 0; qp < numQPs; ++qp)
        {
          ScalarT scale = 0.5 - atan(flux_reg_coeff * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi;
          //scale = Albany::ADValue(scale);
          Residual(cell,node) += powm3 * scale * K_i * (EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
              EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
              EnthalpyGrad(cell,qp,2)*wGradBF(cell,node,qp,2));

          Residual(cell,node) += (Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
                  Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + verticalVel(cell,qp)*EnthalpyGrad(cell,qp,2))*wBF(cell,node,qp)/scyr ;
//          Residual(cell,node) += (Albany::ADValue(Velocity(cell,qp,0))*EnthalpyGrad(cell,qp,0) +
//                  Albany::ADValue(Velocity(cell,qp,1))*EnthalpyGrad(cell,qp,1) + verticalVel(cell,qp)*EnthalpyGrad(cell,qp,2))*wBF(cell,node,qp)/scyr ;

          Residual(cell,node) += powm9*(1 - scale)*(k_i + rho_i*c_i*nu) * (meltTempGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
              meltTempGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
              meltTempGrad(cell,qp,2)*wGradBF(cell,node,qp,2));

          Residual(cell,node) -= powm6 * drainage_coeff*alpha_om*pow(phi(cell,qp),alpha_om-1)*phiGrad(cell,qp,2)*wBF(cell,node,qp);
         // Residual(cell,node) += powm6*(1 - scale) * drainage_coeff*pow(phi(cell,qp),alpha_om)*wGradBF(cell,node,qp,2);
        }
      }
    }


    if((stabilization == STABILIZATION_TYPE::SU) || (stabilization == STABILIZATION_TYPE::SUPG)) {

      for (std::size_t cell = 0; cell < d.numCells; ++cell)
      {
        VelocityType  vmax_xy = 1e-3; //set to a minimum threshold
        ScalarT vmax = 1e-3, vmax_z=1e-5; //set to a minimum threshold
        ParamScalarT diam = 0.0, diam_xy = 0.0, diam_z = 0.0;
        ScalarT wSUPG = 0.0;
        for (std::size_t qp = 0; qp < numQPs; ++qp) {
          //ScalarT scale = - atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi + 0.5;
          vmax = std::max(vmax,std::sqrt(std::pow(Velocity(cell,qp,0),2)+std::pow(Velocity(cell,qp,1),2)+std::pow(verticalVel(cell,qp),2)));
          vmax_xy = std::max(vmax_xy,std::sqrt(std::pow(Velocity(cell,qp,0),2)+std::pow(Velocity(cell,qp,1),2)));
//          vmax = std::max(vmax,std::sqrt(std::pow(Albany::ADValue(Velocity(cell,qp,0)),2)+std::pow(Albany::ADValue(Velocity(cell,qp,1)),2)+std::pow(verticalVel(cell,qp),2)));
//          vmax_xy = std::max(vmax_xy,std::sqrt(std::pow(Albany::ADValue(Velocity(cell,qp,0)),2)+std::pow(Albany::ADValue(Velocity(cell,qp,1)),2)));


          vmax_z = std::max( vmax_z,std::fabs(verticalVel(cell,qp)));
          //vmax_z = std::max( vmax_z,std::fabs(- (1-scale)*alpha_om*pow(phi(cell,qp),alpha_om-1)*drain_vel+verticalVel(cell,qp)));
        }

        for (std::size_t i = 0; i < numNodes-1; ++i)
        {
          for (std::size_t j = i + 1; j < numNodes; ++j)
          {
            diam = std::max(diam,distance<ParamScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
                                                        coordVec(cell,j,0),coordVec(cell,j,1),coordVec(cell,j,2)));
            diam_xy = std::max(diam_xy,distance<ParamScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),0,
                                                                  coordVec(cell,j,0),coordVec(cell,j,1),0));
            diam_z = std::max(diam_z,std::fabs(coordVec(cell,i,2) - coordVec(cell,j,2)));
          }
        }


        for (std::size_t node = 0; node < numNodes; ++node)
        {
          for (std::size_t qp = 0; qp < numQPs; ++qp)
          {
/*
            //ScalarT scale = - atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi + 0.5;
            //ScalarT totalVertVel = verticalVel(cell,qp);// - alpha_om*pow(phi(cell,qp),alpha_om-1)*(1-scale)*drain_vel;
            //wSUPG = delta*(std::sqrt(diam_xy/vmax_xy)*(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1)) + std::sqrt(diam_z/vmax_z)*totalVertVel * wGradBF(cell,node,qp,2));
            //Residual(cell,node) += (std::sqrt(diam_xy/vmax_xy)*(Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
            //    Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1)) + std::sqrt(diam_z/vmax_z)*totalVertVel*EnthalpyGrad(cell,qp,2))*wSUPG/scyr;

            ScalarT totalVertVel = verticalVel(cell,qp);// - alpha_om*pow(phi(cell,qp),alpha_om-1)*(1-scale)*drain_vel;
            ScalarT  wSUPG_xy = delta*diam_xy/vmax_xy*(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1));
            ScalarT  wSUPG_z = delta*diam_z/vmax_z*totalVertVel * wGradBF(cell,node,qp,2);

            Residual(cell,node) += (wSUPG_xy*(Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
                Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1)) +  wSUPG_z*totalVertVel*EnthalpyGrad(cell,qp,2))/scyr;
/*/
            wSUPG = delta*diam/vmax*(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) + verticalVel(cell,qp) * wGradBF(cell,node,qp,2)); // +(velGrad(cell,qp,0,0)+velGrad(cell,qp,1,1))*wBF(cell,node,qp));
//            ScalarT totalVertVel = verticalVel(cell,qp) - alpha_om*pow(phi(cell,qp),alpha_om-1)*drain_vel;
//            wSUPG = delta*diam/vmax*(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) + totalVertVel * wGradBF(cell,node,qp,2)); // +(velGrad(cell,qp,0,0)+velGrad(cell,qp,1,1))*wBF(cell,node,qp));
            Residual(cell,node) += (Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
                Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + verticalVel(cell,qp)*EnthalpyGrad(cell,qp,2))*wSUPG/scyr;
//            wSUPG = delta*diam/vmax*(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) + totalVertVel * wGradBF(cell,node,qp,2)); // +(velGrad(cell,qp,0,0)+velGrad(cell,qp,1,1))*wBF(cell,node,qp));
//           Residual(cell,node) += (Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
//                Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + totalVertVel*EnthalpyGrad(cell,qp,2))*wSUPG/scyr;
           // wSUPG = delta*diam/vmax*(Albany::ADValue(Velocity(cell,qp,0)) * wGradBF(cell,node,qp,0) + Albany::ADValue(Velocity(cell,qp,1)) * wGradBF(cell,node,qp,1) + totalVertVel * wGradBF(cell,node,qp,2)); // +(velGrad(cell,qp,0,0)+velGrad(cell,qp,1,1))*wBF(cell,node,qp));
           // Residual(cell,node) += (Albany::ADValue(Velocity(cell,qp,0))*EnthalpyGrad(cell,qp,0) +
           //     Albany::ADValue(Velocity(cell,qp,1))*EnthalpyGrad(cell,qp,1) + totalVertVel*EnthalpyGrad(cell,qp,2))*wSUPG/scyr;
//*/
          }
        }
      }
    }
  }
}
