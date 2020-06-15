/*
 * LandIce_EnthalpyResid_Def.hpp
 *
 *  Created on: May 11, 2016
 *      Author: mperego, abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_EnthalpyResid.hpp"

namespace LandIce
{

template<typename Type>
KOKKOS_INLINE_FUNCTION
Type distance (const Type& x0, const Type& x1, const Type& x2,
               const Type& y0, const Type& y1, const Type& y2)
{
  Type tmp = std::pow(x0-y0,2)+std::pow(x1-y1,2)+std::pow(x2-y2,2);
  if(tmp > 0.0)    
    return std::sqrt(tmp);
  else 
    return 0;
}

template<typename Type>
KOKKOS_INLINE_FUNCTION
Type deviceMax(Type a, Type b)
{
  return (a > b) ? a : b;
}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
EnthalpyResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  wBF      		 (p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
  wGradBF  		 (p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
  Enthalpy     (p.get<std::string> ("Enthalpy QP Variable Name"), dl->qp_scalar),
  EnthalpyGrad (p.get<std::string> ("Enthalpy Gradient QP Variable Name"), dl->qp_gradient),
  EnthalpyHs   (p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl->qp_scalar ),
  diffEnth     (p.get<std::string> ("Diff Enthalpy Variable Name"), dl->node_scalar),
  Velocity		 (p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
  velGrad      (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  verticalVel	 (p.get<std::string> ("Vertical Velocity QP Variable Name"),  dl->qp_scalar),
  coordVec 		 (p.get<std::string> ("Coordinate Vector Name"),dl->vertices_vector),
  meltTempGrad (p.get<std::string> ("Melting Temperature Gradient QP Variable Name"), dl->qp_gradient),
  phi			     (p.get<std::string> ("Water Content QP Variable Name"), dl->qp_scalar ),
  phiGrad		   (p.get<std::string> ("Water Content Gradient QP Variable Name"), dl->qp_gradient ),
  basalResid   (p.get<std::string> ("Enthalpy Basal Residual Variable Name"), dl->node_scalar),
  Residual 		 (p.get<std::string> ("Residual Variable Name"), dl->node_scalar),
  homotopy		 (p.get<std::string> ("Continuation Parameter Name"), dl->shared_param)
{
  std::vector<PHX::Device::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  vecDimFO = 2;

  if(p.isParameter("LandIce Enthalpy Stabilization")) {
    Teuchos::ParameterList* stabilization_list = p.get<Teuchos::ParameterList*>("LandIce Enthalpy Stabilization");
    const std::string& sname = stabilization_list->get<std::string>("Type");
    stabilization = (sname == "Streamline Upwind") ? STABILIZATION_TYPE::SU :
            (sname == "Upwind") ? STABILIZATION_TYPE::UPWIND :
                STABILIZATION_TYPE::NONE;
    delta = stabilization_list->get("Parameter Delta", 0.1);
  }
  else {
    stabilization = STABILIZATION_TYPE::NONE;
    delta = 0;
  }

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
  scyr = physics_list->get<double>("Seconds per Year");

#ifdef OUTPUT_TO_SCREEN
  std::cout << "Drainage: " << drainage_coeff/rho_w/L*scyr << std::endl;
#endif

  printedRegCoeff = -1.0;


  Teuchos::ParameterList* regularization_list = p.get<Teuchos::ParameterList*>("LandIce Enthalpy Regularization");
  auto flux_reg_list = regularization_list->sublist("Flux Regularization", false);
  flux_reg_alpha = flux_reg_list.get<double>("alpha");
  flux_reg_beta = flux_reg_list.get<double>("beta");
}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
KOKKOS_INLINE_FUNCTION
void EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
stabilizationInitialization(int cell, VelocityST& vmax_xy, ScalarT& vmax, ScalarT& vmax_z, 
  MeshScalarT& diam, MeshScalarT& diam_xy, MeshScalarT& diam_z, ScalarT& wSU) const {
  for (std::size_t qp = 0; qp < numQPs; ++qp) {
      ScalarT w = verticalVel(cell,qp);
      ScalarT arg = Velocity(cell,qp,0)*Velocity(cell,qp,0) + Velocity(cell,qp,1)*Velocity(cell,qp,1) + w*w;
      ScalarT arg2 = 0.0; 
      if (arg > 0) arg2 = std::sqrt(arg); 
      vmax = deviceMax<ScalarT>(vmax, arg2);
      //vmax_xy = std::max(vmax_xy,std::sqrt(std::pow(Velocity(cell,qp,0),2)+std::pow(Velocity(cell,qp,1),2)));
      VelocityST val = Velocity(cell,qp,0)*Velocity(cell,qp,0)+Velocity(cell,qp,1)*Velocity(cell,qp,1); 
      VelocityST sqrtval = 0.0;
      if (val > 0.0)
        sqrtval = std::sqrt(val); 

      vmax_xy = deviceMax<VelocityST>(vmax_xy, sqrtval);

      vmax_z = deviceMax<ScalarT>( vmax_z,std::abs(w));
    }

    for (std::size_t i = 0; i < numNodes; ++i) {
      diam = deviceMax<MeshScalarT>(diam,distance<MeshScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
                                                    coordVec(cell,0,0),coordVec(cell,0,1),coordVec(cell,0,2)));
      diam_xy = deviceMax<MeshScalarT>(diam_xy,distance<MeshScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),MeshScalarT(0.0),
                                                          coordVec(cell,0,0),coordVec(cell,0,1),MeshScalarT(0.0)));
      diam_z = deviceMax<MeshScalarT>(diam_z,std::abs(coordVec(cell,i,2) - coordVec(cell,0,2)));
    }
}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
KOKKOS_INLINE_FUNCTION
void EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
evaluateResidNode(int cell, int node) const {
  Residual(cell,node) = 0.0;

  Residual(cell,node) += powm3*basalResid(cell,node);  //go to zero in temperate region

  for (std::size_t qp = 0; qp < numQPs; ++qp) {
    if (needsDiss)
      Residual(cell,node) -= (diss(cell,qp))*wBF(cell,node,qp);

    ScalarT w = verticalVel(cell,qp);
    ScalarT scale = 0.5 - 0.5*tanh(flux_reg_coeff * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)));
    Residual(cell,node) += scale * K_i * (EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
        EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
        EnthalpyGrad(cell,qp,2)*wGradBF(cell,node,qp,2));

    Residual(cell,node) += pow3*(Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
            Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + w*EnthalpyGrad(cell,qp,2))*wBF(cell,node,qp)/scyr ;

    Residual(cell,node) += powm6*(1 - scale) * k_i * (meltTempGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
        meltTempGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
        meltTempGrad(cell,qp,2)*wGradBF(cell,node,qp,2));

    Residual(cell,node) -= powm3 * (1 - scale) * drainage_coeff*alpha_om*pow(phi(cell,qp),alpha_om-1)*phiGrad(cell,qp,2)*wBF(cell,node,qp) +
                            nu * (1 - scale) * powm6 * rho_w * L * (phiGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
                                phiGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
                                phiGrad(cell,qp,2)*wGradBF(cell,node,qp,2));
  }
}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
KOKKOS_INLINE_FUNCTION
void EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
operator() (const Upwind_Stabilization_Tag& tag, const int& cell) const{

  VelocityST  vmax_xy = 1e-3; //set to a minimum threshold
  ScalarT vmax = 1e-3, vmax_z=1e-5; //set to a minimum threshold
  MeshScalarT diam = 0.0, diam_xy = 0.0, diam_z = 0.0;
  ScalarT wSU = 0.0;

  stabilizationInitialization(cell, vmax_xy, vmax, vmax_z, diam, diam_xy, diam_z, wSU);

  for (std::size_t node = 0; node < numNodes; ++node) {
    evaluateResidNode(cell, node);

    for (std::size_t qp = 0; qp < numQPs; ++qp) {
      Residual(cell,node) += pow3*(EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
                              EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1))*delta*diam_xy*vmax_xy/scyr;
      Residual(cell,node) += pow3*EnthalpyGrad(cell,qp,2)* wGradBF(cell,node,qp,2)*delta*diam_z*vmax_z/scyr;
    }
  }

}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
KOKKOS_INLINE_FUNCTION
void EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
operator() (const SU_Stabilization_Tag& tag, const int& cell) const{
  VelocityST  vmax_xy = 1e-3; //set to a minimum threshold
  ScalarT vmax = 1e-3, vmax_z=1e-5; //set to a minimum threshold
  MeshScalarT diam = 0.0, diam_xy = 0.0, diam_z = 0.0;
  ScalarT wSU = 0.0;

  stabilizationInitialization(cell, vmax_xy, vmax, vmax_z, diam, diam_xy, diam_z, wSU);

  for (std::size_t node = 0; node < numNodes; ++node) {
    evaluateResidNode(cell, node);

    for (std::size_t qp = 0; qp < numQPs; ++qp) {
      ScalarT w = verticalVel(cell,qp);
      wSU = delta*diam/vmax*(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) + w * wGradBF(cell,node,qp,2)); // +(velGrad(cell,qp,0,0)+velGrad(cell,qp,1,1))*wBF(cell,node,qp));
      Residual(cell,node) += pow3*(Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
          Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + w*EnthalpyGrad(cell,qp,2))*wSU/scyr;
    }
  }

}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
KOKKOS_INLINE_FUNCTION
void EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
operator() (const Other_Stabilization_Tag& tag, const int& cell) const{

  for (std::size_t node = 0; node < numNodes; ++node) {
    evaluateResidNode(cell, node);
  }
  
}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
void EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
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

  this->utils.setFieldData(Residual,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
void EnthalpyResid<EvalT,Traits,VelocityST,MeltTempST>::
evaluateFields(typename Traits::EvalData d)
{
  ScalarT K;
  double pi = atan(1.) * 4.;
  ScalarT hom = homotopy(0);

  flux_reg_coeff = flux_reg_alpha*exp(flux_reg_beta*hom); // [adim]

#ifdef OUTPUT_TO_SCREEN
  if (std::fabs(printedRegCoeff - flux_reg_coeff) > 0.0001*flux_reg_coeff)
  {
    std::cout << "[Diffusivity()] alpha = " << flux_reg_coeff << " :: " <<hom << "\n";
    printedRegCoeff = flux_reg_coeff;
  }
#endif

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if(stabilization == STABILIZATION_TYPE::UPWIND){
    Kokkos::parallel_for(Upwind_Stabilization_Policy(0,d.numCells), *this);
  }
  else if(stabilization == STABILIZATION_TYPE::SU){
    Kokkos::parallel_for(SU_Stabilization_Policy(0,d.numCells), *this);
  }
  else{
    Kokkos::parallel_for(Other_Stabilization_Policy(0,d.numCells), *this);
  }
#else
  for (std::size_t cell = 0; cell < d.numCells; ++cell)
  {
    VelocityST  vmax_xy = 1e-3; //set to a minimum threshold
    ScalarT vmax = 1e-3, vmax_z=1e-5; //set to a minimum threshold
    MeshScalarT diam = 0.0, diam_xy = 0.0, diam_z = 0.0;
    ScalarT wSU = 0.0;

    if((stabilization == STABILIZATION_TYPE::UPWIND) || (stabilization == STABILIZATION_TYPE::SU))
    {
      
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        ScalarT w = verticalVel(cell,qp);
        ScalarT arg = Velocity(cell,qp,0)*Velocity(cell,qp,0) + Velocity(cell,qp,1)*Velocity(cell,qp,1) + w*w;
        ScalarT arg2 = 0.0; 
        if (arg > 0) arg2 = std::sqrt(arg); 
        vmax = std::max(vmax, arg2);
        //vmax_xy = std::max(vmax_xy,std::sqrt(std::pow(Velocity(cell,qp,0),2)+std::pow(Velocity(cell,qp,1),2)));
        VelocityST val = Velocity(cell,qp,0)*Velocity(cell,qp,0)+Velocity(cell,qp,1)*Velocity(cell,qp,1); 
        VelocityST sqrtval = 0.0;
        if (val > 0.0)
          sqrtval = std::sqrt(val); 

        vmax_xy = std::max(vmax_xy, sqrtval);

        vmax_z = std::max( vmax_z,std::abs(w));
      }

      for (std::size_t i = 0; i < numNodes; ++i)
      {
        diam = std::max(diam,distance<MeshScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
                                                      coordVec(cell,0,0),coordVec(cell,0,1),coordVec(cell,0,2)));
        diam_xy = std::max(diam_xy,distance<MeshScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),MeshScalarT(0.0),
                                                            coordVec(cell,0,0),coordVec(cell,0,1),MeshScalarT(0.0)));
        diam_z = std::max(diam_z,std::abs(coordVec(cell,i,2) - coordVec(cell,0,2)));
      }
    }
    
    for (std::size_t node = 0; node < numNodes; ++node)
    {
      Residual(cell,node) = 0.0;

      Residual(cell,node) += powm3*basalResid(cell,node);  //go to zero in temperate region

      for (std::size_t qp = 0; qp < numQPs; ++qp)
      {
        if (needsDiss)
          Residual(cell,node) -= (diss(cell,qp))*wBF(cell,node,qp);

        ScalarT w = verticalVel(cell,qp);
        ScalarT scale = 0.5 - 0.5*tanh(flux_reg_coeff * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)));
        Residual(cell,node) += scale * K_i * (EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
            EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
            EnthalpyGrad(cell,qp,2)*wGradBF(cell,node,qp,2));

        Residual(cell,node) += pow3*(Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
                Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + w*EnthalpyGrad(cell,qp,2))*wBF(cell,node,qp)/scyr ;

        Residual(cell,node) += powm6*(1 - scale) * k_i * (meltTempGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
            meltTempGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
            meltTempGrad(cell,qp,2)*wGradBF(cell,node,qp,2));

        Residual(cell,node) -= powm3 * (1 - scale) * drainage_coeff*alpha_om*pow(phi(cell,qp),alpha_om-1)*phiGrad(cell,qp,2)*wBF(cell,node,qp) +
                               nu * (1 - scale) * powm6 * rho_w * L * (phiGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
                                   phiGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
                                   phiGrad(cell,qp,2)*wGradBF(cell,node,qp,2));
      }

      if(stabilization == STABILIZATION_TYPE::UPWIND) 
      {
        for (std::size_t node = 0; node < numNodes; ++node)
        {
          for (std::size_t qp = 0; qp < numQPs; ++qp)
          {
            Residual(cell,node) += pow3*(EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
                                    EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1))*delta*diam_xy*vmax_xy/scyr;
            Residual(cell,node) += pow3*EnthalpyGrad(cell,qp,2)* wGradBF(cell,node,qp,2)*delta*diam_z*vmax_z/scyr;
          }
        }
      }


      else if(stabilization == STABILIZATION_TYPE::SU) 
      {
        for (std::size_t node = 0; node < numNodes; ++node)
        {
          for (std::size_t qp = 0; qp < numQPs; ++qp)
          {
            ScalarT w = verticalVel(cell,qp);
            wSU = delta*diam/vmax*(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) + w * wGradBF(cell,node,qp,2)); // +(velGrad(cell,qp,0,0)+velGrad(cell,qp,1,1))*wBF(cell,node,qp));
            Residual(cell,node) += pow3*(Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
                Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + w*EnthalpyGrad(cell,qp,2))*wSU/scyr;
          }
        }
      }
    }
  }
#endif

}

} // namespace LandIce
