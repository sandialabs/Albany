//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyRhs<EvalT, Traits>::HydrologyRhs (const Teuchos::ParameterList& p,
                                           const Teuchos::RCP<Albany::Layouts>& dl) :
  mu_i      (p.get<std::string> ("Ice Viscosity QP Variable Name"), dl->qp_scalar),
  h         (p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar),
  H         (p.get<std::string> ("Ice Thickness QP Variable Name"), dl->qp_scalar),
  z_b       (p.get<std::string> ("Basal Height QP Variable Name"), dl->qp_scalar),
  u_b       (p.get<std::string> ("Basal Velocity Magnitude QP Variable Name"), dl->qp_scalar),
  beta      (p.get<std::string> ("Basal Friction Coefficient QP Variable Name"), dl->qp_scalar),
  omega     (p.get<std::string> ("Surface Water Input QP Variable Name"), dl->qp_scalar),
  G         (p.get<std::string> ("Geothermal Heat Source QP Variable Name"), dl->qp_scalar),
  rhs       (p.get<std::string> ("RHS QP Name"),dl->qp_scalar)
{
  this->addDependentField(mu_i);
  this->addDependentField(h);
  this->addDependentField(H);
  this->addDependentField(z_b);
  this->addDependentField(u_b);
  this->addDependentField(beta);
  this->addDependentField(omega);
  this->addDependentField(G);

  this->addEvaluatedField(rhs);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  R = hydrology_params.get<double>("Bed Roughness");

  mu_w         = physical_params.get<double>("Water Viscosity");
  rho_i        = physical_params.get<double>("Ice Density");
  rho_w        = physical_params.get<double>("Water Density");
  L            = physical_params.get<double>("Ice Latent Heat");
  rho_combo    = (rho_i - rho_w)/(rho_i*rho_w);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_scalar->dimensions(dims);
  numQPs = dims[1];

  this->setName("HydrologyRhs"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyRhs<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(mu_i,fm);
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(z_b,fm);
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(omega,fm);
  this->utils.setFieldData(G,fm);

  this->utils.setFieldData(rhs,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyRhs<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT gradPhiNormSq, resRhs, resMass, resNlin, resStiff;
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int qp=0; qp < numQPs; ++qp)
    {
      rhs(cell,qp) = omega(cell,qp) - R*u_b(cell,qp)
                     + rho_i*h(cell,qp)*H(cell,qp)/mu_i(cell,qp)
                     + rho_w*h(cell,qp)*z_b(cell,qp)/mu_i(cell,qp)
                     + rho_combo*G(cell,qp)/L
                     + rho_combo*pow(u_b(cell,qp),2)*beta(cell,qp)/L;
    }
  }
}

} // Namespace FELIX
