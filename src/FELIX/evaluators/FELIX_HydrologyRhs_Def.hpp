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
  mu_i      (p.get<std::string> ("Ice Viscosity Variable Name"), dl->cell_scalar2),
  h         (p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar),
  H         (p.get<std::string> ("Ice Thickness QP Variable Name"), dl->qp_scalar),
  z_s       (p.get<std::string> ("Surface Height QP Variable Name"), dl->qp_scalar),
  u_b       (p.get<std::string> ("Sliding Velocity Norm QP Variable Name"), dl->qp_scalar),
  omega     (p.get<std::string> ("Surface Water Input QP Variable Name"), dl->qp_scalar),
  rhs       (p.get<std::string> ("RHS QP Name"),dl->qp_scalar)
{
  this->addDependentField(mu_i);
  this->addDependentField(h);
  this->addDependentField(H);
  this->addDependentField(z_s);
  this->addDependentField(u_b);
  this->addDependentField(omega);

  this->addEvaluatedField(rhs);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  h_b = hydrology_params.get<double>("Bed Bumps Height");
  l_b = hydrology_params.get<double>("Bed Bumps Length");

  if (hydrology_params.get<bool>("Use Net Bump Height",false))
  {
    use_net_bump_height = 1.0;
  }
  else
  {
    use_net_bump_height = 0.0;
  }

  mu_w  = physical_params.get<double>("Water Viscosity");
  rho_i = physical_params.get<double>("Ice Density");
  rho_w = physical_params.get<double>("Water Density");
  g     = physical_params.get<double>("Gravity Acceleration");

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
  this->utils.setFieldData(z_s,fm);
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(omega,fm);

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
      rhs(cell,qp) = omega(cell,qp) - u_b(cell,qp)*(h_b - use_net_bump_height*h(cell,qp))/l_b
                     + rho_i*g*h(cell,qp)*H(cell,qp)/mu_i(cell)
                     + rho_w*g*h(cell,qp)*(z_s(cell,qp)-H(cell,qp))/mu_i(cell);
    }
  }
}

} // Namespace FELIX
