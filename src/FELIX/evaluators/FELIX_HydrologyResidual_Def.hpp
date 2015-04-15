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
HydrologyResidual<EvalT, Traits>::HydrologyResidual (const Teuchos::ParameterList& p,
                                                     const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF       (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF   (p.get<std::string> ("Weighted Gradient BF Name"), dl->node_qp_vector),
  phi       (p.get<std::string> ("Hydraulic Potential QP Variable Name"), dl->qp_scalar),
  gradPhi   (p.get<std::string> ("Hydraulic Potential Gradient QP Variable Name"), dl->qp_vector),
  mu_i      (p.get<std::string> ("Ice Viscosity QP Variable Name"), dl->qp_scalar),
  h         (p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar),
  rhs       (p.get<std::string> ("RHS QP Name"), dl->qp_scalar),
  residual  (p.get<std::string> ("Residual Name"),dl->node_scalar)
{
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(phi);
  this->addDependentField(gradPhi);
  this->addDependentField(h);
  this->addDependentField(rhs);
  this->addDependentField(mu_i);

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  k_0          = hydrology_params.get<double>("Permissivity");
  nonlin_coeff = hydrology_params.get<double>("Nonlinearity Coefficient");

  mu_w         = physical_params.get<double>("Water Viscosity");
  rho_i        = physical_params.get<double>("Ice Density");
  rho_w        = physical_params.get<double>("Water Density");
  rho_combo    = (rho_i - rho_w)/(rho_i*rho_w);
  L            = physical_params.get<double>("Ice Latent Heat");

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  this->setName("HydrologyResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(phi,fm);
  this->utils.setFieldData(gradPhi,fm);
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(rhs,fm);
  this->utils.setFieldData(mu_i,fm);

  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyResidual<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT gradPhiNormSq, resRhs, resMass, resNlin, resStiff;

  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int node=0; node < numNodes; ++node)
    {
      gradPhiNormSq = 0.;
      for (int i=0; i<numDims; i++)
      {
        gradPhiNormSq += pow(gradPhi(cell,node,i),2);
      }
      gradPhiNormSq *= nonlin_coeff;

      resRhs = resMass = resNlin = resStiff = 0.;
      for (int qp=0; qp < numQPs; ++qp)
      {
        resRhs  += rhs(cell,qp) * wBF(cell,node,qp);

        resMass += h(cell,qp)*phi(cell,qp)*wBF(cell,node,qp)/mu_i(cell,qp);

        resNlin += rho_combo*k_0*pow(h(cell,qp),3)*gradPhiNormSq/mu_w * wBF(cell,node,qp);

        for (int i=0; i<numDims; ++i)
        {
          resStiff += - k_0*pow(h(cell,qp),3)*gradPhi(cell,qp,i)/mu_w * wGradBF(cell,node,qp,i);
        }
      }
      residual (cell,node) = resRhs - resMass - resNlin - resStiff;
    }
  }
}

} // Namespace FELIX
