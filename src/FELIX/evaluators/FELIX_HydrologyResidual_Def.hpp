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
  wBF         (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF     (p.get<std::string> ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  phi         (p.get<std::string> ("Hydraulic Potential QP Variable Name"), dl->qp_scalar),
  q           (p.get<std::string> ("Discharge QP Variable Name"), dl->qp_gradient),
  mu_i        (p.get<std::string> ("Ice Viscosity Variable Name"), dl->cell_scalar2),
  h           (p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar),
  m           (p.get<std::string> ("Melting Rate QP Variable Name"), dl->qp_scalar),
  constantRhs (p.get<std::string> ("RHS QP Name"), dl->qp_scalar),
  residual    (p.get<std::string> ("Residual Name"),dl->node_scalar)
{
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(phi);
  this->addDependentField(q);
  this->addDependentField(h);
  this->addDependentField(m);
  this->addDependentField(constantRhs);
  this->addDependentField(mu_i);

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  mu_w         = physical_params.get<double>("Water Viscosity");
  rho_i        = physical_params.get<double>("Ice Density");
  rho_w        = physical_params.get<double>("Water Density");

  if (hydrology_params.get<bool>("Has Melt Opening",false))
  {
    has_melt_opening = 1.0;
  }
  else
  {
    has_melt_opening = 0.0;
  }

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_gradient->dimensions(dims);
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
  this->utils.setFieldData(q,fm);
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(m,fm);
  this->utils.setFieldData(constantRhs,fm);
  this->utils.setFieldData(mu_i,fm);

  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyResidual<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT resRhs, resMass, resStiff;

  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int node=0; node < numNodes; ++node)
    {
      resRhs = resMass = resStiff = 0.;
      for (int qp=0; qp < numQPs; ++qp)
      {
        resRhs  += (m(cell,qp)/rho_w - has_melt_opening * m(cell,qp)/rho_i + constantRhs(cell,qp)) * wBF(cell,node,qp);

        resMass += h(cell,qp)*phi(cell,qp)*wBF(cell,node,qp)/mu_i(cell);

        for (int dim=0; dim<numDims; ++dim)
        {
          resStiff += q(cell,qp,dim) * wGradBF(cell,node,qp,dim);
        }
      }
      residual (cell,node) = resRhs - resMass - resStiff;
    }
  }
}

} // Namespace FELIX
