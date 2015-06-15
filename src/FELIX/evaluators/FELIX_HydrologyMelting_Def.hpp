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
HydrologyMelting<EvalT, Traits>::HydrologyMelting (const Teuchos::ParameterList& p,
                                                   const Teuchos::RCP<Albany::Layouts>& dl) :
  q       (p.get<std::string> ("Discharge QP Variable Name"), dl->qp_gradient),
  gradPhi (p.get<std::string> ("Hydraulic Potential Gradient QP Variable Name"), dl->qp_gradient),
  u_b     (p.get<std::string> ("Sliding Velocity Norm QP Variable Name"), dl->qp_scalar),
  beta    (p.get<std::string> ("Basal Friction Coefficient QP Variable Name"), dl->qp_scalar),
  G       (p.get<std::string> ("Geothermal Heat Source QP Variable Name"), dl->qp_scalar),
  m       (p.get<std::string> ("Melting Rate QP Variable Name"),dl->qp_scalar)
{
  this->addDependentField(q);
  this->addDependentField(gradPhi);
  this->addDependentField(u_b);
  this->addDependentField(beta);
  this->addDependentField(G);

  this->addEvaluatedField(m);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  nonlin_coeff = hydrology_params.get<double>("Nonlinearity Coefficient");

  mu_w = physical_params.get<double>("Water Viscosity");
  L    = physical_params.get<double>("Ice Latent Heat");
/*
  std::cout << "HydrologyMelting:\n"
            << "    mu_w : " << mu_w << "\n"
            << "    L    : " << L << "\n";
*/
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs = dims[1];
  numDim = dims[2];

  this->setName("HydrologyMelting"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMelting<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(q,fm);
  this->utils.setFieldData(gradPhi,fm);
  this->utils.setFieldData(u_b,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(G,fm);

  this->utils.setFieldData(m,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyMelting<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT prod;
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int qp=0; qp < numQPs; ++qp)
    {
      for (int dim(0); dim<numDim; ++dim)
        prod += q(cell,qp,dim)*gradPhi(cell,qp,dim);

      m(cell,qp) = (G(cell,qp) - beta(cell,qp) * u_b(cell,qp)*u_b(cell,qp) - nonlin_coeff * prod) / L;
    }
  }
}

} // Namespace FELIX
