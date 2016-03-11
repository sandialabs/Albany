//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyDischarge<EvalT, Traits>::HydrologyDischarge (const Teuchos::ParameterList& p,
                                                   const Teuchos::RCP<Albany::Layouts>& dl) :
  gradPhi (p.get<std::string> ("Hydraulic Potential Gradient QP Variable Name"), dl->qp_gradient),
  h       (p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar),
  q       (p.get<std::string> ("Discharge QP Variable Name"), dl->qp_gradient)
{
  this->addDependentField(h);
  this->addDependentField(gradPhi);

  this->addEvaluatedField(q);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  mu_w = physical_params.get<double>("Water Viscosity");
  k_0  = hydrology_params.get<double>("Transmissivity");
/*
  std::cout << "HydrologyDischarge:\n"
            << "    mu_w : " << mu_w << "\n"
            << "    k_0  : " << k_0 << "\n";
*/
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs = dims[1];
  numDim = dims[2];

  this->setName("HydrologyDischarge"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyDischarge<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(gradPhi,fm);

  this->utils.setFieldData(q,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyDischarge<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT gradPhiNormSq, resDischarge, resMass, resNlin, resStiff;
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int qp=0; qp < numQPs; ++qp)
    {
      for (int dim(0); dim<numDim; ++dim)
        q(cell,qp,dim) = -k_0 * std::pow(h(cell,qp),3) * gradPhi(cell,qp,dim) / mu_w;
    }
  }
}

} // Namespace FELIX
