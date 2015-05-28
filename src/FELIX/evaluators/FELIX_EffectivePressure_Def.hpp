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
EffectivePressure<EvalT, Traits>::EffectivePressure (const Teuchos::ParameterList& p,
                                           const Teuchos::RCP<Albany::Layouts>& dl) :
  phi       (p.get<std::string> ("Hydraulic Potential Variable Name"), dl->node_scalar),
  H         (p.get<std::string> ("Ice Thickness Variable Name"), dl->node_scalar),
  z_b       (p.get<std::string> ("Basal Height Variable Name"), dl->node_scalar),
  N         (p.get<std::string> ("Effective Pressure Name"),dl->node_scalar)
{
  this->addDependentField(phi);
  this->addDependentField(H);
  this->addDependentField(z_b);

  this->addEvaluatedField(N);

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  rho_i        = physical_params.get<double>("Ice Density");
  rho_w        = physical_params.get<double>("Water Density");
  g            = physical_params.get<double>("Gravity Acceleration");

std::cout << "rho_i = " << rho_i << "\n";
std::cout << "rho_w = " << rho_w << "\n";
std::cout << "g = " << g << "\n";

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_scalar->dimensions(dims);
  numNodes = dims[1];

  this->setName("EffectivePressure"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(phi,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(z_b,fm);

  this->utils.setFieldData(N,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int node=0; node < numNodes; ++node)
    {
      N (cell,node) = rho_w*g*z_b(cell,node) + rho_i*g*H(cell,node) - phi(cell,node);
    }
  }
}

} // Namespace FELIX
