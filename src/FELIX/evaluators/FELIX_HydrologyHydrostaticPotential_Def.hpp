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
HydrologyHydrostaticPotential<EvalT, Traits>::HydrologyHydrostaticPotential (const Teuchos::ParameterList& p,
                                                   const Teuchos::RCP<Albany::Layouts>& dl) :
  H     (p.get<std::string> ("Ice Thickness QP Variable Name"), dl->qp_scalar),
  z_s   (p.get<std::string> ("Surface Height QP Variable Name"), dl->qp_scalar),
  phi_H (p.get<std::string> ("Hydrostatic Potential QP Variable Name"),dl->qp_scalar)
{
  this->addDependentField(H);
  this->addDependentField(z_s);

  this->addEvaluatedField(phi_H);

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");

  rho_i = physical_params.get<double>("Ice Density", 910.0);
  rho_w = physical_params.get<double>("Water Density", 1028.0);
  g     = physical_params.get<double>("Gravity Acceleration",  9.8);

  std::cout << "HydrologyHydrostaticPotential:\n"
            << "    rho_i : " << rho_i << "\n"
            << "    rho_w : " << rho_w << "\n"
            << "    g     : " << g << "\n";

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_scalar->dimensions(dims);
  numQPs = dims[1];

  this->setName("HydrologyHydrostaticPotential"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyHydrostaticPotential<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(z_s,fm);

  this->utils.setFieldData(phi_H,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyHydrostaticPotential<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int qp=0; qp < numQPs; ++qp)
    {
      phi_H(cell,qp) = rho_i*g*H(cell,qp) + rho_w*g*(z_s(cell,qp) - H(cell,qp));
    }
  }
}

} // Namespace FELIX
