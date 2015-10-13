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
EffectivePressureSurrogate<EvalT, Traits>::EffectivePressureSurrogate (const Teuchos::ParameterList& p,
                                           const Teuchos::RCP<Albany::Layouts>& dl) :
  H   (p.get<std::string> ("Ice Thickness QP Variable Name"), dl->side_qp_scalar),
  z_s (p.get<std::string> ("Surface Height QP Variable Name"), dl->side_qp_scalar),
  N   (p.get<std::string> ("Effective Pressure QP Variable Name"),dl->side_qp_scalar)
{
  this->addDependentField(H);
  this->addDependentField(z_s);

  this->addEvaluatedField(N);

  // Setting parameters
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("Physical Parameters");
  rho_i = physical_params.get<double>("Ice Density", 910.0);
  rho_w = physical_params.get<double>("Water Density", 1028.0);
  g     = physical_params.get<double>("Gravity Acceleration", 9.8);
  alpha = p.get<double>("Hydraulic-Over-Hydrostatic Potential Ratio");

  numSideQPs = dl->side_qp_scalar->dimension(2);

  basalSideName = p.get<std::string>("Side Set Name");

  this->setName("EffectivePressureSurrogate"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressureSurrogate<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(z_s,fm);

  this->utils.setFieldData(N,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressureSurrogate<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
  {
    // Get the local data of side and cell
    const int cell = iter_s->elem_LID;
    const int side = iter_s->side_local_id;

    for (int qp=0; qp<numSideQPs; ++qp)
    {
      N (cell,side,qp) = std::max(rho_w*g*(z_s(cell,side,qp)-H(cell,side,qp)) + (1-alpha)*rho_i*g*H(cell,side,qp),0.0);
    }
  }
}

} // Namespace FELIX
