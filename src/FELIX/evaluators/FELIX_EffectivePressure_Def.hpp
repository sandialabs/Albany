//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Teuchos_VerboseObject.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
// PARTIAL SPECIALIZATION: StokesFO ************************************
//**********************************************************************
template<typename EvalT, typename Traits>
EffectivePressure<EvalT, Traits, false, true>::
EffectivePressure (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
  N          (p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar)
{
  basalSideName = p.get<std::string>("Side Set Name");
  numNodes = dl->node_scalar->dimension(2);

  alphaParam = PHX::MDField<ScalarT,Dim> ("Hydraulic-Over-Hydrostatic Potential Ratio",dl->shared_param);
  this->addDependentField (alphaParam);

  regularized = p.get<Teuchos::ParameterList*>("Parameter List")->get("Regularize With Continuation",false);
  printedAlpha = -1.0;

  if (regularized)
    regularizationParam = PHX::MDField<ScalarT,Dim>(p.get<std::string>("Regularization Parameter Name"),dl->shared_param);

  H   = PHX::MDField<ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), dl->node_scalar);
  this->addDependentField (H);

  this->addEvaluatedField (N);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_i = physics.get<double>("Ice Density",910);
  rho_w = physics.get<double>("Water Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("EffectivePressure"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits, false, true>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(alphaParam,fm);
  if (regularized)
    this->utils.setFieldData(regularizationParam,fm);
  this->utils.setFieldData(N,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits, false, true>::
evaluateFields (typename Traits::EvalData workset)
{
  ParamScalarT alpha = Albany::ScalarConverter<ParamScalarT>::apply(alphaParam(0));
  if (regularized)
  {
    alpha = alpha*std::sqrt(Albany::ScalarConverter<ParamScalarT>::apply(regularizationParam(0)));
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (std::fabs(printedAlpha-alpha)>0.0001)
  {
    *output << "[Effective Pressure<" << PHX::typeAsString<EvalT>() << ">]] alpha = " << alpha << "\n";
    printedAlpha = alpha;
  }
#endif

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

    for (int node=0; node<numNodes; ++node)
    {
      // N = p_i-p_w
      // p_i = rho_i*g*H
      // p_w = alpha*p_i
      N (cell,side,node) = (1-alpha)*rho_i*g*H(cell,side,node);
    }
  }
}

//**********************************************************************
// PARTIAL SPECIALIZATION: StokesFOHydrology ***************************
//**********************************************************************
template<typename EvalT, typename Traits>
EffectivePressure<EvalT, Traits, true, true>::
EffectivePressure (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
  N          (p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar)
{
  basalSideName = p.get<std::string>("Side Set Name");
  numNodes = dl->node_scalar->dimension(2);

  z_s  = PHX::MDField<ParamScalarT>(p.get<std::string> ("Surface Height Variable Name"), dl->node_scalar);
  phi  = PHX::MDField<ScalarT>(p.get<std::string> ("Hydraulic Potential Variable Name"), dl->node_scalar);
  H    = PHX::MDField<ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), dl->node_scalar);

  this->addDependentField (phi);
  this->addDependentField (z_s);
  this->addDependentField (H);

  this->addEvaluatedField (N);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_i = physics.get<double>("Ice Density",910);
  rho_w = physics.get<double>("Water Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("EffectivePressure"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits, true, true>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(z_s,fm);
  this->utils.setFieldData(phi,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(N,fm);
}
//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits, true, true>::
evaluateFields (typename Traits::EvalData workset)
{
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;

  for (const auto& it : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it.elem_LID;
    const int side = it.side_local_id;

    for (int node=0; node<numNodes; ++node)
    {
      // N = p_i-p_w
      // p_i = rho_i*g*H
      // p_w = rho_w*g*z_b - phi
      N (cell,side,node) = rho_i*g*H(cell,side,node) + rho_w*g*(z_s(cell,side,node) - H(cell,side,node)) - phi (cell,side,node);
    }
  }
}

//**********************************************************************
// PARTIAL SPECIALIZATION: Hydrology ***********************************
//**********************************************************************
template<typename EvalT, typename Traits>
EffectivePressure<EvalT, Traits, true, false>::
EffectivePressure (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
  N          (p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar)
{
  numNodes = dl->node_scalar->dimension(1);

  z_s  = PHX::MDField<ParamScalarT>(p.get<std::string> ("Surface Height Variable Name"), dl->node_scalar);
  phi  = PHX::MDField<ScalarT>(p.get<std::string> ("Hydraulic Potential Variable Name"), dl->node_scalar);
  H   = PHX::MDField<ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), dl->node_scalar);

  this->addDependentField (phi);
  this->addDependentField (z_s);
  this->addDependentField (H);

  this->addEvaluatedField (N);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_i = physics.get<double>("Ice Density",910);
  rho_w = physics.get<double>("Water Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("EffectivePressure"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits, true, false>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(z_s,fm);
  this->utils.setFieldData(phi,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(N,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits, true, false>::evaluateFields (typename Traits::EvalData workset)
{
  for (int cell=0; cell<workset.numCells; ++cell)
  {
    for (int node=0; node<numNodes; ++node)
    {
      // N = p_i-p_w
      // p_i = rho_i*g*H
      // p_w = rho_w*g*z_b - phi
      N(cell,node) = rho_i*g*H(cell,node) + rho_w*g*(z_s(cell,node) - H(cell,node)) - phi(cell,node);
    }
  }
}

} // Namespace FELIX
