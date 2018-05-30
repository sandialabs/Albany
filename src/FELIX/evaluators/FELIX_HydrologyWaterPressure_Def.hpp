//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

template<typename EvalT, typename Traits, bool IsStokes>
HydrologyWaterPressure<EvalT, Traits, IsStokes>::
HydrologyWaterPressure (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
{
  Teuchos::RCP<PHX::DataLayout> layout;
  if (p.isParameter("Nodal") && p.get<bool>("Nodal")) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }

  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numPts = layout->dimension(2);
  } else {
    numPts = layout->dimension(1);
  }

  z_s   = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Surface Height Variable Name"), layout);
  H     = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
  phi   = PHX::MDField<const ScalarT>(p.get<std::string> ("Hydraulic Potential Variable Name"), layout);
  P_w     = PHX::MDField<ScalarT>(p.get<std::string> ("Water Pressure Variable Name"), layout);

  this->addDependentField (z_s);
  this->addDependentField (H);
  this->addDependentField (phi);

  this->addEvaluatedField (P_w);

  use_h = false;
  Teuchos::ParameterList& hydro_params = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  if (hydro_params.get<bool>("Use Water Thickness In Effective Pressure Formula",false)) {
    use_h = true;

    h = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Thickness Variable Name"), layout);
    this->addDependentField(h);
  }

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_w = physics.get<double>("Water Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("HydrologyWaterPressure"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyWaterPressure<EvalT, Traits, IsStokes>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(z_s,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(phi,fm);
  this->utils.setFieldData(P_w,fm);
  if (use_h) {
    this->utils.setFieldData(h,fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyWaterPressure<EvalT, Traits, IsStokes>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyWaterPressure<EvalT, Traits, IsStokes>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end()) {
    return;
  }

  ScalarT zero(0.0);
  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (const auto& it : sideSet) {
    // Get the local data of side and cell
    const int cell = it.elem_LID;
    const int side = it.side_local_id;

    for (int pt=0; pt<numPts; ++pt) {
      P_w (cell,side,pt) = phi(cell,side,pt) - rho_w*g*( (z_s(cell,side,pt) - H(cell,side,pt)) + (use_h ? h(cell,side,pt) : zero) );
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydrologyWaterPressure<EvalT, Traits, IsStokes>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT zero(0.0);
  for (int cell=0; cell<workset.numCells; ++cell) {
    for (int pt=0; pt<numPts; ++pt) {
      P_w (cell,pt) = phi(cell,pt) - rho_w*g*( (z_s(cell,pt) - H(cell,pt)) + (use_h ? h(cell,pt)/1000 : zero) ) ;
    }
  }
}

} // Namespace FELIX
