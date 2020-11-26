//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_HydraulicPotential.hpp"

#include "Albany_DiscretizationUtils.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, bool IsStokes>
HydraulicPotential<EvalT, Traits, IsStokes>::
HydraulicPotential (const Teuchos::ParameterList& p,
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
    numPts = layout->extent(2);
  } else {
    numPts = layout->extent(1);
  }

  P_w   = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Pressure Variable Name"), layout);
  phi_0 = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Basal Gravitational Water Potential Variable Name"), layout);
  phi   = PHX::MDField<ScalarT>(p.get<std::string> ("Hydraulic Potential Variable Name"), layout);

  this->addDependentField (P_w);
  this->addDependentField (phi_0);

  this->addEvaluatedField (phi);

  use_h = false;
  Teuchos::ParameterList& hydro_params = *p.get<Teuchos::ParameterList*>("LandIce Hydrology");
  if (hydro_params.get<bool>("Use Water Thickness In Effective Pressure Formula",false)) {
    use_h = true;

    h = PHX::MDField<const ScalarT>(p.get<std::string> ("Water Thickness Variable Name"), layout);
    this->addDependentField(h);

    // Setting parameters
    Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

    rho_w = physics.get<double>("Water Density",1000);
    g     = physics.get<double>("Gravity Acceleration",9.8);
  }

  this->setName("HydraulicPotential"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydraulicPotential<EvalT, Traits, IsStokes>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(P_w,fm);
  this->utils.setFieldData(phi_0,fm);
  if (use_h) {
    this->utils.setFieldData(h,fm);
  }
  this->utils.setFieldData(phi,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydraulicPotential<EvalT, Traits, IsStokes>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes>
void HydraulicPotential<EvalT, Traits, IsStokes>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end()) {
    return;
  }

  ScalarT zero(0.0);
  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  for (const auto& it : sideSet) {
    // Get the local data of side and cell
    const int cell = it.elem_LID;
    const int side = it.side_local_id;

    for (unsigned int pt=0; pt<numPts; ++pt) {
      // Recall that phi is in kPa, but h is in m. Need to convert to km.
      phi(cell,side,pt) = P_w(cell,side,pt) + phi_0(cell,side,pt) + (use_h ? rho_w*g*h(cell,side,pt)/1000 : zero);
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void HydraulicPotential<EvalT, Traits, IsStokes>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT zero(0.0);
  for (unsigned int cell=0; cell<workset.numCells; ++cell) {
    for (unsigned int pt=0; pt<numPts; ++pt) {
      // Recall that phi is in kPa, but h is in m. Need to convert to km.
      phi(cell,pt) = P_w(cell,pt) + phi_0(cell,pt) + (use_h ? rho_w*g*h(cell,pt)/1000 : zero);
    }
  }
}

} // Namespace LandIce
