//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, bool IsStokes>
IceOverburden<EvalT, Traits, IsStokes>::
IceOverburden (const Teuchos::ParameterList& p,
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

  H   = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
  P_o = PHX::MDField<ParamScalarT>(p.get<std::string> ("Ice Overburden Variable Name"), layout);

  this->addDependentField (H);

  this->addEvaluatedField (P_o);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_i = physics.get<double>("Ice Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("IceOverburden"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void IceOverburden<EvalT, Traits, IsStokes>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(P_o,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void IceOverburden<EvalT, Traits, IsStokes>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes>
void IceOverburden<EvalT, Traits, IsStokes>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end()) {
    return;
  }

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (const auto& it : sideSet) {
    // Get the local data of side and cell
    const int cell = it.elem_LID;
    const int side = it.side_local_id;

    for (int pt=0; pt<numPts; ++pt) {
      P_o (cell,side,pt) = rho_i*g*H(cell,side,pt);
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void IceOverburden<EvalT, Traits, IsStokes>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  for (int cell=0; cell<workset.numCells; ++cell) {
    for (int pt=0; pt<numPts; ++pt) {
      P_o (cell,pt) = rho_i*g*H(cell,pt);
    }
  }
}

} // Namespace LandIce
