//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Teuchos_VerboseObject.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace LandIce {

template<typename EvalT, typename Traits, bool IsStokes>
BasalGravitationalWaterPotential<EvalT, Traits, IsStokes>::
BasalGravitationalWaterPotential (const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl)
{
  bool nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : true;
  Teuchos::RCP<PHX::DataLayout> layout = nodal ? dl->node_scalar : dl->qp_scalar;

  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numNodes = layout->extent(2);
  } else {
    numNodes = layout->extent(1);
  }

  z_s   = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Surface Height Variable Name"), layout);
  H     = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
  phi_0 = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Gravitational Water Potential Variable Name"), layout);

  this->addDependentField (z_s);
  this->addDependentField (H);

  this->addEvaluatedField (phi_0);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_w = physics.get<double>("Water Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("BasalGravitationalWaterPotential"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void BasalGravitationalWaterPotential<EvalT, Traits, IsStokes>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(z_s,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(phi_0,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void BasalGravitationalWaterPotential<EvalT, Traits, IsStokes>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes>
void BasalGravitationalWaterPotential<EvalT, Traits, IsStokes>::
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

    for (unsigned int node=0; node<numNodes; ++node) {
      phi_0 (cell,side,node) = rho_w*g*(z_s(cell,side,node) - H(cell,side,node));
    }
  }
}

template<typename EvalT, typename Traits, bool IsStokes>
void BasalGravitationalWaterPotential<EvalT, Traits, IsStokes>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  for (unsigned int cell=0; cell<workset.numCells; ++cell)
  {
    for (unsigned int node=0; node<numNodes; ++node)
    {
      phi_0 (cell,node) = rho_w*g*(z_s(cell,node) - H(cell,node));
    }
  }
}

} // Namespace LandIce
