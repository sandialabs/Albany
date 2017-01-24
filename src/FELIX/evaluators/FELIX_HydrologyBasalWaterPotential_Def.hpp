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

template<typename EvalT, typename Traits>
BasalGravitationalWaterPotential<EvalT, Traits>::
BasalGravitationalWaterPotential (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  phi_0 (p.get<std::string> ("Basal Gravitational Water Potential Variable Name"), dl->node_scalar),
  z_s   (p.get<std::string> ("Surface Height Variable Name"), dl->node_scalar),
  H     (p.get<std::string> ("Ice Thickness Variable Name"), dl->node_scalar)
{
  stokes = p.get<bool>("Is Stokes");

  if (stokes)
  {
    basalSideName = p.get<std::string>("Side Set Name");
    numNodes = dl->node_scalar->dimension(2);
  }
  else
  {
    numNodes = dl->node_scalar->dimension(1);
  }

  this->addDependentField (z_s.fieldTag());
  this->addDependentField (H.fieldTag());

  this->addEvaluatedField (phi_0);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_w = physics.get<double>("Water Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("BasalGravitationalWaterPotential"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalGravitationalWaterPotential<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(z_s,fm);
  this->utils.setFieldData(H,fm);
  this->utils.setFieldData(phi_0,fm);
}
//**********************************************************************
template<typename EvalT, typename Traits>
void BasalGravitationalWaterPotential<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (stokes)
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
        phi_0 (cell,side,node) = rho_w*g*(z_s(cell,side,node) - H(cell,side,node));
      }
    }
  }
  else
  {
    for (int cell=0; cell<workset.numCells; ++cell)
    {
      for (int node=0; node<numNodes; ++node)
      {
        phi_0 (cell,node) = rho_w*g*(z_s(cell,node) - H(cell,node));
      }
    }
  }
}

} // Namespace FELIX
