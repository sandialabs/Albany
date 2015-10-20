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
HydrologyHydrostaticPotential<EvalT, Traits>::
HydrologyHydrostaticPotential (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  if (p.isParameter("stokes_coupling"))
  {
    stokes_coupling = p.get<bool>("stokes_coupling");
  }
  else
  {
    stokes_coupling = false;
  }

  if (stokes_coupling)
  {
    sideSetNames = *p.get<std::set<std::string>*>("Side Set Names");

    H     = PHX::MDField<ScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), dl->side_node_scalar);
    z_s   = PHX::MDField<ScalarT>(p.get<std::string> ("Surface Height Variable Name"), dl->side_node_scalar);
    phi_H = PHX::MDField<ScalarT>(p.get<std::string> ("Hydrostatic Potential Variable Name"),dl->side_node_scalar);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->side_node_scalar->dimensions(dims);
    numNodes   = dims[2];
  }
  else
  {
    H     = PHX::MDField<ScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), dl->node_scalar);
    z_s   = PHX::MDField<ScalarT>(p.get<std::string> ("Surface Height Variable Name"), dl->node_scalar);
    phi_H = PHX::MDField<ScalarT>(p.get<std::string> ("Hydrostatic Potential Variable Name"),dl->node_scalar);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_scalar->dimensions(dims);
    numNodes   = dims[1];
  }

  this->addDependentField(H);
  this->addDependentField(z_s);

  this->addEvaluatedField(phi_H);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_i = physics.get<double>("Ice Density");
  rho_w = physics.get<double>("Water Density");
  g     = physics.get<double>("Gravity Acceleration");

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
  if (!stokes_coupling)
  {
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int node=0; node < numNodes; ++node)
      {
        phi_H(cell,node) = rho_i*g*H(cell,node) + rho_w*g*(z_s(cell,node) - H(cell,node));
      }
    }
  }
  else
  {
    for (std::set<std::string>::const_iterator it_names=sideSetNames.begin(); it_names!=sideSetNames.end(); ++it_names)
    {
      const Albany::SideSetList& ssList = *(workset.sideSets);
      Albany::SideSetList::const_iterator it_ss = ssList.find(*it_names);

      if (it_ss==ssList.end())
        continue;

      const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
      std::vector<Albany::SideStruct>::const_iterator iter_s;
      for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
      {
        // Get the local data of side and cell
        const int cell = iter_s->elem_LID;
        const int side = iter_s->side_local_id;

        for (int node=0; node < numNodes; ++node)
        {
          phi_H(cell,side,node) = rho_i*g*H(cell,side,node) + rho_w*g*(z_s(cell,side,node) - H(cell,side,node));
        }
      }
    }
  }
}

} // Namespace FELIX
