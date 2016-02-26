//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
EffectivePressure<EvalT, Traits>::EffectivePressure (const Teuchos::ParameterList& p,
                                           const Teuchos::RCP<Albany::Layouts>& dl)
{
  bool surrogate = p.isParameter("Surrogate") ? p.get<bool>("Surrogate") : true;
  bool stokes = p.isParameter("Stokes") ? p.get<bool>("Stokes") : false;

  if (stokes)
  {
    phiH = PHX::MDField<ScalarT>(p.get<std::string> ("Hydrostatic Potential Variable Name"), dl->side_node_scalar);
    N    = PHX::MDField<ScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), dl->side_node_scalar);

    basalSideName = p.get<std::string>("Side Set Name");
    numNodes = dl->side_node_scalar->dimension(2);
  }
  else
  {
    phiH = PHX::MDField<ScalarT>(p.get<std::string> ("Hydrostatic Potential Variable Name"), dl->node_scalar);
    N    = PHX::MDField<ScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar);

    numNodes = dl->node_scalar->dimension(1);
  }

  if (surrogate)
  {
    alpha = p.get<double>("Hydraulic-Over-Hydrostatic Potential Ratio");

    TEUCHOS_TEST_FOR_EXCEPTION (alpha<0 || alpha>1, Teuchos::Exceptions::InvalidParameter,
                                "Error! 'Hydraulic-Over-Hydrostatic Potential Ratio' must be in [0,1].\n");
  }
  else
  {
    if (stokes)
      phi = PHX::MDField<ScalarT>(p.get<std::string> ("Hydraulic Potential Variable Name"), dl->side_node_scalar);
    else
      phi = PHX::MDField<ScalarT>(p.get<std::string> ("Hydraulic Potential Variable Name"), dl->node_scalar);

    this->addDependentField (phi);
  }

  this->addDependentField (phiH);
  this->addEvaluatedField (N);

  this->setName("EffectivePressure"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(phiH,fm);
  if (!surrogate)
    this->utils.setFieldData(phi,fm);

  this->utils.setFieldData(N,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void EffectivePressure<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (stokes)
  {
    const Albany::SideSetList& ssList = *(workset.sideSets);
    Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

    if (it_ss==ssList.end())
      return;

    const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
    std::vector<Albany::SideStruct>::const_iterator iter_s;

    if (surrogate)
    {
      for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
      {
        // Get the local data of side and cell
        const int cell = iter_s->elem_LID;
        const int side = iter_s->side_local_id;

        for (int node=0; node<numNodes; ++node)
        {
          N (cell,side,node) = (1-alpha)*phiH(cell,side,node);
        }
      }
    }
    else
    {
      for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
      {
        // Get the local data of side and cell
        const int cell = iter_s->elem_LID;
        const int side = iter_s->side_local_id;

        for (int node=0; node<numNodes; ++node)
        {
          N (cell,side,node) = phiH(cell,side,node) - phi (cell,side,node);
        }
      }
    }
  }
  else
  {
    if (surrogate)
    {
      for (int cell=0; cell<workset.numCells; ++cell)
      {
        for (int node=0; node<numNodes; ++node)
        {
          N(cell,node) = (1-alpha)*phiH(cell,node);
        }
      }
    }
    else
    {
      for (int cell=0; cell<workset.numCells; ++cell)
      {
        for (int node=0; node<numNodes; ++node)
        {
          N(cell,node) = phi(cell,node) - phiH(cell,node);
        }
      }
    }
  }
}

} // Namespace FELIX
