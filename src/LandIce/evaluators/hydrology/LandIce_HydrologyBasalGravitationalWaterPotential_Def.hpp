//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_HydrologyBasalGravitationalWaterPotential.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace LandIce {

template<typename EvalT, typename Traits>
BasalGravitationalWaterPotential<EvalT, Traits>::
BasalGravitationalWaterPotential (const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  Teuchos::RCP<PHX::DataLayout> layout;
  if (p.isParameter("Nodal") && p.get<bool>("Nodal")) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }

  numPts = layout->extent(1);

  z_s   = decltype(z_s)(p.get<std::string> ("Surface Height Variable Name"), layout);
  H     = decltype(H)(p.get<std::string> ("Ice Thickness Variable Name"), layout);
  phi_0 = decltype(phi_0)(p.get<std::string> ("Basal Gravitational Water Potential Variable Name"), layout);

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
template<typename EvalT, typename Traits>
void BasalGravitationalWaterPotential<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) return;
    sideSet = workset.sideSetViews->at(sideSetName);
    worksetSize = sideSet.size;
  } else {
    worksetSize = workset.numCells;
  }

  for (unsigned int cell=0; cell<worksetSize; ++cell) {
    evaluatePotential(cell);
  }
}

template<typename EvalT, typename Traits>
void BasalGravitationalWaterPotential<EvalT, Traits>::
evaluatePotential (unsigned int cell)
{
  for (int ipt=0; ipt<numPts; ++ipt) {
    phi_0 (cell,ipt) = rho_w*g*(z_s(cell,ipt) - H(cell,ipt));
  }
}

} // Namespace LandIce
