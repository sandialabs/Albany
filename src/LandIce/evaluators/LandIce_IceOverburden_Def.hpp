//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_IceOverburden.hpp"

namespace LandIce {

template<typename EvalT, typename Traits>
IceOverburden<EvalT, Traits>::
IceOverburden (const Teuchos::ParameterList& p,
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

  H   = PHX::MDField<const RealType>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
  P_o = PHX::MDField<RealType>(p.get<std::string> ("Ice Overburden Variable Name"), layout);

  this->addDependentField (H);
  this->addEvaluatedField (P_o);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_i = physics.get<double>("Ice Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("IceOverburden"+PHX::print<EvalT>());
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void IceOverburden<EvalT, Traits>::
operator() (const IceOverburden_Tag&, const int& side_or_cell_idx) const
{
  for (unsigned int pt=0; pt<numPts; ++pt) {
    P_o (side_or_cell_idx,pt) = rho_i*g*H(side_or_cell_idx,pt);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void IceOverburden<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
    sideSet = workset.sideSetViews->at(sideSetName);
    Kokkos::parallel_for(IceOverburden_Policy(0, sideSet.size), *this);
  } else {
    Kokkos::parallel_for(IceOverburden_Policy(0, workset.numCells), *this);
  }
}

} // Namespace LandIce
