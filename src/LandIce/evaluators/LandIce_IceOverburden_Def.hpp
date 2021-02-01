//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, bool IsStokes>
IceOverburden<EvalT, Traits, IsStokes>::
IceOverburden (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
{
  useCollapsedSidesets = (dl->isSideLayouts && dl->useCollapsedSidesets);

  Teuchos::RCP<PHX::DataLayout> layout;
  if (p.isParameter("Nodal") && p.get<bool>("Nodal")) {
    layout = useCollapsedSidesets ? dl->node_scalar_sideset : dl->node_scalar;
  } else {
    layout = useCollapsedSidesets ? dl->qp_scalar_sideset : dl->qp_scalar;
  }

  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numPts = useCollapsedSidesets ? layout->extent(1) : layout->extent(2);
  } else {
    numPts = useCollapsedSidesets ? layout->extent(0) : layout->extent(1);
  }

  H   = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
  P_o = PHX::MDField<ParamScalarT>(p.get<std::string> ("Ice Overburden Variable Name"), layout);

  this->addDependentField (H);

  this->addEvaluatedField (P_o);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_i = physics.get<double>("Ice Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("IceOverburden"+PHX::print<EvalT>());
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

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, bool IsStokes>
KOKKOS_INLINE_FUNCTION
void IceOverburden<EvalT, Traits, IsStokes>::
operator() (const IceOverburden_Tag& tag, const int& sideSet_idx) const {

  for (int pt=0; pt<numPts; ++pt) {
    P_o (sideSet_idx,pt) = rho_i*g*H(sideSet_idx,pt);
  }

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
  if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;
  
  sideSet = workset.sideSetViews->at(basalSideName);
  if (useCollapsedSidesets) {
    Kokkos::parallel_for(IceOverburden_Policy(0, sideSet.size), *this);
  } else {
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of side and cell
      const int cell = sideSet.elem_LID(sideSet_idx);
      const int side = sideSet.side_local_id(sideSet_idx);

      for (int pt=0; pt<numPts; ++pt) {
        P_o (cell,side,pt) = rho_i*g*H(cell,side,pt);
      }
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes>
void IceOverburden<EvalT, Traits, IsStokes>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  for (unsigned int cell=0; cell<workset.numCells; ++cell) {
    for (unsigned int pt=0; pt<numPts; ++pt) {
      P_o (cell,pt) = rho_i*g*H(cell,pt);
    }
  }
}

} // Namespace LandIce
