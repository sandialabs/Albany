//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "LandIce_EffectivePressure.hpp"
#include "LandIce_ParamEnum.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_SacadoTypes.hpp"

#include "PHAL_Dimension.hpp"
//uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

namespace LandIce {

template<typename EvalT, typename Traits, bool IsStokes, bool Surrogate>
EffectivePressure<EvalT, Traits, IsStokes, Surrogate>::
EffectivePressure (const Teuchos::ParameterList& p,
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

  if (Surrogate) {
    // P_w is set to a percentage of the overburden
    alphaParam = PHX::MDField<const ScalarT,Dim> (ParamEnumName::Alpha,dl->shared_param);
    this->addDependentField (alphaParam);

    printedAlpha = -1.0;
  } else {
    P_w  = PHX::MDField<const HydroScalarT>(p.get<std::string> ("Water Pressure Variable Name"), layout);
    this->addDependentField (P_w);
  }

  P_o = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Ice Overburden Variable Name"), layout);
  N   = PHX::MDField<HydroScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
  this->addDependentField (P_o);
  this->addEvaluatedField (N);

  this->setName("EffectivePressure"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes, bool Surrogate>
void EffectivePressure<EvalT, Traits, IsStokes, Surrogate>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(P_o,fm);
  this->utils.setFieldData(N,fm);

  if (Surrogate) {
    this->utils.setFieldData(alphaParam,fm);
  } else {
    this->utils.setFieldData(P_w,fm);
  }
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, bool IsStokes, bool Surrogate>
KOKKOS_INLINE_FUNCTION
void EffectivePressure<EvalT, Traits, IsStokes, Surrogate>::
operator() (const Surrogate_Tag& tag, const int& sideSet_idx) const {

  const ParamScalarT alpha = Albany::convertScalar<const ParamScalarT>(alphaParam(0));

  for (int pt=0; pt<numPts; ++pt) {
    // N = P_o-P_w
    N (sideSet_idx,pt) = (1-alpha)*P_o(sideSet_idx,pt);
  }

}

template<typename EvalT, typename Traits, bool IsStokes, bool Surrogate>
KOKKOS_INLINE_FUNCTION
void EffectivePressure<EvalT, Traits, IsStokes, Surrogate>::
operator() (const NonSurrogate_Tag& tag, const int& sideSet_idx) const {

  for (int node=0; node<numPts; ++node) {
    // N = P_o - P_w
    N (sideSet_idx,node) = P_o(sideSet_idx,node) - P_w(sideSet_idx,node);
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsStokes, bool Surrogate>
void EffectivePressure<EvalT, Traits, IsStokes, Surrogate>::
evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool Surrogate>
void EffectivePressure<EvalT, Traits, IsStokes, Surrogate>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;

  sideSet = workset.sideSetViews->at(basalSideName);

  if (Surrogate) {

    if (useCollapsedSidesets) {
      Kokkos::parallel_for(Surrogate_Policy(0, sideSet.size), *this);
    } else {
      ParamScalarT alpha = Albany::convertScalar<const ParamScalarT>(alphaParam(0));

      for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
      {
        // Get the local data of side and cell
        const int cell = sideSet.elem_LID(sideSet_idx);
        const int side = sideSet.side_local_id(sideSet_idx);

        for (int pt=0; pt<numPts; ++pt) {
          // N = P_o-P_w
          N (cell,side,pt) = (1-alpha)*P_o(cell,side,pt);
        }
      }
    }

  } else {
    if (useCollapsedSidesets) {
      Kokkos::parallel_for(NonSurrogate_Policy(0, sideSet.size), *this);
    } else {
      for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
      {
        // Get the local data of side and cell
        const int cell = sideSet.elem_LID(sideSet_idx);
        const int side = sideSet.side_local_id(sideSet_idx);

        for (int node=0; node<numPts; ++node) {
          // N = P_o - P_w
          N (cell,side,node) = P_o(cell,side,node) - P_w(cell,side,node);
        }
      }
    }
      
  }
}

template<typename EvalT, typename Traits, bool IsStokes, bool Surrogate>
void EffectivePressure<EvalT, Traits, IsStokes, Surrogate>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  if (Surrogate)
  {
    ParamScalarT alpha = Albany::convertScalar<const ParamScalarT>(alphaParam(0));

#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    if (std::fabs(printedAlpha-alpha)>1e-10) {
      *output << "[Effective Pressure " << PHX::print<EvalT>() << "] alpha = " << alpha << "\n";
      printedAlpha = alpha;
    }
#endif

    for (unsigned int cell=0; cell<workset.numCells; ++cell) {
      for (unsigned int node=0; node<numPts; ++node)
      {
        // N = P_o - P_w
        N (cell,node) = (1-alpha)*P_o(cell,node);
      }
    }
  } else {
    for (unsigned int cell=0; cell<workset.numCells; ++cell) {
      for (unsigned int node=0; node<numPts; ++node) {
        // N = P_o - P_w
        N(cell,node) = P_o(cell,node) - P_w(cell,node);
      }
    }
  }
}

} // Namespace LandIce
