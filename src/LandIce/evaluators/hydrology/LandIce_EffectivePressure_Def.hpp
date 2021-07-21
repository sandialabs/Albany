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

namespace LandIce {

template<typename EvalT, typename Traits, bool Surrogate>
EffectivePressure<EvalT, Traits, Surrogate>::
EffectivePressure (const Teuchos::ParameterList& p,
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

  if (Surrogate) {
    // P_w is set to a percentage of the overburden
    alphaParam = PHX::MDField<const ScalarT,Dim> (ParamEnumName::Alpha,dl->shared_param);
    this->addDependentField (alphaParam);

    printedAlpha = -1.0;
  } else {
    P_w  = PHX::MDField<const HydroScalarT>(p.get<std::string> ("Water Pressure Variable Name"), layout);
    this->addDependentField (P_w);
  }

  P_o = PHX::MDField<const RealType>(p.get<std::string> ("Ice Overburden Variable Name"), layout);
  N   = PHX::MDField<HydroScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
  this->addDependentField (P_o);
  this->addEvaluatedField (N);

  this->setName("EffectivePressure"+PHX::print<EvalT>());
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, bool Surrogate>
KOKKOS_INLINE_FUNCTION
void EffectivePressure<EvalT, Traits, Surrogate>::
operator() (const EffectivePressure_Tag&, const int& cell) const
{
  if (Surrogate) {
    const ParamScalarT alpha = Albany::convertScalar<const ParamScalarT>(alphaParam(0));
    for (int pt=0; pt<numPts; ++pt) {
      // N = P_o-P_w
      N (cell,pt) = (1-alpha)*P_o(cell,pt);
    }
  } else {
    for (int node=0; node<numPts; ++node) {
      // N = P_o - P_w
      N (cell,node) = P_o(cell,node) - P_w(cell,node);
    }
  }
  
}

//**********************************************************************
template<typename EvalT, typename Traits, bool Surrogate>
void EffectivePressure<EvalT, Traits, Surrogate>::
evaluateFields (typename Traits::EvalData workset)
{

#ifdef OUTPUT_TO_SCREEN
    if (Surrogate) {
      Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
      ParamScalarT alpha = Albany::convertScalar<const ParamScalarT>(alphaParam(0));
      if (std::fabs(printedAlpha-alpha)>1e-10) {
        *output << "[Effective Pressure " << PHX::print<EvalT>() << "] alpha = " << alpha << "\n";
        printedAlpha = alpha;
      }
    }
#endif

  if (eval_on_side) {
    if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
    sideSet = workset.sideSetViews->at(sideSetName);
    worksetSize = sideSet.size;
  } else {
    worksetSize = workset.numCells;
  }

  Kokkos::parallel_for(EffectivePressure_Policy(0, worksetSize), *this);

}

} // Namespace LandIce
