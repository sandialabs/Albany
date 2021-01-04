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
    numPts = layout->extent(2);
  } else {
    numPts = layout->extent(1);
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
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end()) {
    return;
  }

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  if (Surrogate) {
    ParamScalarT alpha = Albany::convertScalar<const ParamScalarT>(alphaParam(0));

#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    if (std::fabs(printedAlpha-alpha)>1e-10) {
      *output << "[Effective Pressure<" << PHX::print<EvalT>() << ">]] alpha = " << alpha << "\n";
      printedAlpha = alpha;
    }
#endif

    for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s) {
      // Get the local data of side and cell
      const int cell = iter_s->elem_LID;
      const int side = iter_s->side_local_id;

      for (unsigned int pt=0; pt<numPts; ++pt) {
        // N = P_o-P_w
        N (cell,side,pt) = (1-alpha)*P_o(cell,side,pt);
      }
    }
  } else {
    for (const auto& it : sideSet) {
      // Get the local data of side and cell
      const int cell = it.elem_LID;
      const int side = it.side_local_id;

      for (unsigned int node=0; node<numPts; ++node) {
        // N = P_o - P_w
        N (cell,side,node) = P_o(cell,side,node) - P_w(cell,side,node);
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
