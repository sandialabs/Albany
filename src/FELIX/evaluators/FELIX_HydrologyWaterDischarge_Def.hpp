//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

//**********************************************************************
template<typename EvalT, typename Traits>
HydrologyWaterDischarge<EvalT, Traits>::
HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl)
{
  if (p.isParameter("Stokes Coupling"))
  {
    stokes_coupling = p.get<bool>("Stokes Coupling");
  }
  else
  {
    stokes_coupling = false;
  }

  if (stokes_coupling)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    sideSetName = p.get<std::string>("Side Set Name");

    numQPs  = dl->qp_gradient->dimension(2);
    numDim  = dl->qp_gradient->dimension(3);
  }
  else
  {
    numQPs  = dl->qp_gradient->dimension(1);
    numDim  = dl->qp_gradient->dimension(2);
  }

  gradPhi = PHX::MDField<ScalarT>(p.get<std::string> ("Hydraulic Potential Gradient QP Variable Name"), dl->qp_gradient);
  h       = PHX::MDField<ScalarT>(p.get<std::string> ("Drainage Sheet Depth QP Variable Name"), dl->qp_scalar);
  q       = PHX::MDField<ScalarT>(p.get<std::string> ("Water Discharge QP Variable Name"), dl->qp_gradient);

  this->addDependentField(gradPhi);
  this->addDependentField(h);

  this->addEvaluatedField(q);

  // Setting parameters
  Teuchos::ParameterList& hydrology = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  Teuchos::ParameterList& physics   = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  mu_w = physics.get<double>("Water Viscosity");
  k    = hydrology.get<double>("Transmissivity");

  this->setName("HydrologyWaterDischarge"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(gradPhi,fm);
  this->utils.setFieldData(h,fm);

  this->utils.setFieldData(q,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydrologyWaterDischarge<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (!stokes_coupling)
  {
    // q = - \frac{k h^3 \nabla (phiH-N)}{\mu_w}
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
          q(cell,qp,dim) = -k * std::pow(h(cell,qp),3) * gradPhi(cell,qp,dim) / mu_w;
        }
      }
    }
  }
  else
  {
    if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
      return;

    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
          q(cell,qp,dim) = -k * std::pow(h(cell,side,qp),3) * gradPhi(cell,side,qp,dim) / mu_w;
        }
      }
    }
  }
}

} // Namespace FELIX
