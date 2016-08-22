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
// PARTIAL SPECIALIZATION: Hydrology ***********************************
//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn>
HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, false>::
HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  gradPhi     (p.get<std::string> ("Hydraulic Potential Gradient QP Variable Name"), dl->qp_gradient),
  h           (p.get<std::string> ("Water Thickness QP Variable Name"), dl->qp_scalar),
  q           (p.get<std::string> ("Water Discharge QP Variable Name"), dl->qp_gradient)
{
  numQPs  = dl->qp_gradient->dimension(1);
  numDim  = dl->qp_gradient->dimension(2);

  this->addDependentField(gradPhi);
  this->addDependentField(h);

  this->addEvaluatedField(q);

  // Setting parameters
  Teuchos::ParameterList& hydrology = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  Teuchos::ParameterList& physics   = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  mu_w  = physics.get<double>("Water Viscosity");
  k_0   = hydrology.get<double>("Transmissivity");
  alpha = hydrology.get<double>("Water Thickness Exponent (alpha)",3);
  beta  = hydrology.get<double>("Potential Gradient Exponent (beta)",2) - 2.0;

  // The mesh is assumed to be in km, therefore we need to adjust the units
  // of divergence and gradient, since all other distances are in SI units.
  // We do so by simply rescaling k_0
  k_0   *= 1e-6;
/*
  needsGradPhiNorm = false;
  if (beta!=0.0)
  {
    needsGradPhiNorm = true;
    gradPhiNorm = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string>("Hydraulic Potential Gradient Norm QP Variable Name"), dl->qp_scalar);
    this->addDependentField(gradPhiNorm);
  }
*/
  this->setName("HydrologyWaterDischarge"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, false>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(gradPhi,fm);
  this->utils.setFieldData(h,fm);
/*
  if (needsGradPhiNorm)
  {
    this->utils.setFieldData(gradPhiNorm,fm);
  }
*/
  this->utils.setFieldData(q,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, false>::evaluateFields (typename Traits::EvalData workset)
{
/*
  // q = - \frac{k h^3 \nabla (phiH-N)}{\mu_w}
  if (needsGradPhiNorm)
  {
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
          q(cell,qp,dim) = -k_0 * std::pow(h(cell,qp),alpha) * std::pow(gradPhiNorm(cell,qp),beta) * gradPhi(cell,qp,dim) / mu_w;
        }
      }
    }
  }
  else
*/
  {
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
//          q(cell,qp,dim) = -k_0 * std::pow(h(cell,qp),alpha) * gradPhi(cell,qp,dim) / mu_w;
          q(cell,qp,dim) = -k_0 * std::pow(h(cell,qp),3) * gradPhi(cell,qp,dim) / mu_w;
        }
      }
    }
  }
}

//**********************************************************************
// PARTIAL SPECIALIZATION: StokesFOHydrology ***************************
//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn>
HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, true>::
HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  gradPhi     (p.get<std::string> ("Hydraulic Potential Gradient QP Variable Name"), dl->qp_gradient),
  h           (p.get<std::string> ("Water Thickness QP Variable Name"), dl->qp_scalar),
  q           (p.get<std::string> ("Water Discharge QP Variable Name"), dl->qp_gradient)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layout structure does not appear to be that of a side set.\n");

  sideSetName = p.get<std::string>("Side Set Name");

  numQPs  = dl->qp_gradient->dimension(2);
  numDim  = dl->qp_gradient->dimension(3);

  this->addDependentField(gradPhi);
  this->addDependentField(h);

  this->addEvaluatedField(q);

  // Setting parameters
  Teuchos::ParameterList& hydrology = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  Teuchos::ParameterList& physics   = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  mu_w  = physics.get<double>("Water Viscosity");
  k_0   = hydrology.get<double>("Transmissivity");
  alpha = hydrology.get<double>("Water Thickness Exponent (alpha)",3);
  beta  = hydrology.get<double>("Potential Gradient Exponent (beta)",2) - 2.0;
/*
  needsGradPhiNorm = false;
  if (beta!=0.0)
  {
    needsGradPhiNorm = true;
    gradPhiNorm = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string>("Hydraulic Potential Gradient Norm QP Variable Name"), dl->qp_scalar);
    this->addDependentField(gradPhiNorm);
  }
*/
  this->setName("HydrologyWaterDischarge"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, true>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(gradPhi,fm);
  this->utils.setFieldData(h,fm);
/*
  if (needsGradPhiNorm)
  {
    this->utils.setFieldData(gradPhiNorm,fm);
  }
*/
  this->utils.setFieldData(q,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, true>::
evaluateFields (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
/*
  if (needsGradPhiNorm)
  {
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
          q(cell,side,qp,dim) = -k_0 * std::pow(h(cell,side,qp),alpha) * std::pow(gradPhiNorm(cell,side,qp),beta) * gradPhi(cell,side,qp,dim) / mu_w;
        }
      }
    }
  }
  else
*/
  {
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
          q(cell,side,qp,dim) = -k_0 * std::pow(h(cell,side,qp),alpha) * gradPhi(cell,side,qp,dim) / mu_w;
        }
      }
    }
  }
}

} // Namespace FELIX
