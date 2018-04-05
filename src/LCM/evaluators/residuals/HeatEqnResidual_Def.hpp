//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"

namespace LCM {
  //
  //
  //
template <typename EvalT, typename Traits>
HeatEqnResidual<EvalT, Traits>::
HeatEqnResidual(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : wBF(  // dependent
        p.get<std::string>("Weighted BF Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout")),
      wGradBF(  // dependent
        p.get<std::string>("Weighted Gradient BF Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      Temperature(  // dependent
        p.get<std::string>("QP Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      Tdot(  // dependent
        p.get<std::string>("QP Time Derivative Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      TGrad(  // dependent
        p.get<std::string>("QP Gradient Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      thermal_conductivity_(  // dependent
        p.get<std::string>("ACE Thermal Conductivity"), dl->qp_scalar),
      thermal_inertia_(  // dependent
        p.get<std::string>("ACE Thermal Inertia"), dl->qp_scalar),
      TResidual(  // evaluated
        p.get<std::string>("Residual Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout")) 
{
  Teuchos::ParameterList* heatEqnResidual_list =
    p.get<Teuchos::ParameterList*>("Parameter List");
    
  // List dependent fields
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(Temperature);
  this->addDependentField(Tdot);
  this->addDependentField(TGrad);

  this->addDependentField(thermal_conductivity_);
  this->addDependentField(thermal_inertia_);
  
  // List evaluated field
  this->addEvaluatedField(TResidual);

  Teuchos::RCP<PHX::DataLayout>
  vector_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");

  std::vector<PHX::DataLayout::size_type>
  dims;

  vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs = dims[2];
  numDims = dims[3];

  this->setName("HeatEqnResidual");
}

  //
  //
  //
template <typename EvalT, typename Traits>
void
HeatEqnResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits> &fm)
{
  // List all fields
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(Temperature, fm);
  this->utils.setFieldData(Tdot, fm);
  this->utils.setFieldData(TGrad, fm);

  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(thermal_inertia_, fm);

  this->utils.setFieldData(TResidual, fm);

  // Allocate workspace:
  heat_flux_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs, numDims);
  accumulation_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);

  return;
}

  //
  //
  //
template <typename EvalT, typename Traits>
void
HeatEqnResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  using FST = Intrepid2::FunctionSpaceTools<PHX::Device>;
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      // heat flux term:
      heat_flux_(cell, qp) = 0.0;
      for (std::size_t i=0; i < numDims; ++i) {
        heat_flux_(cell, qp) += 
          thermal_conductivity_(cell, qp) * TGrad(cell, qp, i);
      }
      // accumulation term:
      accumulation_(cell,qp) = 
        thermal_inertia_(cell, qp) * Tdot(cell, qp);
    }
  }
  
  FST::integrate(
    TResidual.get_view(),
    heat_flux_, wGradBF.get_view(), 
    false); // "false" overwrites

  FST::integrate(
    TResidual.get_view(),
    accumulation_, wBF.get_view(), 
    true); // "true" sums into

  return;
}



} // namespace LCM
