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
  HeatEqnResidual<EvalT, Traits>::HeatEqnResidual(const Teuchos::ParameterList &p)
    : wBF(p.get<std::string>("Weighted BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout")),
      wGradBF(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      Temperature(
          p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      Tdot(p.get<std::string>("QP Time Derivative Variable Name"),
           p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      TGrad(p.get<std::string>("QP Gradient Variable Name"),
            p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      thermal_conductivity_(
          p.get<std::string>("QP Thermal Conductivity Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      density_(p.get<std::string>("QP Density Variable Name"),
           p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      specific_heat_(p.get<std::string>("QP Specific Heat Variable Name"),
           p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      pressure_(p.get<std::string>("QP Pressure Variable Name"),
           p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      salinity_(p.get<std::string>("QP Salinity Variable Name"),
           p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      TResidual(
          p.get<std::string>("Residual Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout")) {

  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(Temperature);
  this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(thermal_conductivity_);
  this->addDependentField(density_);
  this->addDependentField(specific_heat_);
  this->addDependentField(pressure_);
  this->addDependentField(salinity_);

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
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(Temperature, fm);
  this->utils.setFieldData(Tdot, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(density_, fm);
  this->utils.setFieldData(specific_heat_, fm);
  this->utils.setFieldData(pressure_, fm);
  this->utils.setFieldData(salinity_, fm);

  this->utils.setFieldData(TResidual, fm);

  // Allocate workspace
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

  // heat flux term:
  FST::scalarMultiplyDataData(
    heat_flux_,
    thermal_conductivity_.get_view(), 
    TGrad.get_view());

  FST::integrate(
    TResidual.get_view(),
    heat_flux_, wGradBF.get_view(), 
    false); // "false" overwrites
  
  // accumulation term:
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      accumulation_(cell,qp) = 0.0;
      accumulation_(cell,qp) += thermalInertia(cell,qp) * Tdot(cell,qp);
    }
  }

  FST::integrate(
    TResidual.get_view(),
    accumulation_, wBF.get_view(), 
    true); // "true" sums into

  return;
}

  //
  //
  //
template <typename EvalT, typename Traits>
typename EvalT::ScalarT
HeatEqnResidual<EvalT, Traits>::
meltingTemperature(std::size_t cell, std::size_t qp) {

  ScalarT
  melting_temperature = 0.0;

  return melting_temperature;
}

  //
  //
  //
template <typename EvalT, typename Traits>
typename EvalT::ScalarT
HeatEqnResidual<EvalT, Traits>::
thermalInertia(std::size_t cell, std::size_t qp) {

  ScalarT
  chi = 0.0;

  return chi;
}

} // namespace LCM
