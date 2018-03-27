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
      density_(  // dependent
        p.get<std::string>("ACE Density"), dl->qp_scalar),
      heat_capacity_(  // dependent
        p.get<std::string>("ACE Heat Capacity"), dl->qp_scalar),
      melting_temperature_(  // dependent
        p.get<std::string>("ACE Melting Temperature"), dl->qp_scalar),
      porosity_(  // dependent
        p.get<std::string>("ACE Porosity"), dl->qp_scalar),
      pressure_(  // dependent
        p.get<std::string>("QP Pressure Variable Name"), dl->qp_scalar),
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
    
  // Read heat equation parameter values
  rho_ice_ = heatEqnResidual_list->get<double>("ACE Ice Density");
  latent_heat_ = 
    heatEqnResidual_list->get<double>("ACE Latent Heat of Phase Change");

  // List dependent fields
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(Temperature);
  this->addDependentField(Tdot);
  this->addDependentField(TGrad);

  this->addDependentField(density_);
  this->addDependentField(heat_capacity_);
  this->addDependentField(melting_temperature_);
  this->addDependentField(porosity_);
  this->addDependentField(pressure_);
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

  this->utils.setFieldData(density_, fm);
  this->utils.setFieldData(heat_capacity_, fm);
  this->utils.setFieldData(melting_temperature_, fm);
  this->utils.setFieldData(porosity_, fm);
  this->utils.setFieldData(pressure_, fm);
  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(thermal_inertia_, fm);

  this->utils.setFieldData(TResidual, fm);

  // Allocate workspace:
  heat_flux_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs, numDims);
  accumulation_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  Temperature_old_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  delTemp_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  Tmelt_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  dfdT_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  f_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  w_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  f_old_ = Kokkos::createDynRankView(
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
  
  // update saturations and thermal properties:
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      // the order these are called is important:
      updateTemperatureChange(cell,qp);
      update_dfdT(cell,qp);
      updateSaturations(cell,qp);
    }
  }
  
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

  //
  // Updates the change in temperature since last time step.
  //
template <typename EvalT, typename Traits>
void HeatEqnResidual<EvalT, Traits>::
updateTemperatureChange(std::size_t cell, std::size_t qp) 
{
  delTemp_(cell,qp) = Temperature(cell,qp) - Temperature_old_(cell,qp);
  
  return;
}

  //
  // Updates the freezing curve slope.
  //
template <typename EvalT, typename Traits>
void HeatEqnResidual<EvalT, Traits>::
update_dfdT(std::size_t cell, std::size_t qp) 
{
  ScalarT
  f_evaluated = 0.0;
  
  f_evaluated = evaluateFreezingCurve(cell, qp);
  dfdT_(cell, qp) = (f_evaluated - f_old_(cell, qp)) / delTemp_(cell, qp);
  
  // swap old and new temperatures now:
  Temperature_old_(cell,qp) = Temperature(cell,qp);

  return;
}

  //
  // Updates the ice and water saturations.
  //
template <typename EvalT, typename Traits>
void HeatEqnResidual<EvalT, Traits>::
updateSaturations(std::size_t cell, std::size_t qp) 
{
  f_(cell,qp) += dfdT_(cell,qp) * delTemp_(cell,qp);
  w_(cell,qp) -= dfdT_(cell,qp) * delTemp_(cell,qp);
  
  // check on realistic bounds:
  f_(cell,qp) = std::max(0.0,f_(cell,qp));
  f_(cell,qp) = std::min(1.0,f_(cell,qp));
  w_(cell,qp) = std::max(0.0,w_(cell,qp));
  w_(cell,qp) = std::min(1.0,w_(cell,qp));
  
  // swap old and new saturations now:
  f_old_(cell,qp) = f_(cell,qp);

  return;
}

  //
  // Calculates ice saturation given a temperature from the soil freezing curve.
  //
template <typename EvalT, typename Traits>
typename EvalT::ScalarT 
HeatEqnResidual<EvalT, Traits>::
evaluateFreezingCurve(std::size_t cell, std::size_t qp) 
{
  ScalarT
  f_evaluated = 0.0;  // ice saturation
  
  ScalarT
  T_range = 1.0;  // temperature range over which phase change occurs
  
  ScalarT
  T_low = Tmelt_(cell,qp) - (T_range/2.0);
  
  ScalarT
  T_high = Tmelt_(cell,qp) + (T_range/2.0);
  
  // completely frozen
  if (Temperature(cell,qp) <= T_low) {
    f_evaluated = 1.0;
  }
  // completely melted
  if (Temperature(cell,qp) >= T_high) {
    f_evaluated = 0.0;
  }
  // in phase change
  if ((Temperature(cell,qp) > T_low) && (Temperature(cell,qp) < T_high)) {
    f_evaluated = -1.0*(Temperature(cell,qp)/T_range) + T_high;
  }
  // Note: The freezing curve is a simple linear relationship that is sharp
  // at the T_low and T_high points. I don't know if this will actually cause
  // problems or not. If it does, we can try a curved relationship.
    
  return f_evaluated;
}

} // namespace LCM
