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
    : wBF(
        p.get<std::string>("Weighted BF Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout")),
      wGradBF(
        p.get<std::string>("Weighted Gradient BF Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      Temperature(
        p.get<std::string>("QP Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      Tdot(
        p.get<std::string>("QP Time Derivative Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      TGrad(
        p.get<std::string>("QP Gradient Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      pressure_(
        p.get<std::string>("QP Pressure Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      salinity_(
        p.get<std::string>("QP Salinity Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      porosity_(
        p.get<std::string>("QP Salinity Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      thermal_K_ice_(
        p.get<std::string>("QP Thermal Conductivity of Ice Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      thermal_K_water_(
        p.get<std::string>("QP Thermal Conductivity of Water Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      thermal_K_sed_(
        p.get<std::string>("QP Thermal Conductivity of Sediments Variable Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      TResidual(
        p.get<std::string>("Residual Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout")) {

  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(Temperature);
  this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(pressure_);
  this->addDependentField(salinity_);
  this->addDependentField(porosity_);
  this->addDependentField(thermal_K_ice_);
  this->addDependentField(thermal_K_water_);
  this->addDependentField(thermal_K_sed_);
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
  this->utils.setFieldData(pressure_, fm);
  this->utils.setFieldData(salinity_, fm);
  this->utils.setFieldData(porosity_, fm);
  this->utils.setFieldData(thermal_K_ice_, fm);
  this->utils.setFieldData(thermal_K_water_, fm);
  this->utils.setFieldData(thermal_K_sed_, fm);

  this->utils.setFieldData(TResidual, fm);

  // Allocate workspace:
  heat_flux_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs, numDims);
  accumulation_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  Tmelt_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  dfdT_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  f_ = Kokkos::createDynRankView(
    Temperature.get_view(), "XXX", worksetSize, numQPs);
  w_ = Kokkos::createDynRankView(
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
      updateMeltingTemperature(cell,qp);
      update_dfdT(cell,qp);
      updateSaturations(cell,qp);
    }
  }
  

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      // heat flux term:
      heat_flux_(cell,qp) = 0.0;
      for (std::size_t dims=0; dims < numDims; ++dims) {
        heat_flux_(cell,qp) += 
          thermalConductivity(cell,qp) * TGrad(cell,qp,dims);
      }
      // accumulation term:
      accumulation_(cell,qp) = thermalInertia(cell,qp) * Tdot(cell,qp);
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
  // Updates the local melting temperature [C].
  //
template <typename EvalT, typename Traits>
void HeatEqnResidual<EvalT, Traits>::
updateMeltingTemperature(std::size_t cell, std::size_t qp) 
{

  ScalarT    
  sal = 0.0;  // salinity in [ppt]
  sal = salinity_(cell,qp);
  
  ScalarT   
  press = 0.0;  // hydrostatic pressure in [Pa]
  press = pressure_(cell,qp);
  
  Tmelt_(cell,qp) = -0.0575*sal + 0.00170523*pow(sal,1.5)
    - 0.0002154996*pow(sal,2) - (0.000753/10000)*press;

  return;
}

  //
  // Updates the freezing curve slope.
  //
template <typename EvalT, typename Traits>
void HeatEqnResidual<EvalT, Traits>::
update_dfdT(std::size_t cell, std::size_t qp) 
{

  return;
}

  //
  // Updates the ice and water saturations.
  //
template <typename EvalT, typename Traits>
void HeatEqnResidual<EvalT, Traits>::
updateSaturations(std::size_t cell, std::size_t qp) 
{

  return;
}

  //
  // Calculates the thermal conductivity.
  //
template <typename EvalT, typename Traits>
typename EvalT::ScalarT 
HeatEqnResidual<EvalT, Traits>::
thermalConductivity(std::size_t cell, std::size_t qp) 
{
  
  ScalarT
  thermal_K = 0.0;  // this is isotropic for the moment
  
  thermal_K = pow(thermal_K_ice_(cell,qp),(f_(cell,qp)*porosity_(cell,qp))) *
              pow(thermal_K_water_(cell,qp),(w_(cell,qp)*porosity_(cell,qp))) *
              pow(thermal_K_sed_(cell,qp),(1.0-porosity_(cell,qp)));
  
  return thermal_K;
}

  //
  // Calculates the density.
  //
template <typename EvalT, typename Traits>
typename EvalT::ScalarT 
HeatEqnResidual<EvalT, Traits>::
density(std::size_t cell, std::size_t qp) 
{
  
  ScalarT
  density = 0.0;
  
  return density;
}

  //
  // Calculates the specific heat.
  //
template <typename EvalT, typename Traits>
typename EvalT::ScalarT 
HeatEqnResidual<EvalT, Traits>::
specificHeat(std::size_t cell, std::size_t qp) 
{
  
  ScalarT
  specific_heat = 0.0;
  
  return specific_heat;
}

  //
  // Calculates the thermal inertia term.
  //
template <typename EvalT, typename Traits>
typename EvalT::ScalarT
HeatEqnResidual<EvalT, Traits>::
thermalInertia(std::size_t cell, std::size_t qp) 
{

  ScalarT
  chi = 0.0;  
  
  ScalarT  // placeholder for now - should come from input deck material properties
  latent_heat = 334.0;  // latent heat of formation water/ice [kJ/kg-C] 
  
  ScalarT // placeholder for now - should come from input deck material properties
  rho_ice = 900.0;  // ice density in [kg/m3]
  
  chi = (density(cell,qp) * specificHeat(cell,qp)) - 
        (rho_ice * latent_heat * dfdT_(cell,qp));

  return chi;
}

} // namespace LCM
