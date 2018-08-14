//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
ACEthermalConductivity<EvalT, Traits>::ACEthermalConductivity(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thermal_conductivity_(  // evaluated
          p.get<std::string>("ACE Thermal Conductivity"),
          dl->qp_scalar),
      porosity_(  // dependent
          p.get<std::string>("ACE Porosity"),
          dl->qp_scalar),
      ice_saturation_(  // dependent
          p.get<std::string>("ACE Ice Saturation"),
          dl->qp_scalar),
      water_saturation_(  // dependent
          p.get<std::string>("ACE Water Saturation"),
          dl->qp_scalar)
{
  Teuchos::ParameterList* thermal_conductivity_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read thermal conductivity values
  k_ice_ = thermal_conductivity_list->get<double>("Ice Value");
  k_wat_ = thermal_conductivity_list->get<double>("Water Value");
  k_sed_ = thermal_conductivity_list->get<double>("Sediment Value");

  // Add thermal conductivity as a Sacado-ized parameter
  this->registerSacadoParameter("ACE Thermal Conductivity", paramLib);

  // List evaluated fields
  this->addEvaluatedField(thermal_conductivity_);

  // List dependent fields
  this->addDependentField(porosity_);
  this->addDependentField(ice_saturation_);
  this->addDependentField(water_saturation_);

  this->setName("ACE Thermal Conductivity" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEthermalConductivity<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermal_conductivity_, fm);
  this->utils.setFieldData(porosity_, fm);
  this->utils.setFieldData(ice_saturation_, fm);
  this->utils.setFieldData(water_saturation_, fm);
  return;
}

// The thermal K calculation is based on a volume average mixture model.
template <typename EvalT, typename Traits>
void
ACEthermalConductivity<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      thermal_conductivity_(cell, qp) =
          pow(k_ice_, (ice_saturation_(cell, qp) * porosity_(cell, qp))) *
          pow(k_wat_, (water_saturation_(cell, qp) * porosity_(cell, qp))) *
          pow(k_sed_, (1.0 - porosity_(cell, qp)));
    }
  }

  return;
}

//
template <typename EvalT, typename Traits>
typename ACEthermalConductivity<EvalT, Traits>::ScalarT&
ACEthermalConductivity<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "ACE Ice Thermal Conductivity") { return k_ice_; }
  if (n == "ACE Water Thermal Conductivity") { return k_wat_; }
  if (n == "ACE Sediment Thermal Conductivity") { return k_sed_; }

  ALBANY_ASSERT(
      false, "Invalid request for value of ACE Component Thermal Conductivity");

  return k_wat_;  // does it matter what we return here?
}

}  // namespace LCM
