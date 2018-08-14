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
ACEthermalInertia<EvalT, Traits>::ACEthermalInertia(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thermal_inertia_(  // evaluated
          p.get<std::string>("ACE Thermal Inertia"),
          dl->qp_scalar),
      density_(  // dependent
          p.get<std::string>("ACE Density"),
          dl->qp_scalar),
      heat_capacity_(  // dependent
          p.get<std::string>("ACE Heat Capacity"),
          dl->qp_scalar),
      dfdT_(  // dependent
          p.get<std::string>("ACE Freezing Curve Slope"),
          dl->qp_scalar)
{
  Teuchos::ParameterList* thermal_inertia_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read thermal inertia values
  latent_heat_ =
      thermal_inertia_list->get<double>("ACE Latent Heat of Phase Change");
  rho_ice_ = thermal_inertia_list->get<double>("ACE Ice Density");

  // Add thermal inertia as a Sacado-ized parameter
  this->registerSacadoParameter("ACE Thermal Inertia", paramLib);

  // List evaluated fields
  this->addEvaluatedField(thermal_inertia_);

  // List dependent fields
  this->addDependentField(density_);
  this->addDependentField(heat_capacity_);
  this->addDependentField(dfdT_);

  this->setName("ACE Thermal Inertia" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEthermalInertia<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(thermal_inertia_, fm);
  this->utils.setFieldData(density_, fm);
  this->utils.setFieldData(heat_capacity_, fm);
  this->utils.setFieldData(dfdT_, fm);
  return;
}

// The thermal inertia calculation uses mixture model values for density
// and heat capacity.
// During phase change only, dfdT_ is non zero.
template <typename EvalT, typename Traits>
void
ACEthermalInertia<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      thermal_inertia_(cell, qp) =
          (density_(cell, qp) * heat_capacity_(cell, qp)) -
          (rho_ice_ * latent_heat_ * dfdT_(cell, qp));
    }
  }

  return;
}

//
template <typename EvalT, typename Traits>
typename ACEthermalInertia<EvalT, Traits>::ScalarT&
ACEthermalInertia<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "ACE Latent Heat of Phase Change") { return latent_heat_; }
  if (n == "ACE Ice Density") { return rho_ice_; }

  ALBANY_ASSERT(
      false, "Invalid request for value of ACE Component Thermal Inertia");

  return latent_heat_;  // does it matter what we return here?
}

}  // namespace LCM
