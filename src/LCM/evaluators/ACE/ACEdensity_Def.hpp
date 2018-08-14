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
ACEdensity<EvalT, Traits>::ACEdensity(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : density_(  // evaluated
          p.get<std::string>("ACE Density"),
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
  Teuchos::ParameterList* density_p_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read density values
  rho_ice_ = density_p_list->get<double>("Ice Value");
  rho_wat_ = density_p_list->get<double>("Water Value");
  rho_sed_ = density_p_list->get<double>("Sediment Value");

  // Add density as a Sacado-ized parameter
  this->registerSacadoParameter("ACE Density", paramLib);

  // List evaluated fields
  this->addEvaluatedField(density_);

  // List dependent fields
  this->addDependentField(porosity_);
  this->addDependentField(ice_saturation_);
  this->addDependentField(water_saturation_);

  this->setName("ACE Density" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEdensity<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(density_, fm);
  this->utils.setFieldData(porosity_, fm);
  this->utils.setFieldData(ice_saturation_, fm);
  this->utils.setFieldData(water_saturation_, fm);
  return;
}

// The density calculation is based on a volume average mixture model.
template <typename EvalT, typename Traits>
void
ACEdensity<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      density_(cell, qp) =
          porosity_(cell, qp) * (rho_ice_ * ice_saturation_(cell, qp) +
                                 rho_wat_ * water_saturation_(cell, qp)) +
          ((1.0 - porosity_(cell, qp)) * rho_sed_);
    }
  }

  return;
}

//
template <typename EvalT, typename Traits>
typename ACEdensity<EvalT, Traits>::ScalarT&
ACEdensity<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "ACE Ice Density") { return rho_ice_; }
  if (n == "ACE Water Density") { return rho_wat_; }
  if (n == "ACE Sediment Density") { return rho_sed_; }

  ALBANY_ASSERT(false, "Invalid request for value of ACE Component Density");

  return rho_wat_;  // does it matter what we return here?
}

}  // namespace LCM
