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
ACEtemperatureChange<EvalT, Traits>::ACEtemperatureChange(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : delta_temperature_(  // evaluated
          p.get<std::string>("ACE Temperature Change"),
          dl->qp_scalar),
      Temperature(  // dependent
          p.get<std::string>("Temperature Name"),
          dl->qp_scalar)
{
  Teuchos::ParameterList* temperatureChange_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read parameter values from input
  // min_water_saturation_ =
  //    temperatureChange_list->get<double>("Minimum Water Saturation");

  // Add temperature change as Sacado-ized parameters
  this->registerSacadoParameter("ACE Temperature Change", paramLib);

  // List evaluated fields
  this->addEvaluatedField(delta_temperature_);

  // List dependent fields
  this->addDependentField(Temperature);

  this->setName("ACE Temperature Change" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEtemperatureChange<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(delta_temperature_, fm);
  this->utils.setFieldData(Temperature, fm);
  return;
}

// This function updates the temperature change since the last time step.
template <typename EvalT, typename Traits>
void
ACEtemperatureChange<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      delta_temperature_(cell, qp) =
          Temperature(cell, qp) - temperature_old_(cell, qp);
      // Swap temperatures now
      temperature_old_(cell, qp) = Temperature(cell, qp);
      // set Boolean fields
      if (delta_temperature_(cell, qp) > 0.0) {
        temp_increasing_(cell, qp) = true;
        temp_decreasing_(cell, qp) = false;
      } else if (delta_temperature_(cell, qp) < 0.0) {
        temp_increasing_(cell, qp) = false;
        temp_decreasing_(cell, qp) = true;
      } else {
        temp_increasing_(cell, qp) = false;
        temp_decreasing_(cell, qp) = false;
      }
    }
  }

  return;
}

}  // namespace LCM
