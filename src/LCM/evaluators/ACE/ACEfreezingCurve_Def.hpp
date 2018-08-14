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
ACEfreezingCurve<EvalT, Traits>::ACEfreezingCurve(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : dfdT_(  // evaluated
          p.get<std::string>("ACE Freezing Curve Slope"),
          dl->qp_scalar),
      ice_saturation_evaluated_(  // evaluated
          p.get<std::string>("ACE Evaluated Ice Saturation"),
          dl->qp_scalar),
      Temperature(  // dependent
          p.get<std::string>("Temperature Name"),
          dl->qp_scalar),
      melting_temperature_(  // dependent
          p.get<std::string>("ACE Melting Temperature"),
          dl->qp_scalar),
      delta_temperature_(  // dependent
          p.get<std::string>("ACE Temperature Change"),
          dl->qp_scalar)
{
  Teuchos::ParameterList* freezingCurve_list =
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
  temperature_range_ =
      freezingCurve_list->get<double>("Phase Change Temperature Range");

  // Add evaluated parameters as Sacado-ized parameters
  this->registerSacadoParameter("ACE Evaluated Ice Saturation", paramLib);
  this->registerSacadoParameter("ACE Freezing Curve Slope", paramLib);

  // List evaluated fields
  this->addEvaluatedField(ice_saturation_evaluated_);
  this->addEvaluatedField(dfdT_);

  // List dependent fields
  this->addDependentField(Temperature);
  this->addDependentField(melting_temperature_);
  this->addDependentField(delta_temperature_);

  this->setName("ACE Freezing Curve" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEfreezingCurve<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(ice_saturation_evaluated_, fm);
  this->utils.setFieldData(dfdT_, fm);
  this->utils.setFieldData(Temperature, fm);
  this->utils.setFieldData(melting_temperature_, fm);
  this->utils.setFieldData(delta_temperature_, fm);
  return;
}

// This function evaluates the ice saturation based on the freezing curve
template <typename EvalT, typename Traits>
void
ACEfreezingCurve<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  ScalarT T_low;
  ScalarT T_high;

  ScalarT f_old = 0.0;  // temporary here

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      T_low = melting_temperature_(cell, qp) - (temperature_range_ / 2.0);

      T_high = melting_temperature_(cell, qp) + (temperature_range_ / 2.0);

      // completely frozen
      if (Temperature(cell, qp) <= T_low) {
        ice_saturation_evaluated_(cell, qp) = 1.0;
      }
      // completely melted
      if (Temperature(cell, qp) >= T_high) {
        ice_saturation_evaluated_(cell, qp) = 0.0;
      }
      // in phase change
      if ((Temperature(cell, qp) > T_low) && (Temperature(cell, qp) < T_high)) {
        ice_saturation_evaluated_(cell, qp) =
            -1.0 * (Temperature(cell, qp) / temperature_range_) + T_high;
      }

      // Note: The freezing curve is a simple linear relationship that is sharp
      // at the T_low and T_high points. I don't know if this will actually
      // cause problems or not. If it does, we can try a curved relationship.
      dfdT_(cell, qp) = (ice_saturation_evaluated_(cell, qp) - f_old) /
                        delta_temperature_(cell, qp);
    }
  }

  return;
}

//
template <typename EvalT, typename Traits>
typename ACEfreezingCurve<EvalT, Traits>::ScalarT&
ACEfreezingCurve<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Phase Change Temperature Range") { return temperature_range_; }

  ALBANY_ASSERT(
      false, "Invalid request for value of Phase Change Temperature Range");

  return temperature_range_;  // does it matter what we return here?
}

}  // namespace LCM
