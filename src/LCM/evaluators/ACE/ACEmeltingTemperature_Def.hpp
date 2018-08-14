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
ACEmeltingTemperature<EvalT, Traits>::ACEmeltingTemperature(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : melting_temperature_(  // evaluated
          p.get<std::string>("ACE Melting Temperature"),
          dl->qp_scalar),
      pressure_(  // dependent
          p.get<std::string>("ACE Pressure"),
          dl->qp_scalar),
      salinity_(  // dependent
          p.get<std::string>("ACE Salinity"),
          dl->qp_scalar)
{
  Teuchos::ParameterList* melting_temp_p_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read melting temperature values
  // rho_ice_ = melting_temp_p_list->get<double>("Ice Value");

  // Add melting temperature as a Sacado-ized parameter
  this->registerSacadoParameter("ACE Melting Temperature", paramLib);

  // List evaluated fields
  this->addEvaluatedField(melting_temperature_);

  // List dependent fields
  this->addDependentField(pressure_);
  this->addDependentField(salinity_);

  this->setName("ACE Melting Temperature" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEmeltingTemperature<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(melting_temperature_, fm);
  this->utils.setFieldData(pressure_, fm);
  this->utils.setFieldData(salinity_, fm);
  return;
}

// The melting temperature calculation is based on . . . need citation.
// It assumes the salinity is in [ppt] units.
// It assumes the pressure is in [Pa] units.
template <typename EvalT, typename Traits>
void
ACEmeltingTemperature<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      ScalarT const
          // sal = salinity_(cell, qp); salinity residual not written yet
          sal = 0.10;

      ScalarT const sal15 = std::sqrt(sal * sal * sal);

      melting_temperature_(cell, qp) =
          (-0.057 * sal) + (0.00170523 * sal15) - (0.0002154996 * sal * sal) -
          ((0.000753 / 10000.0) * pressure_(cell, qp));
    }
  }

  return;
}

//
// template <typename EvalT, typename Traits>
// typename ACEmeltingTemperature<EvalT, Traits>::ScalarT&
// ACEmeltingTemperature<EvalT, Traits>::getValue(const std::string& n)
//{
//  if (n == "ACE Ice Density") {
//    return rho_ice_;
//  }
//  if (n == "ACE Water Density") {
//    return rho_wat_;
//  }
//  if (n == "ACE Sediment Density") {
//    return rho_sed_;
//  }
//
//  ALBANY_ASSERT(false, "Invalid request for value of ACE Component Density");
//
//  return rho_wat_; // does it matter what we return here?
//}

}  // namespace LCM
