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
ACEthermalConductivity<EvalT, Traits>::ACEthermalConductivity(Teuchos::ParameterList& p)
    : thermal_conductivity_(
          p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* thermal_conductivity_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = thermal_conductivity_list->get("ACE Thermal Conductivity Type", "Constant");
  if (type == "Constant") {
    is_constant_ = true;
    constant_value_ = thermal_conductivity_list->get<double>("Value");

    // Add thermal conductivity as a Sacado-ized parameter
    this->registerSacadoParameter("ACE Thermal Conductivity", paramLib);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
             "Invalid ACE thermal conductivity type " << type);
  }

  this->addEvaluatedField(thermal_conductivity_);
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
  return;
}

//
template <typename EvalT, typename Traits>
void
ACEthermalConductivity<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  if (is_constant_ == true) {
    for (int cell = 0; cell < num_cells; ++cell) {
      for (int qp = 0; qp < num_qps_; ++qp) {
        thermal_conductivity_(cell, qp) = constant_value_;
      }
    }
  }

  return;
}

//
template <typename EvalT, typename Traits>
typename ACEthermalConductivity<EvalT, Traits>::ScalarT&
ACEthermalConductivity<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "ACE Thermal Conductivity") {
    return constant_value_;
  }

  ALBANY_ASSERT(false, "Invalid request for value of ACE Thermal Conductivity");

  return constant_value_;
}

}  // namespace LCM
