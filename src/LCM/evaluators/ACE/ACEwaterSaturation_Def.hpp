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
ACEwaterSaturation<EvalT, Traits>::ACEwaterSaturation(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : water_saturation_(  // evaluated
          p.get<std::string>("ACE Water Saturation"),
          dl->qp_scalar),
      ice_saturation_(  // dependent
          p.get<std::string>("ACE Ice Saturation"),
          dl->qp_scalar)
{
  Teuchos::ParameterList* waterSaturation_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read minimum water saturation value
  min_water_saturation_ =
      waterSaturation_list->get<double>("Minimum Water Saturation");

  // Add water saturation as Sacado-ized parameters
  this->registerSacadoParameter("ACE Water Saturation", paramLib);

  // List evaluated fields
  this->addEvaluatedField(water_saturation_);

  // List dependent fields
  this->addDependentField(ice_saturation_);

  this->setName("ACE Water Saturation" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEwaterSaturation<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(water_saturation_, fm);
  return;
}

// This function updates the water saturation based on the
// ice saturation change.
template <typename EvalT, typename Traits>
void
ACEwaterSaturation<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      water_saturation_(cell, qp) = 1.0 - ice_saturation_(cell, qp);
      // check on realistic bounds
      water_saturation_(cell, qp) =
          std::max(min_water_saturation_, water_saturation_(cell, qp));
      water_saturation_(cell, qp) = std::min(1.0, water_saturation_(cell, qp));
    }
  }

  return;
}

//
template <typename EvalT, typename Traits>
typename ACEwaterSaturation<EvalT, Traits>::ScalarT&
ACEwaterSaturation<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Minimum Water Saturation") { return min_water_saturation_; }

  ALBANY_ASSERT(false, "Invalid request for value of Minimum Water Saturation");

  return min_water_saturation_;
}

}  // namespace LCM
