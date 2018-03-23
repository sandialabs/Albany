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
ACEiceSaturation<EvalT, Traits>::ACEiceSaturation(Teuchos::ParameterList& p)
    : ice_saturation_(
          p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      water_saturation_(
          p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* iceSaturation_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read initial saturation values
  ice_saturation_init_ = 
      iceSaturation_list->get<double>("Initial Ice Saturation");

  // Add iceSaturation as Sacado-ized parameters
  this->registerSacadoParameter("ACE Ice Saturation", paramLib);
  this->registerSacadoParameter("ACE Water Saturation", paramLib);

  // List evaluated fields
  this->addEvaluatedField(ice_saturation_);
  this->addEvaluatedField(water_saturation_);
  
  // List dependent fields
  
  this->setName("ACE Saturations" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEiceSaturation<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // List all fields
  this->utils.setFieldData(ice_saturation_, fm);
  this->utils.setFieldData(water_saturation_, fm);
  return;
}


// This function updates the ice and water saturations.
template <typename EvalT, typename Traits>
void
ACEiceSaturation<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int num_cells = workset.numCells;

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      ice_saturation_(cell, qp) = 1.0;
      water_saturation_(cell, qp) = 1.0 - ice_saturation_(cell, qp);      
      // check on realistic bounds
      ice_saturation_(cell, qp) = std::max(0.0,ice_saturation_(cell, qp));
      ice_saturation_(cell, qp) = std::min(1.0,ice_saturation_(cell, qp));
      water_saturation_(cell, qp) = std::max(0.0,water_saturation_(cell, qp));
      water_saturation_(cell, qp) = std::min(1.0,water_saturation_(cell, qp));
    }
  }

  return;
}

//
// template <typename EvalT, typename Traits>
// typename ACEiceSaturation<EvalT, Traits>::ScalarT&
// ACEiceSaturation<EvalT, Traits>::getValue(const std::string& n)
// {
//   if (n == "ACE Ice Saturation") {
//     return ice_saturation_;
//   }
//   if (n == "ACE Water Saturation") {
//     return water_saturation_;
//   }
// 
//   ALBANY_ASSERT(
//       false, 
//       "Invalid request for value of ACE Ice or Water Saturation");
// 
//   return ice_saturation_; // does it matter what we return here?
// }

}  // namespace LCM
