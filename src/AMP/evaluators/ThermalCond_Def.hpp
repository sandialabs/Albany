//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <fstream>
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace AMP {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermalCond<EvalT, Traits>::
ThermalCond(Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  coord_      (p.get<std::string>("Coordinate Name"),
               dl->qp_vector),
  k_          (p.get<std::string>("Thermal Conductivity Name"),
               dl->qp_scalar),
  psi_         (p.get<std::string>("Psi Name"),
               dl->qp_scalar)      

{

  this->addDependentField(coord_);
  this->addDependentField(psi_);
  this->addEvaluatedField(k_);
 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::Device::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Powder Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidThermalCondParameters();

//  cond_list->validateParameters(*reflist, 0,
//      Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  std::string type = cond_list->get("Thermal Conductivity Type", "Constant");

  powder_value_ = cond_list->get("Value", 1.0);

  // get substrate
  cond_list =
    p.get<Teuchos::ParameterList*>("Solid Parameter List");

  cond_list->validateParameters(*reflist, 0,
      Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  type = cond_list->get("Thermal Conductivity Type", "Constant");

  solid_value_ = cond_list->get("Value", 1.0);
  

  this->setName("ThermalCond"+PHX::typeAsString<EvalT>());

}


//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalCond<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coord_,fm);
  this->utils.setFieldData(psi_,fm);
  this->utils.setFieldData(k_,fm);
}

//**********************************************************************

template<typename EvalT, typename Traits>
void ThermalCond<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

    // thermal conductivity
    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    {
        for (std::size_t qp = 0; qp < num_qps_; ++qp)
        {
            k_(cell, qp) = (1.0 - psi_(cell, qp)) * powder_value_
                    + psi_(cell, qp) * solid_value_;
        }
    }
}

//**********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
ThermalCond<EvalT, Traits>::
getValidThermalCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Thermal Conductivity Params"));;

  valid_pl->set<std::string>("Thermal Conductivity Type", "Constant",
      "Constant thermal conductivity across the element block");
  valid_pl->set<double>("Value", 1.0, "Constant thermal conductivity value");

  return valid_pl;
}

//**********************************************************************
}
