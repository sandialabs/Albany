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
Local_Porosity<EvalT, Traits>::
Local_Porosity(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  coord_      (p.get<std::string>("Coordinate Name"),dl->qp_vector),
  porosity_   (p.get<std::string>("Porosity Name"),dl->qp_scalar),
  psi_        (p.get<std::string>("Psi Name"),dl->qp_scalar)

{
  this->addDependentField(coord_);
  this->addDependentField(psi_);
  this->addEvaluatedField(porosity_);
 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::Device::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidLocal_PorosityParameters();

  cond_list->validateParameters(*reflist, 0,
      Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  // dummy variable used multiple times below
  std::string type; 

  type = cond_list->get("Porosity Type", "Constant");
  Initial_porosity = cond_list->get("Value", 0.0);

  this->setName("Porosity"+PHX::typeAsString<EvalT>());

}


//**********************************************************************
template<typename EvalT, typename Traits>
void Local_Porosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coord_,fm);
  this->utils.setFieldData(psi_,fm);
  this->utils.setFieldData(porosity_,fm);
}

//**********************************************************************

template<typename EvalT, typename Traits>
void Local_Porosity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
    // porosity
    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    {
        for (std::size_t qp = 0; qp < num_qps_; ++qp)
        {
            porosity_(cell, qp) = Initial_porosity*(1.0 - psi_(cell, qp));
        }
    }
}

//**********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Local_Porosity<EvalT, Traits>::
getValidLocal_PorosityParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Porosity Params"));;

  valid_pl->set<std::string>("Porosity Type", "Constant",
      "Constant porosity across the element block");
  valid_pl->set<double>("Value", 1.0, "Constant Porosity value");

  return valid_pl;
}

//**********************************************************************
}