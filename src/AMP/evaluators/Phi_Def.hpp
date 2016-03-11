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
Phi<EvalT, Traits>::
Phi(Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  T_          (p.get<std::string>("Temperature Name"),
               dl->qp_scalar),
  phi_        (p.get<std::string>("Phi Name"),
               dl->qp_scalar)
{

  this->addDependentField(T_);
  this->addEvaluatedField(phi_);

 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::Device::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidPhiParameters();

  cond_list->validateParameters(*reflist, 0,
      Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  // dummy variable used multiple times below
  std::string type; 

  type = cond_list->get("Melting Temperature Type", "Constant");
  MeltingTemperature_ = cond_list->get("Melting Temperature Value", 1700.0);

  type = cond_list->get("delta Temperature Type", "Constant");
  deltaTemperature_ = cond_list->get("delta Temperature Value", 50.0); 

  this->setName("Phi"+PHX::typeAsString<EvalT>());
}

//**********************************************************************



//**********************************************************************
template<typename EvalT, typename Traits>
void Phi<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(T_,fm);
  this->utils.setFieldData(phi_,fm);
}

//**********************************************************************

template<typename EvalT, typename Traits>
void Phi<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
    // current time
    const RealType t = workset.current_time;

    if (t > 0.0)
    {
        // defining phi. Note that phi = 0 if T < Tm and phi = 1 if T > Tm
        for (std::size_t cell = 0; cell < workset.numCells; ++cell)
        {
            for (std::size_t qp = 0; qp < num_qps_; ++qp)
            {
                // Compute phi
                phi_(cell, qp) = 0.5 * (std::tanh((T_(cell, qp) - MeltingTemperature_) / deltaTemperature_) + 1.0);
            }
        }
    }
}

//**********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Phi<EvalT, Traits>::
getValidPhiParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Phi Params"));
  
  valid_pl->set<std::string>("Melting Temperature Type", "Constant");
  valid_pl->set<double>("Melting Temperature Value", 1.0);
  
  valid_pl->set<std::string>("delta Temperature Type", "Constant");
  valid_pl->set<double>("delta Temperature Value", 1.0);
  
  return valid_pl;
}

//**********************************************************************

}
