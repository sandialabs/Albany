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


namespace TDM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  Psi2<EvalT, Traits>::
  Psi2(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl) :
    phi2_        (p.get<std::string>("Phi2 Name"),
		  dl->qp_scalar),
    psi2_        (p.get<std::string>("Psi2 Name"),
		  dl->qp_scalar)
  {

    this->addDependentField(phi2_);
    this->addEvaluatedField(psi2_);
 
    Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
    std::vector<PHX::Device::size_type> dims;
    scalar_dl->dimensions(dims);
    workset_size_ = dims[0];
    num_qps_      = dims[1];

    Teuchos::ParameterList* cond_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

    Teuchos::RCP<const Teuchos::ParameterList> reflist = 
      this->getValidPsi2Parameters(); 

    cond_list->validateParameters(*reflist, 0, 
				  Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED); 

    constant_value_ = cond_list->get("Psi2", 0.0); 
 
    psi2_Name_ = p.get<std::string>("Psi2 Name")+"_old";
    phi2_Name_ = p.get<std::string>("Phi2 Name")+"_old";
    this->setName("Psi2"+PHX::print<EvalT>());
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void Psi2<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(phi2_,fm);
    this->utils.setFieldData(psi2_,fm);
  }

  //**********************************************************************

  template<typename EvalT, typename Traits>
  void Psi2<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {

    //grab old phi value
    Albany::MDArray phi2_old = (*workset.stateArrayPtr)[phi2_Name_];

    //grab old psi value
    Albany::MDArray psi2_old = (*workset.stateArrayPtr)[psi2_Name_];

    // current time
    const RealType t = workset.current_time;

    // do this only at the first eqiulibration step
    if (t == 0.0)
      {
        // initializing psi_ values:
        for (std::size_t cell = 0; cell < workset.numCells; ++cell){
            for (std::size_t qp = 0; qp < num_qps_; ++qp){
                psi2_(cell, qp) = constant_value_;
	        }
	    }
    }
    else{
        // defining psi_
        for (std::size_t cell = 0; cell < workset.numCells; ++cell){
            for (std::size_t qp = 0; qp < num_qps_; ++qp){
                ScalarT phi2_bar = std::max(phi2_old(cell, qp), phi2_(cell, qp));
                psi2_(cell, qp) = std::max(psi2_old(cell, qp), phi2_bar);
	        }
	    }
    }
  }
  //**********************************************************************
  template<typename EvalT, typename Traits>
  Teuchos::RCP<const Teuchos::ParameterList>
  Psi2<EvalT, Traits>::
  getValidPsi2Parameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid Psi2 Params"));;

    valid_pl->set<double>("Psi2", 1.0);
    return valid_pl;
  }
  //**********************************************************************

}
