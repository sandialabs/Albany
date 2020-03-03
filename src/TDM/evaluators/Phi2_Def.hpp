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
  Phi2<EvalT, Traits>::
  Phi2(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
    T_          (p.get<std::string>("Temperature Name"),dl->qp_scalar),
    phi2_        (p.get<std::string>("Phi2 Name"),dl->qp_scalar)
  {

    this->addDependentField(T_);
    this->addEvaluatedField(phi2_);

 
    Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
    std::vector<PHX::Device::size_type> dims;
    scalar_dl->dimensions(dims);
    workset_size_ = dims[0];
    num_qps_      = dims[1];

    Teuchos::ParameterList* cond_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

    Teuchos::RCP<const Teuchos::ParameterList> reflist =
      this->getValidPhi2Parameters();

    cond_list->validateParameters(*reflist, 0,
				  Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

    std::string type; 

    VaporizationTemperature_ = cond_list->get("Vaporization Temperature", 3300.0);
    deltaTemperature_ = cond_list->get("delta Temperature", 50.0); 

    this->setName("Phi2"+PHX::print<EvalT>());
  }

  //**********************************************************************



  //**********************************************************************
  template<typename EvalT, typename Traits>
  void Phi2<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(T_,fm);
    this->utils.setFieldData(phi2_,fm);
  }

  //**********************************************************************

  template<typename EvalT, typename Traits>
  void Phi2<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {

    // current time
    const RealType t = workset.current_time;

    if (t > 0.0)  {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell){
        for (std::size_t qp = 0; qp < num_qps_; ++qp){
          phi2_(cell, qp) = 0.5 * (std::tanh((T_(cell, qp) - VaporizationTemperature_) / deltaTemperature_) + 1.0);
	}
      }
    }
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  Teuchos::RCP<const Teuchos::ParameterList>
  Phi2<EvalT, Traits>::
  getValidPhi2Parameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid Phi2 Params"));
  
    valid_pl->set<double>("Vaporization Temperature", 1.0);
    valid_pl->set<double>("delta Temperature", 1.0);
  
    return valid_pl;
  }

  //**********************************************************************

}
