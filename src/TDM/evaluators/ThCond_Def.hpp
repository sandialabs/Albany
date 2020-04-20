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
  ThCond<EvalT, Traits>::
  ThCond(Teuchos::ParameterList& p,
	 const Teuchos::RCP<Albany::Layouts>& dl) :
    coord_      (p.get<std::string>("Coordinate Name"), dl->qp_vector),
    T_          (p.get<std::string>("Temperature Name"), dl->qp_scalar),
    k_          (p.get<std::string>("Thermal Conductivity Name"), dl->qp_scalar),
    phi1_       (p.get<std::string>("Phi1 Name"), dl->qp_scalar),
    phi2_       (p.get<std::string>("Phi2 Name"), dl->qp_scalar),    
    psi1_	(p.get<std::string>("Psi1 Name"), dl->qp_scalar),
    psi2_       (p.get<std::string>("Psi2 Name"), dl->qp_scalar),
    depth_      (p.get<std::string>("Depth Name"), dl->qp_scalar)

  {

    this->addDependentField(coord_);
    this->addDependentField(T_);
    this->addDependentField(phi1_);
    this->addDependentField(phi2_);
    this->addDependentField(psi1_);	
    this->addDependentField(psi2_);
    this->addDependentField(depth_);

    this->addEvaluatedField(k_);
 
    Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
    std::vector<PHX::Device::size_type> dims;
    scalar_dl->dimensions(dims);
    workset_size_ = dims[0];
    num_qps_      = dims[1];

/*
    Teuchos::RCP<const Teuchos::ParameterList> reflist =
       this->getValidThCondParameters();

    cond_list->validateParameters(*reflist, 0,
       Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);
*/	   
	//Parameters for the functional form of the premelted thermal conductivity
    Teuchos::ParameterList* cond_list_premelted =
    p.get<Teuchos::ParameterList*>("Parameter List Pre-melted");
    aPre = cond_list_premelted->get("a", 1.0);
    bPre = cond_list_premelted->get("b", 1.0);
    cPre = cond_list_premelted->get("c", 1.0);
    dPre = cond_list_premelted->get("d", 1.0);
    ePre = cond_list_premelted->get("e", 1.0);

  //Parameters for the functional form of the liquid thermal conductivity
    Teuchos::ParameterList* cond_list_liquid =
    p.get<Teuchos::ParameterList*>("Parameter List Liquid");
    aL = cond_list_liquid->get("a", 1.0);
    bL = cond_list_liquid->get("b", 1.0);
    cL = cond_list_liquid->get("c", 1.0);
    dL = cond_list_liquid->get("d", 1.0);
    eL = cond_list_liquid->get("e", 1.0);

  //Parameters for the functional form of the postmelted thermal conductivity
    Teuchos::ParameterList* cond_list_postmelted =
    p.get<Teuchos::ParameterList*>("Parameter List Post-melted");
    aPo = cond_list_postmelted->get("a", 1.0);
    bPo = cond_list_postmelted->get("b", 1.0);
    cPo = cond_list_postmelted->get("c", 1.0);
    dPo = cond_list_postmelted->get("d", 1.0);
    ePo = cond_list_postmelted->get("e", 1.0);

  //Parameters for the functional form of the vapor thermal conductivity
    Teuchos::ParameterList* cond_list_vapor =
    p.get<Teuchos::ParameterList*>("Parameter List Vapor");
    aV = cond_list_vapor->get("a", 1.0);
    bV = cond_list_vapor->get("b", 1.0);
    cV = cond_list_vapor->get("c", 1.0);
    dV = cond_list_vapor->get("d", 1.0);
    eV = cond_list_vapor->get("e", 1.0);

  //Parameters for the initial porosity
    Teuchos::ParameterList* porosity_list =
    p.get<Teuchos::ParameterList*>("InitialPorosity Parameter List");

    initial_porosity_ = porosity_list->get("Value", 1.0);

    /*
    Teuchos::ParameterList* input_list =
    		p.get<Teuchos::ParameterList*>("Input List");  
			
	std::string sim_type = input_list->get<std::string>("Simulation Type");

      */
		/*
	if (sim_type == "SLM Additive"){
		Teuchos::ParameterList* cond_list = p.get<Teuchos::ParameterList*>("Powder Parameter List");
		//Assume constant thermal conductivity in the powder
		Kp_ = cond_list->get("a", 1.0);
	}	
	else{
		Kp_ = 0;
	}
	*/
    this->setName("ThCond"+PHX::print<EvalT>());
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void ThCond<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(coord_,fm);
    this->utils.setFieldData(T_,fm);
    this->utils.setFieldData(phi1_,fm);
    this->utils.setFieldData(phi2_,fm);
    this->utils.setFieldData(psi1_,fm);	
    this->utils.setFieldData(psi2_,fm);
    this->utils.setFieldData(depth_,fm);

    this->utils.setFieldData(k_,fm);
  }

  //**********************************************************************

  template<typename EvalT, typename Traits>
  void ThCond<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {

    // thermal conductivity
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_qps_; ++qp){
        if(depth_(cell,qp)<0){  // if element already vaporized, then set conductivity to 0
	  k_(cell,qp) = 0;
	}
	else{
	  //calculating pre-melted k value
	  Kp_ = (aPre + bPre*T_(cell, qp) + cPre*T_(cell, qp)*T_(cell, qp) + dPre/T_(cell, qp) + ePre/(T_(cell, qp)*T_(cell, qp)));  
	  //calculating liquid k value
	  Kl_ = (aL + bL*T_(cell, qp) + cL*T_(cell, qp)*T_(cell, qp) + dL/T_(cell, qp) + eL/(T_(cell, qp)*T_(cell, qp)));  
	  //calculating post-melted k value
	  Kd_ = (aPo + bPo*T_(cell, qp) + cPo*T_(cell, qp)*T_(cell, qp) + dPo/T_(cell, qp) + ePo/(T_(cell, qp)*T_(cell, qp)));  
	  //calculating vapor k value
	  Kv_ = (aV + bV*T_(cell, qp) + cV*T_(cell, qp)*T_(cell, qp) + dV/T_(cell, qp) + eV/(T_(cell, qp)*T_(cell, qp)));  
    //calculating powder k value
    Kpowder_ = 0.05 * Kd_;
	  //calculating solid k value
    if (initial_porosity_== 0.0 ){
      Ks_ = Kd_;
    }
    else {
      Ks_ = (1 - psi1_(cell,qp))*Kpowder_ + psi1_(cell,qp)*Kd_;
    }

	  // calculating the final k value
	  if(psi2_(cell,qp)<1){	// if element never fully vaporized, use phi2 to reflect instant temperature
	     k_(cell, qp) = (Ks_*(1.0 - phi1_(cell, qp)) + Kl_*phi1_(cell, qp))*(1.0 - phi2_(cell, qp)) + Kv_*phi2_(cell, qp);
	  }
	  else{	// if element fully vaporized, use psi2
	     k_(cell, qp) = (Ks_*(1.0 - phi1_(cell, qp)) + Kl_*phi1_(cell, qp))*(1.0 - psi2_(cell, qp)) + Kv_*psi2_(cell, qp);
	  }
	}
      }
    }
    //std::cout << "thCond has been finished\n" ; 
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  Teuchos::RCP<const Teuchos::ParameterList>
  ThCond<EvalT, Traits>::
  getValidThCondParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Thermal Conductivity Params"));;

    valid_pl->set<double>("a", 1.0);
    valid_pl->set<double>("b", 1.0);
    valid_pl->set<double>("c", 1.0);
    valid_pl->set<double>("d", 1.0);
    valid_pl->set<double>("e", 1.0);
	
    valid_pl->set<double>("Thermal Conductivity Liquid", 1.0);
    valid_pl->set<double>("Thermal Conductivity Vapor", 1.0);
    valid_pl->set<double>("Thermal Conductivity Powder", 1.0);
    return valid_pl;
  }

  //**********************************************************************
}
