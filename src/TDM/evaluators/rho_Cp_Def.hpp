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
  rho_Cp<EvalT, Traits>::
  rho_Cp(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
    coord_      (p.get<std::string>("Coordinate Name"),dl->qp_vector),
    psi1_           (p.get<std::string>("Psi1 Name"), dl->qp_scalar),
    //porosity_   (p.get<std::string>("Porosity Name"),dl->qp_scalar),
    rho_cp_     (p.get<std::string>("rho_Cp Name"),dl->qp_scalar)
  {

    this->addDependentField(coord_);
    this->addDependentField(psi1_);
    this->addEvaluatedField(rho_cp_);
 
    Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
    std::vector<PHX::Device::size_type> dims;
    scalar_dl->dimensions(dims);
    workset_size_ = dims[0];
    num_qps_      = dims[1];

    Teuchos::ParameterList* cond_list =
      p.get<Teuchos::ParameterList*>("Parameter List");
    Teuchos::ParameterList* porosity_list =
      p.get<Teuchos::ParameterList*>("InitialPorosity Parameter List");      
    /*
    Teuchos::RCP<const Teuchos::ParameterList> reflist =
      this->getValidrho_CpParameters();

    cond_list->validateParameters(*reflist, 0,
				  Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);
    */
    constant_value_ = cond_list->get("Value", 1.0);
    initial_porosity_ = porosity_list->get("Value",0);
    this->setName("rho_Cp"+PHX::print<EvalT>());
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void rho_Cp<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(coord_,fm);
    this->utils.setFieldData(rho_cp_,fm); 
    this->utils.setFieldData(psi1_,fm); 

    //this->utils.setFieldData(porosity_,fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void rho_Cp<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {


    // current time
    const RealType time = workset.current_time;

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_qps_; ++qp) {
        rho_cp_(cell,qp) = constant_value_ * ( 1 - initial_porosity_ * ( 1 - psi1_(cell,qp) ) ) ;
      }
    } 
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  Teuchos::RCP<const Teuchos::ParameterList>
  rho_Cp<EvalT, Traits>::
  getValidrho_CpParameters() const
  {
 
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid rho_Cp Params"));

    valid_pl->set<double>("Value", 1.0);

    return valid_pl;

  }
  //**********************************************************************
}
