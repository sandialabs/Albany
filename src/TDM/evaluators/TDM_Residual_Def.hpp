//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

#include <PCU.h>
#include <pumi.h>
#include <apf.h>
#include <pcu_util.h>

#include "Albany_Application.hpp"
#include "Albany_APFMeshStruct.hpp"
#include "Albany_APFDiscretization.hpp"

namespace TDM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  Phase_Residual<EvalT, Traits>::
  Phase_Residual(const Teuchos::ParameterList& p,
		 const Teuchos::RCP<Albany::Layouts>& dl) :
    w_bf_             (p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
    w_grad_bf_        (p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_vector),
    T_                (p.get<std::string>("Temperature Name"), dl->qp_scalar),
    T_grad_           (p.get<std::string>("Temperature Gradient Name"), dl->qp_vector),
    k_                (p.get<std::string>("Thermal Conductivity Name"), dl->qp_scalar),
    rho_cp_           (p.get<std::string>("rho_Cp Name"), dl->qp_scalar),
    laser_source_     (p.get<std::string>("Laser Source Name"), dl->qp_scalar),
    time              (p.get<std::string>("Time Name"), dl->workset_scalar),
    psi1_             (p.get<std::string>("Psi1 Name"), dl->qp_scalar),
    psi2_             (p.get<std::string>("Psi2 Name"), dl->qp_scalar),
    phi1_             (p.get<std::string>("Phi1 Name"), dl->qp_scalar),
    phi2_             (p.get<std::string>("Phi2 Name"), dl->qp_scalar),
    energyDot_        (p.get<std::string>("Energy Rate Name"), dl->qp_scalar),
    deltaTime         (p.get<std::string>("Delta Time Name"), dl->workset_scalar),
    residual_         (p.get<std::string>("Residual Name"), dl->node_scalar)

  {
    this->addDependentField(w_bf_);
    this->addDependentField(w_grad_bf_);
    this->addDependentField(T_);
    this->addDependentField(T_grad_);
    this->addDependentField(k_);
    this->addDependentField(rho_cp_);
    this->addDependentField(laser_source_);
    this->addDependentField(phi1_);
    this->addDependentField(phi2_);
    this->addDependentField(psi1_);
    this->addDependentField(psi2_);
    this->addDependentField(energyDot_);
    this->addDependentField(time);
    this->addDependentField(deltaTime);
    this->addEvaluatedField(residual_);
   
    std::vector<PHX::Device::size_type> dims;
    w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
    workset_size_ = dims[0];
    num_nodes_    = dims[1];
    num_qps_      = dims[2];
    num_dims_     = dims[3];
	
  /*
	Teuchos::ParameterList* input_list =
		p.get<Teuchos::ParameterList*>("Input List");
		
	//From main 3DM input file
	sim_type = input_list->get<std::string>("Simulation Type");
	if (sim_type == "SLM Additive"){
		initial_porosity = input_list->get("Powder Layer Initial Porosity", 1.0);
	}
  */

    Teuchos::ParameterList* cond_list = 
    p.get<Teuchos::ParameterList*>("InitialPorosity Parameter List");
    initial_porosity= cond_list->get("Value",0.0);

    Temperature_Name_ = p.get<std::string>("Temperature Name")+"_old";

    this->setName("Phase_Residual"+PHX::print<EvalT>());
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void Phase_Residual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(w_bf_,fm);
    this->utils.setFieldData(w_grad_bf_,fm);
    this->utils.setFieldData(T_,fm);
    this->utils.setFieldData(T_grad_,fm);
    this->utils.setFieldData(k_,fm);
    this->utils.setFieldData(rho_cp_,fm);
    this->utils.setFieldData(laser_source_,fm);
    this->utils.setFieldData(time,fm);
    this->utils.setFieldData(deltaTime,fm);
    this->utils.setFieldData(phi1_,fm);
    this->utils.setFieldData(phi2_,fm);
    this->utils.setFieldData(psi1_,fm);
    this->utils.setFieldData(psi2_,fm);
    this->utils.setFieldData(energyDot_,fm);
    this->utils.setFieldData(residual_,fm);


    term1_ = Kokkos::createDynRankView(k_.get_view(), "term1_", workset_size_,num_qps_,num_dims_);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void Phase_Residual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    // time step
    ScalarT dt = deltaTime(0);
    typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

    //  if (dt == 0.0) dt = 1.0e-15;
  
    //grab old temperature
    Albany::MDArray T_old = (*workset.stateArrayPtr)[Temperature_Name_];
    
    // diffusive term multiplication
    FST::scalarMultiplyDataData<ScalarT> (term1_, k_.get_view(), T_grad_.get_view());
    
    // zero out residual
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < num_nodes_; ++node) {
        residual_(cell,node) = 0.0;
      }
    }
	
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < num_qps_; ++qp) {
	for (int node = 0; node < num_nodes_; ++node) {
	  //if (sim_type == "SLM Additive"){
	  det_F = (1 - initial_porosity)/(1 - initial_porosity*(1-psi1_(cell,qp)));
	  F_inv = 1.0/det_F;
	  //}

	  //diffusive term
	  residual_(cell, node) += (w_grad_bf_(cell, node, qp, 0) * term1_(cell, qp, 0)
				    +  w_grad_bf_(cell, node, qp, 1) * term1_(cell, qp, 1)
				    + w_grad_bf_(cell, node, qp, 2) * term1_(cell, qp, 2) * F_inv*F_inv);
	  // laser heat source term
	  residual_(cell, node) -= (w_bf_(cell, node, qp) * laser_source_(cell, qp));
			   
	  // transient term		
	  residual_(cell, node) += (w_bf_(cell, node, qp) * energyDot_(cell, qp));
				
	  //multiply by detF
	  residual_(cell, node) = residual_(cell, node)*det_F;
			
	}
      }
    } 

  }
  //*********************************************************************
}
