//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace AMP {

//**********************************************************************
template<typename EvalT, typename Traits>
PhaseResidual<EvalT, Traits>::
PhaseResidual(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  w_bf_       (p.get<std::string>("Weighted BF Name"),
               dl->node_qp_scalar),
  w_grad_bf_  (p.get<std::string>("Weighted Gradient BF Name"),
               dl->node_qp_vector),
  T_          (p.get<std::string>("Temperature Name"),
               dl->qp_scalar),
  T_grad_     (p.get<std::string>("Temperature Gradient Name"),
               dl->qp_vector),
  T_dot_      (p.get<std::string>("Temperature Time Derivative Name"),
               dl->qp_scalar),
  k_          (p.get<std::string>("Thermal Conductivity Name"),
               dl->qp_scalar),
  rho_cp_     (p.get<std::string>("Rho Cp Name"),
               dl->qp_scalar),
  source_     (p.get<std::string>("Source Name"),
               dl->qp_scalar),
  laser_source_     (p.get<std::string>("Laser Source Name"),
               dl->qp_scalar),
  time        (p.get<std::string>("Time Name"),
               dl->workset_scalar),
  deltaTime   (p.get<std::string>("Delta Time Name"),
               dl->workset_scalar),
  residual_   (p.get<std::string>("Residual Name"),
               dl->node_scalar)
{

  this->addDependentField(w_bf_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(T_);
  this->addDependentField(T_grad_);
  this->addDependentField(T_dot_);
  this->addDependentField(k_);
  this->addDependentField(rho_cp_);
  this->addDependentField(source_);
  this->addDependentField(laser_source_);
  this->addDependentField(time);
  this->addDependentField(deltaTime);

  this->addEvaluatedField(residual_);
  
  std::vector<PHX::Device::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_qps_      = dims[2];
  num_dims_     = dims[3];

  Temperature_Name_ = p.get<std::string>("Temperature Name")+"_old";

  term1_.resize(dims[0],num_qps_,num_dims_);
  term2_.resize(dims[0],num_qps_);

  this->setName("PhaseResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PhaseResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(w_bf_,fm);
  this->utils.setFieldData(w_grad_bf_,fm);
  this->utils.setFieldData(T_,fm);
  this->utils.setFieldData(T_grad_,fm);
  this->utils.setFieldData(T_dot_,fm);
  this->utils.setFieldData(k_,fm);
  this->utils.setFieldData(rho_cp_,fm);
  this->utils.setFieldData(source_,fm);
  this->utils.setFieldData(laser_source_,fm);
  this->utils.setFieldData(time,fm);
  this->utils.setFieldData(deltaTime,fm);
  this->utils.setFieldData(residual_,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PhaseResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 //current time
//  const RealType time = workset.current_time;
 // time step
  ScalarT dt = deltaTime(0);
//  std::cout<<"dt value ="<<dt<<std::endl;
  typedef Intrepid2::FunctionSpaceTools FST;
 
  if (dt == 0.0) dt = 1.0e-15;
  //grab old temperature
  Albany::MDArray T_old = (*workset.stateArrayPtr)[Temperature_Name_];
  // Compute Temp rate
// ScalarT T_dot_;

  for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for(std::size_t qp = 0; qp < num_qps_; ++qp) {
        T_dot_(cell,qp) = (T_(cell,qp)-T_old(cell,qp))/dt;
	//std::cout<<"T = "<<T_(cell,qp)<<" "<<"T_old = "<<T_old(cell,qp)<<"----";
    }
	//std::cout<<std::endl;
  }
      // diffusive term
  FST::scalarMultiplyDataData<ScalarT> (term1_,k_,T_grad_);
  FST::integrate<ScalarT>(residual_,term1_,w_grad_bf_,Intrepid2::COMP_CPP,false);

  // transient term
  FST::scalarMultiplyDataData<ScalarT> (term2_,rho_cp_,T_dot_);
  FST::integrate<ScalarT>(residual_,term2_,w_bf_,Intrepid2::COMP_CPP,true);

  // heat source from laser 
  PHAL::scale(laser_source_, -1.0);
  FST::integrate<ScalarT>(residual_,laser_source_,w_bf_,Intrepid2::COMP_CPP,true);  

  // all other problem sources
  PHAL::scale(source_, -1.0);
  FST::integrate<ScalarT>(residual_,source_,w_bf_,Intrepid2::COMP_CPP,true);

/*
//print terms:
   for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
     for(std::size_t node = 0; node < num_nodes_; ++node) {
       std::cout<<"residual = "<< residual_(cell,node)<<std::endl;
     }
   }

  for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for(std::size_t qp = 0; qp < num_qps_; ++qp) {
        std::cout<<"T = "<<T_(cell,qp)<<"----";
    }
	std::cout<<std::endl;
  }
*/
}

//**********************************************************************
}
