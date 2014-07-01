//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace SEE {

//**********************************************************************
template<typename EvalT, typename Traits>
NonlinearPoissonResidual<EvalT, Traits>::
NonlinearPoissonResidual(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  w_bf_       (p.get<std::string>("Weighted BF Name"),
               dl->node_qp_scalar),
  w_grad_bf_  (p.get<std::string>("Weighted Gradient BF Name"),
               dl->node_qp_vector),
  u_          (p.get<std::string>("Unknown Name"),
               dl->qp_scalar),
  u_grad_     (p.get<std::string>("Unknown Gradient Name"),
               dl->qp_vector),
  u_dot_      (p.get<std::string>("Unknown Time Derivative Name"),
               dl->qp_scalar),
  residual_   (p.get<std::string>("Residual Name"),
               dl->node_scalar)
{

  if (p.isType<bool>("Disable Transient"))
    enable_transient_ = !p.get<bool>("Disable Transient");
  else enable_transient_ = true;

  this->addDependentField(w_bf_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(u_);
  this->addDependentField(u_grad_);
  if (enable_transient_) 
    this->addDependentField(u_dot_);
  
  this->addEvaluatedField(residual_);
  
  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_qps_      = dims[2];
  num_dims_     = dims[3];

  this->setName("NonlinearPoissonResidual"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NonlinearPoissonResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(w_bf_,fm);
  this->utils.setFieldData(w_grad_bf_,fm);
  this->utils.setFieldData(u_,fm);
  this->utils.setFieldData(u_grad_,fm);
  if (enable_transient_)
    this->utils.setFieldData(u_dot_,fm);

  this->utils.setFieldData(residual_,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NonlinearPoissonResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // u residual
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < num_nodes_; ++node) {
      residual_(cell,node) = 0.0;
    }
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        for (std::size_t i = 0; i < num_dims_; ++i) {
          residual_(cell,node) +=
            (1.0 + u_(cell,qp)*u_(cell,qp)) *
            u_grad_(cell,qp,i) * w_grad_bf_(cell,node,qp,i);
        }
      }
    }
  }

}

//**********************************************************************
}

