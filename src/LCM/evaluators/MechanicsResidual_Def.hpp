//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor_Mechanics.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
MechanicsResidual<EvalT, Traits>::
MechanicsResidual(Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  stress_(p.get<std::string>("Stress Name"), dl->qp_tensor),
  w_grad_bf_(p.get<std::string>("Weighted Gradient BF Name"),
             dl->node_qp_vector),
  w_bf_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
  residual_(p.get<std::string>("Residual Name"), dl->node_vector),
  have_body_force_(false),
  density_(p.get<RealType>("Density", 1.0))
{
  this->addDependentField(stress_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(w_bf_);

  this->addEvaluatedField(residual_);

  if (p.isType<bool>("Disable Dynamics"))
    enable_dynamics_ = !p.get<bool>("Disable Dynamics");
  else enable_dynamics_ = true;

  if (enable_dynamics_) {
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> tmp
      (p.get<std::string>("Acceleration Name"), dl->qp_vector);
    acceleration_ = tmp;
    this->addDependentField(acceleration_);
  }

  this->setName("MechanicsResidual" + PHX::TypeString<EvalT>::value);

  if (have_body_force_) {
    // grab the pore pressure
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim>
    tmp(p.get<std::string>("Body Force Name"), dl->qp_vector);
    body_force_ = tmp;
    this->addDependentField(body_force_);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_ = dims[2];
  num_dims_ = dims[3];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void MechanicsResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(w_grad_bf_, fm);
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(residual_, fm);
  if (have_body_force_) {
    this->utils.setFieldData(body_force_, fm);
  }
  if (enable_dynamics_) {
    this->utils.setFieldData(acceleration_, fm);
  }
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void MechanicsResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //std::cout.precision(15);
  // initilize Tensors
  // Intrepid::Tensor<ScalarT> F(num_dims_), P(num_dims_), sig(num_dims_);
  // Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

  // for large deformation, map Cauchy stress to 1st PK stress
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < num_nodes_; ++node) {
      for (std::size_t dim = 0; dim < num_dims_; ++dim) {
        residual_(cell, node, dim) = 0.0;
      }
    }
    for (std::size_t pt = 0; pt < num_pts_; ++pt) {
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            residual_(cell, node, i) +=
              stress_(cell, pt, i, j) * w_grad_bf_(cell, node, pt, j);
          }
        }
      }
    }
  }

  // optional body force
  if (have_body_force_) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        for (std::size_t pt = 0; pt < num_pts_; ++pt) {
          for (std::size_t dim = 0; dim < num_dims_; ++dim) {
            residual_(cell, node, dim) +=
                w_bf_(cell, node, pt) * body_force_(cell, pt, dim);
          }
        }
      }
    }
  }

  // dynamic term
  if (workset.transientTerms && enable_dynamics_) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < num_nodes_; ++node) {
        for (std::size_t pt=0; pt < num_pts_; ++pt) {
          for (std::size_t dim=0; dim < num_dims_; ++dim) {
            residual_(cell,node,dim) += density_ *
              acceleration_(cell,pt,dim) * w_bf_(cell,node,pt);
          }
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
}

