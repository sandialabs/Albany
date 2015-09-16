//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor_Mechanics.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>

using std::string;
//using int;

namespace LCM
{

//
//
//
template<typename EvalT, typename Traits>
CMResidualFine<EvalT, Traits>::
CMResidualFine(Teuchos::ParameterList & p,
    Teuchos::RCP<Albany::Layouts> const & dl) :
      stress_(p.get<string>("Stress Name"), dl->qp_tensor),
      def_grad_(p.get<string>("DefGrad Name"), dl->qp_tensor),
      w_grad_bf_(p.get<string>("Weighted Gradient BF Name"),dl->node_qp_vector),
      w_bf_(p.get<string>("Weighted BF Name"), dl->node_qp_scalar),
      residual_(p.get<string>("Residual Name"), dl->node_vector),
      have_body_force_(p.get<bool>("Have Body Force", false))
{
  this->addDependentField(stress_);
  this->addDependentField(def_grad_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(w_bf_);

  this->addEvaluatedField(residual_);

  this->setName("CMResidualFine" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type>
  dims;

  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  int num_nodes_ = dims[1];
  int num_pts_ = dims[2];
  int num_dims_ = dims[3];

  Teuchos::RCP<ParamLib>
  paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
}

//
//
//
template<typename EvalT, typename Traits>
void CMResidualFine<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(w_grad_bf_, fm);
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(residual_, fm);
  if (have_body_force_ == true) {
    this->utils.setFieldData(body_force_, fm);
  }
}

//
//
//
template<typename EvalT, typename Traits>
void CMResidualFine<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Intrepid::Tensor<ScalarT>
  F(num_dims_), P(num_dims_), sig(num_dims_);

  Intrepid::Tensor<ScalarT>
  I(Intrepid::eye<ScalarT>(num_dims_));

  // initialize residual
  for (int cell = 0; cell < workset.numCells; ++cell) {
     for (int node = 0; node < num_nodes_; ++node) {
       for (int dim = 0; dim < num_dims_; ++dim) {
         residual_(cell, node, dim) = 0.0;
       }
     }
     for (int pt = 0; pt < num_pts_; ++pt) {
       F.fill(def_grad_,cell, pt,0,0);
       sig.fill(stress_,cell, pt,0,0);

       // map Cauchy stress to 1st PK
       P = Intrepid::piola(F, sig);

       for (size_t node = 0; node < num_nodes_; ++node) {
         for (size_t i = 0; i < num_dims_; ++i) {
           for (size_t j = 0; j < num_dims_; ++j) {
             residual_(cell, node, i) +=
                 P(i, j) * w_grad_bf_(cell, node, pt, j);
           }
         }
       }
     }
   }

  // optional body force
  if (have_body_force_) {
    for (size_t cell = 0; cell < workset.numCells; ++cell) {
      for (size_t node = 0; node < num_nodes_; ++node) {
        for (size_t pt = 0; pt < num_pts_; ++pt) {
          for (size_t dim = 0; dim < num_dims_; ++dim) {
            residual_(cell, node, dim) +=
                w_bf_(cell, node, pt) * body_force_(cell, pt, dim);
          }
        }
      }
    }
  }
}

}

