//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <Intrepid_MiniTensor.h>

#include <typeinfo>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  Kinematics<EvalT, Traits>::
  Kinematics(Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
    grad_u_   (p.get<std::string>("Gradient QP Variable Name"),dl->qp_tensor),
    weights_  (p.get<std::string>("Weights Name"),dl->qp_scalar),
    def_grad_ (p.get<std::string>("DefGrad Name"),dl->qp_tensor),
    j_        (p.get<std::string>("DetDefGrad Name"),dl->qp_scalar),
    weighted_average_(p.get<bool>("Weighted Volume Average J", false)),
    alpha_(p.get<RealType>("Average J Stabilization Parameter", 0.0)),
    needs_vel_grad_(false),
    needs_strain_(false)
  {
    if ( p.isType<bool>("Velocity Gradient Flag") )
      needs_vel_grad_ = p.get<bool>("Velocity Gradient Flag");
    if ( p.isType<std::string>("Strain Name") ) {
      needs_strain_ = true;
      PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim>
        tmp(p.get<std::string>("Strain Name"),dl->qp_tensor);
      strain_ = tmp;
      this->addEvaluatedField(strain_);
    }

    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    num_pts_  = dims[1];
    num_dims_ = dims[2];

    this->addDependentField(grad_u_);
    this->addDependentField(weights_);

    this->addEvaluatedField(def_grad_);
    this->addEvaluatedField(j_);

    if (needs_vel_grad_) {
      PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim>
        tmp(p.get<std::string>("Velocity Gradient Name"),dl->qp_tensor);
      vel_grad_ = tmp;
      this->addEvaluatedField(vel_grad_);
    }

    this->setName("Kinematics"+PHX::TypeString<EvalT>::value);

  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Kinematics<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(weights_,fm);
    this->utils.setFieldData(def_grad_,fm);
    this->utils.setFieldData(j_,fm);
    this->utils.setFieldData(grad_u_,fm);
    if (needs_strain_) this->utils.setFieldData(strain_,fm);
    if (needs_vel_grad_) this->utils.setFieldData(vel_grad_,fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Kinematics<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    Intrepid::Tensor<ScalarT> F(num_dims_), strain(num_dims_), gradu(num_dims_);
    Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

    // Compute DefGrad tensor from displacement gradient
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        gradu.fill( &grad_u_(cell,pt,0,0) );
        F = I + gradu;
        j_(cell,pt) = Intrepid::det(F);
        for (std::size_t i(0); i < num_dims_; ++i) {
          for (std::size_t j(0); j < num_dims_; ++j) {
            def_grad_(cell,pt,i,j) = F(i,j);
          }
        }
      }
    }

    if (weighted_average_) {
      ScalarT jbar, weighted_jbar, volume;
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        jbar = 0.0;
        volume = 0.0;
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          jbar += weights_(cell,pt) * std::log( j_(cell,pt) );
          volume += weights_(cell,pt);
        }
        jbar /= volume;

        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          weighted_jbar = 
            std::exp( (1-alpha_) * jbar + alpha_ * std::log( j_(cell,pt) ) );
          F.fill( &def_grad_(cell,pt,0,0) );
          F = F*std::pow( (weighted_jbar/j_(cell,pt)), 1./3. );
          j_(cell,pt) = weighted_jbar;
          for (std::size_t i(0); i < num_dims_; ++i) {
            for (std::size_t j(0); j < num_dims_; ++j) {
              def_grad_(cell,pt,i,j) = F(i,j);
            }
          }
        }
      }
    }

    if (needs_strain_) {
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          gradu.fill( &grad_u_(cell,pt,0,0) );
          strain = 0.5 * (gradu + Intrepid::transpose(gradu));
          for (std::size_t i(0); i < num_dims_; ++i) {
            for (std::size_t j(0); j < num_dims_; ++j) {
              strain_(cell,pt,i,j) = strain(i,j);
            }
          }
        }
      }
    }
  }
  //----------------------------------------------------------------------------
}
