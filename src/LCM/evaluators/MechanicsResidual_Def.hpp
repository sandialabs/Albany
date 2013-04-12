//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor_Mechanics.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  MechanicsResidual<EvalT, Traits>::
  MechanicsResidual(Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
    stress_   (p.get<std::string>("Stress Name"),dl->qp_tensor),
    def_grad_ (p.get<std::string>("DefGrad Name"),dl->qp_tensor),
    w_grad_bf_(p.get<std::string>("Weighted Gradient BF Name"),dl->node_qp_vector),
    w_bf_     (p.get<std::string>("Weighted BF Name"),dl->node_qp_scalar),
    residual_ (p.get<std::string>("Residual Name"),dl->node_vector),
    have_pore_pressure_(p.get<bool>("Have Pore Pressure",false)),
    have_body_force_   (p.get<bool>("Have Body Force",false))
  {
    this->addDependentField(stress_);
    this->addDependentField(def_grad_);
    this->addDependentField(w_grad_bf_);
    this->addDependentField(w_bf_);

    this->addEvaluatedField(residual_);

    this->setName("MechanicsResidual"+PHX::TypeString<EvalT>::value);

    // logic to modify stress in the presence of a pore pressure
    if (have_pore_pressure_) {
      // grab the pore pressure
      PHX::MDField<ScalarT,Cell,QuadPoint> 
        tmp(p.get<string>("Pore Pressure Name"), dl->qp_scalar);
      pore_pressure_ = tmp;
      // grab Boit's coefficient
      PHX::MDField<ScalarT,Cell,QuadPoint> 
        tmp2(p.get<string>("Biot Coefficient Name"), dl->qp_scalar);
      biot_coeff_ = tmp2;
      this->addDependentField(pore_pressure_);
      this->addDependentField(biot_coeff_);
    }

    if (have_body_force_) {
      // grab the pore pressure
      PHX::MDField<ScalarT,Cell,QuadPoint,Dim> 
        tmp(p.get<string>("Body Force Name"), dl->qp_vector);
      body_force_ = tmp;
      this->addDependentField(body_force_);
    }

    std::vector<PHX::DataLayout::size_type> dims;
    w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
    num_nodes_ = dims[1];
    num_pts_   = dims[2];
    num_dims_  = dims[3];
    int worksetSize = dims[0];

    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library");
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void MechanicsResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress_,fm);
    this->utils.setFieldData(def_grad_,fm);
    this->utils.setFieldData(w_grad_bf_,fm);
    this->utils.setFieldData(w_bf_,fm);
    this->utils.setFieldData(residual_,fm);
    if (have_pore_pressure_) {
      this->utils.setFieldData(pore_pressure_,fm);    
      this->utils.setFieldData(biot_coeff_,fm);
    }
    if (have_body_force_) {
      this->utils.setFieldData(body_force_,fm);
    }
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void MechanicsResidual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    cout.precision(15);
    // initilize Tensors
    Intrepid::Tensor<ScalarT> F(num_dims_), P(num_dims_), sig(num_dims_);
    Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

    if (have_pore_pressure_) {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t pt=0; pt < num_pts_; ++pt) {

          // Effective Stress theory
          sig.fill( &stress_(cell,pt,0,0) );
          sig -= biot_coeff_(cell,pt) * pore_pressure_(cell,pt) * I;

          for (std::size_t i=0; i<num_dims_; i++) {
            for (std::size_t j=0; j<num_dims_; j++) {
              stress_(cell,pt,i,j) = sig(i,j);
            }
          }
        }
      }
    }

    // initilize residual
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < num_nodes_; ++node) {
        for (std::size_t dim=0; dim<num_dims_; ++dim)  {
          residual_(cell,node,dim)=0.0;
        }
      }
      for (std::size_t pt=0; pt < num_pts_; ++pt) {
        F.fill( &def_grad_(cell,pt,0,0) );
        sig.fill( &stress_(cell,pt,0,0) );

        // map Cauchy stress to 1st PK
        P = Intrepid::piola(F,sig);

        for (std::size_t node=0; node < num_nodes_; ++node) {
          for (std::size_t i=0; i<num_dims_; ++i) {
            for (std::size_t j=0; j<num_dims_; ++j) {
              residual_(cell,node,i) += 
                P(i, j) * w_grad_bf_(cell, node, pt, j);
            } 
          } 
        } 
      } 
    }
    
    // optional body force
    if (have_body_force_) {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < num_nodes_; ++node) {
          for (std::size_t pt=0; pt < num_pts_; ++pt) {
            for (std::size_t dim=0; dim<num_dims_; ++dim)  {
              residual_(cell,node,dim) += 
                w_bf_(cell,node,pt) * body_force_(cell,pt,dim);
            }
          }
        }
      }
    }
  }
  //----------------------------------------------------------------------------
}

