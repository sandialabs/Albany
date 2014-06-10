//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
FirstPK<EvalT, Traits>::
FirstPK(Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl) :
  stress_(p.get<std::string>("Stress Name"), dl->qp_tensor),
  def_grad_(p.get<std::string>("DefGrad Name"), dl->qp_tensor),
  first_pk_stress_(p.get<std::string>("First PK Stress Name"), dl->qp_tensor),
  have_pore_pressure_(p.get<bool>("Have Pore Pressure", false)),
  have_stab_pressure_(p.get<bool>("Have Stabilized Pressure", false)),
  small_strain_(p.get<bool>("Small Strain", false))
{
  this->addDependentField(stress_);
  this->addDependentField(def_grad_);

  this->addEvaluatedField(first_pk_stress_);

  this->setName("FirstPK" + PHX::TypeString<EvalT>::value);

  // logic to modify stress in the presence of a pore pressure
  if (have_pore_pressure_) {
    // grab the pore pressure
    PHX::MDField<ScalarT, Cell, QuadPoint>
    tmp(p.get<std::string>("Pore Pressure Name"), dl->qp_scalar);
    pore_pressure_ = tmp;
    // grab Biot's coefficient
    PHX::MDField<ScalarT, Cell, QuadPoint>
    tmp2(p.get<std::string>("Biot Coefficient Name"), dl->qp_scalar);
    biot_coeff_ = tmp2;
    this->addDependentField(pore_pressure_);
    this->addDependentField(biot_coeff_);
  }

  // deal with stabilized pressure
  if (have_stab_pressure_) {
    // grab the pore pressure
    PHX::MDField<ScalarT, Cell, QuadPoint>
    tmp(p.get<std::string>("Pressure Name"), dl->qp_scalar);
    stab_pressure_ = tmp;
    this->addDependentField(stab_pressure_);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  stress_.fieldTag().dataLayout().dimensions(dims);
  num_pts_ = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void FirstPK<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(first_pk_stress_, fm);
  if (have_pore_pressure_) {
    this->utils.setFieldData(pore_pressure_, fm);
    this->utils.setFieldData(biot_coeff_, fm);
  }
  if (have_stab_pressure_) {
    this->utils.setFieldData(stab_pressure_, fm);
  }
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void FirstPK<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //std::cout.precision(15);
  // initilize Tensors
  Intrepid::Tensor<ScalarT> F(num_dims_), P(num_dims_), sig(num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

  if (have_stab_pressure_) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < num_pts_; ++pt) {
        sig.fill(&stress_(cell, pt, 0, 0));
        sig += stab_pressure_(cell,pt)*I - (1.0/3.0)*Intrepid::trace(sig)*I;

        for (std::size_t i = 0; i < num_dims_; i++) {
          for (std::size_t j = 0; j < num_dims_; j++) {
            stress_(cell, pt, i, j) = sig(i, j);
          }
        }
      }
    }
  }

  if (have_pore_pressure_) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < num_pts_; ++pt) {

        // Effective Stress theory
        sig.fill(&stress_(cell, pt, 0, 0));
        sig -= biot_coeff_(cell, pt) * pore_pressure_(cell, pt) * I;

        for (std::size_t i = 0; i < num_dims_; i++) {
          for (std::size_t j = 0; j < num_dims_; j++) {
            stress_(cell, pt, i, j) = sig(i, j);
          }
        }
      }
    }
  }

  // for small deformation, trivially copy Cauchy stress into first PK
  if (small_strain_) { 
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < num_pts_; ++pt) {
        sig.fill(&stress_(cell,pt,0,0));
        for (std::size_t dim0 = 0; dim0 < num_dims_; ++dim0) {
          for (std::size_t dim1 = 0; dim1 < num_dims_; ++dim1) {
            first_pk_stress_(cell,pt,dim0,dim1) = stress_(cell,pt,dim0,dim1);
          }
        }
      }
    }
  } else {
    // for large deformation, map Cauchy stress to 1st PK stress
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < num_pts_; ++pt) {
        F.fill(&def_grad_(cell, pt, 0, 0));
        sig.fill(&stress_(cell, pt, 0, 0));

        // map Cauchy stress to 1st PK
        P = Intrepid::piola(F, sig);

        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            first_pk_stress_(cell,pt,i,j) = P(i, j);
          }
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
}

