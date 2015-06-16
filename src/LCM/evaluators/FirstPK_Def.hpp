//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>
#include <PHAL_Utilities.hpp>
#ifdef ALBANY_TIMER
#include <chrono>
#endif

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

  this->setName("FirstPK" + PHX::typeAsString<EvalT>());

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

 #ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  int deriv_dims=PHAL::getDerivativeDimensionsFromView(stress_.get_kokkos_view());

  ddims_.push_back(deriv_dims);
  const int num_cells=stress_.dimension(0);
  sig=PHX::MDField<ScalarT,Cell,Dim,Dim>("sig",Teuchos::rcp(new PHX::MDALayout<Cell,Dim,Dim>(num_cells,num_dims_,num_dims_)));
  F=PHX::MDField<ScalarT,Cell,Dim,Dim>("F",Teuchos::rcp(new PHX::MDALayout<Cell,Dim,Dim>(num_cells,num_dims_,num_dims_)));
  P=PHX::MDField<ScalarT,Cell,Dim,Dim>("P",Teuchos::rcp(new PHX::MDALayout<Cell,Dim,Dim>(num_cells,num_dims_,num_dims_)));
  I=PHX::MDField<ScalarT,Dim,Dim>("I",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));

  sig.setFieldData(ViewFactory::buildView(sig.fieldTag(),ddims_));
  F.setFieldData(ViewFactory::buildView(F.fieldTag(),ddims_));
  P.setFieldData(ViewFactory::buildView(P.fieldTag(),ddims_));
  I.setFieldData(ViewFactory::buildView(I.fieldTag(),ddims_));

  for (int i=0; i<num_dims_; i++){
     for (int j=0; j<num_dims_; j++){
        I(i,j)=ScalarT(0.0);
        if (i==j) I(i,j)=ScalarT(1.0);
     }
   }
 #endif

}
//-----------------------------------------------------------------------------
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
//Kokkos functions:
template<typename EvalT, typename Traits>
template <class ArrayT>
KOKKOS_INLINE_FUNCTION
const typename FirstPK<EvalT, Traits>::ScalarT
FirstPK<EvalT, Traits>::
trace (const ArrayT &A, const int cell) const
{

  ScalarT s = 0.0;

  switch (num_dims_) {

    default:
      for (int i = 0; i < num_dims_; ++i) {
        s += A(cell,i,i);
      }
      break;

    case 3:
      s = A(cell,0,0) + A(cell,1,1) + A(cell,2,2);
      break;

    case 2:
      s = A(cell,0,0) + A(cell,1,1);
      break;

  }

  return s;
}

/*//template<typename EvalT, typename Traits>
template <class ArrayOut, class ArrayT>
KOKKOS_INLINE_FUNCTION
//void FirstPK<EvalT, Traits>::
void piola (ArrayOut &C, const ArrayT &F, const ArrayT &A) 
{
 
   const int num_dims_=A.dimension(0);
   switch (num_dims_) {  
 
   default:
     TEUCHOS_TEST_FOR_EXCEPTION( !( (num_dims_ == 2) || (num_dims_ == 3) ), std::invalid_argument,
                                  ">>> ERROR (LCM FirstPK): piola function is defined only for rank-2 or 3 ."); 
    break;

   case 3:

      C(0,0) = A(0,0)*(-F(1,2)*F(2,1) + F(1,1)*F(2,2)) + A(0,1)*( F(0,2)*F(2,1) - F(0,1)*F(2,2)) + A(0,2)*(-F(0,2)*F(1,1) + F(0,1)*F(1,2));
      C(0,1) = A(0,0)*( F(1,2)*F(2,0) - F(1,0)*F(2,2)) + A(0,1)*(-F(0,2)*F(2,0) + F(0,0)*F(2,2)) + A(0,2)*( F(0,2)*F(1,0) - F(0,0)*F(1,2));
      C(0,2) = A(0,0)*(-F(1,1)*F(2,0) + F(1,0)*F(2,1)) + A(0,1)*( F(0,1)*F(2,0) - F(0,0)*F(2,1)) + A(0,2)*(-F(0,1)*F(1,0) + F(0,0)*F(1,1));

      C(1,0) = A(1,0)*(-F(1,2)*F(2,1) + F(1,1)*F(2,2)) + A(1,1)*( F(0,2)*F(2,1) - F(0,1)*F(2,2)) + A(1,2)*(-F(0,2)*F(1,1) + F(0,1)*F(1,2));
      C(1,1) = A(1,0)*( F(1,2)*F(2,0) - F(1,0)*F(2,2)) + A(1,1)*(-F(0,2)*F(2,0) + F(0,0)*F(2,2)) + A(1,2)*( F(0,2)*F(1,0) - F(0,0)*F(1,2));
      C(1,2) = A(1,0)*(-F(1,1)*F(2,0) + F(1,0)*F(2,1))+ A(1,1)*( F(0,1)*F(2,0) - F(0,0)*F(2,1)) + A(1,2)*(-F(0,1)*F(1,0) + F(0,0)*F(1,1));

      C(2,0) = A(2,0)*(-F(1,2)*F(2,1) + F(1,1)*F(2,2)) + A(2,1)*( F(0,2)*F(2,1) - F(0,1)*F(2,2))+ A(2,2)*(-F(0,2)*F(1,1) + F(0,1)*F(1,2));
      C(2,1) = A(2,0)*( F(1,2)*F(2,0) - F(1,0)*F(2,2))+ A(2,1)*(-F(0,2)*F(2,0) + F(0,0)*F(2,2)) + A(2,2)*( F(0,2)*F(1,0) - F(0,0)*F(1,2));
      C(2,2) = A(2,0)*(-F(1,1)*F(2,0) + F(1,0)*F(2,1))+ A(2,1)*( F(0,1)*F(2,0) - F(0,0)*F(2,1)) + A(2,2)*(-F(0,1)*F(1,0) + F(0,0)*F(1,1));

    break;

   case 2:
      C(0,0) = A(0,0)*F(1,1) - A(0,1)*F(0,1);
      C(0,1) = -A(0,0)*F(1,0) + A(0,1)*F(0,0);

      C(1,0) = A(1,0)*F(1,1) - A(1,1)*F(0,1);
      C(1,1) = -A(1,0)*F(1,0) + A(1,1)*F(0,0);
    break;
   }
}
*/
//Kokkos functors:
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void FirstPK<EvalT, Traits>::
operator() (const have_stab_pressure_Tag& tag, const int& cell) const{
/*  for (int pt = 0; pt < num_pts_; ++pt) {
        for (int i = 0; i < num_dims_; i++) {
          for (int j = 0; j < num_dims_; j++) {
             sig(cell,i,j)=stress_(cell, pt, i, j);
             sig(cell,i,j)+= stab_pressure_(cell,pt)*I(i,j)-(1.0/3.0)*trace(sig,cell)*I(i,j);
             stress_(cell, pt, i, j) = sig(cell,i,j);
          }
        }
      }
*/

  ScalarT sig[3][3], traceSig;
  
  for (int pt = 0; pt < num_pts_; ++pt) {
        for (int i = 0; i < num_dims_; i++) {
          for (int j = 0; j < num_dims_; j++) {
             sig[i][j]=stress_(cell, pt, i, j);
             traceSig=0.0;
 
             for (int k = 0; k < num_dims_; ++k) {
              traceSig += sig[k][k];;
             }
    
             sig[i][j]+= stab_pressure_(cell,pt)*I(i,j)-(1.0/3.0)*traceSig*I(i,j);
             stress_(cell, pt, i, j) = sig[i][j];
          }
        }
      }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void FirstPK<EvalT, Traits>::
operator() (const have_pore_pressure_Tag& tag, const int& cell) const{
/*  for (int pt = 0; pt < num_pts_; ++pt) {
       for (int i = 0; i < num_dims_; i++) {
          for (int j = 0; j < num_dims_; j++) {
             sig(cell,i,j)=stress_(cell, pt, i, j);
             sig(cell,i,j)-=biot_coeff_(cell, pt) * pore_pressure_(cell, pt) * I(i,j);
             stress_(cell, pt, i, j)=sig(cell,i,j);
          }
        }
      }
*/
   ScalarT sig[3][3];   
    for (int pt = 0; pt < num_pts_; ++pt) {
       for (int i = 0; i < num_dims_; i++) {
          for (int j = 0; j < num_dims_; j++) {
             sig[i][j]=stress_(cell, pt, i, j);
             sig[i][j]-=biot_coeff_(cell, pt) * pore_pressure_(cell, pt) * I(i,j);
             stress_(cell, pt, i, j)=sig[i][j];
          }
        }
      }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void FirstPK<EvalT, Traits>::
operator() (const small_strain_Tag& tag, const int& cell) const{
    for (int pt = 0; pt < num_pts_; ++pt) {
        for (int dim0 = 0; dim0 < num_dims_; ++dim0) {
          for (int dim1 = 0; dim1 < num_dims_; ++dim1) {
            first_pk_stress_(cell,pt,dim0,dim1) = stress_(cell,pt,dim0,dim1);
          }
        }
      }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void FirstPK<EvalT, Traits>::
operator() (const no_small_strain_Tag& tag, const int& cell) const{

/*
     for (int pt = 0; pt < num_pts_; ++pt) {
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {    
            F(cell,i,j)=def_grad_(cell, pt,i,j);
            sig(cell,i,j)=stress_(cell, pt,i,j);
          }
        } 
        
       //piola(P, F, sig);
        {
          switch (num_dims_) {

           default:
             Kokkos::abort("Error(LCM FirstPK): piola function is defined only for rank-2 or 3 .");
           break;

           case 3:

           P(cell,0,0) = sig(cell,0,0)*(-F(cell,1,2)*F(cell,2,1) + F(cell,1,1)*F(cell,2,2)) + sig(cell,0,1)*( F(cell,0,2)*F(cell,2,1) - F(cell,0,1)*F(cell,2,2)) 
                        + sig(cell,0,2)*(-F(cell,0,2)*F(cell,1,1) + F(cell,0,1)*F(cell,1,2));
           P(cell,0,1) = sig(cell,0,0)*( F(cell,1,2)*F(cell,2,0) - F(cell,1,0)*F(cell,2,2)) + sig(cell,0,1)*(-F(cell,0,2)*F(cell,2,0) + F(cell,0,0)*F(cell,2,2)) 
                        + sig(cell,0,2)*( F(cell,0,2)*F(cell,1,0) - F(cell,0,0)*F(cell,1,2));
           P(cell,0,2) = sig(cell,0,0)*(-F(cell,1,1)*F(cell,2,0) + F(cell,1,0)*F(cell,2,1)) + sig(cell,0,1)*( F(cell,0,1)*F(cell,2,0) - F(cell,0,0)*F(cell,2,1)) 
                        + sig(cell,0,2)*(-F(cell,0,1)*F(cell,1,0) + F(cell,0,0)*F(cell,1,1));

           P(cell,1,0) = sig(cell,1,0)*(-F(cell,1,2)*F(cell,2,1) + F(cell,1,1)*F(cell,2,2)) + sig(cell,1,1)*( F(cell,0,2)*F(cell,2,1) - F(cell,0,1)*F(cell,2,2)) 
                       + sig(cell,1,2)*(-F(cell,0,2)*F(cell,1,1) + F(cell,0,1)*F(cell,1,2));
           P(cell,1,1) = sig(cell,1,0)*( F(cell,1,2)*F(cell,2,0) - F(cell,1,0)*F(cell,2,2)) + sig(cell,1,1)*(-F(cell,0,2)*F(cell,2,0) + F(cell,0,0)*F(cell,2,2)) 
                        + sig(cell,1,2)*( F(cell,0,2)*F(cell,1,0) - F(cell,0,0)*F(cell,1,2));
           P(cell,1,2) = sig(cell,1,0)*(-F(cell,1,1)*F(cell,2,0) + F(cell,1,0)*F(cell,2,1)) + sig(cell,1,1)*( F(cell,0,1)*F(cell,2,0) - F(cell,0,0)*F(cell,2,1))
                        + sig(cell,1,2)*(-F(cell,0,1)*F(cell,1,0) + F(cell,0,0)*F(cell,1,1));

           P(cell,2,0) = sig(cell,2,0)*(-F(cell,1,2)*F(cell,2,1) + F(cell,1,1)*F(cell,2,2))+ sig(cell,2,1)*( F(cell,0,2)*F(cell,2,1) - F(cell,0,1)*F(cell,2,2))
                        + sig(cell,2,2)*(-F(cell,0,2)*F(cell,1,1) + F(cell,0,1)*F(cell,1,2));
           P(cell,2,1) = sig(cell,2,0)*( F(cell,1,2)*F(cell,2,0) - F(cell,1,0)*F(cell,2,2))+ sig(cell,2,1)*(-F(cell,0,2)*F(cell,2,0) + F(cell,0,0)*F(cell,2,2)) 
                        + sig(cell,2,2)*( F(cell,0,2)*F(cell,1,0) - F(cell,0,0)*F(cell,1,2));
           P(cell,2,2) = sig(cell,2,0)*(-F(cell,1,1)*F(cell,2,0) + F(cell,1,0)*F(cell,2,1))+ sig(cell,2,1)*( F(cell,0,1)*F(cell,2,0) - F(cell,0,0)*F(cell,2,1)) 
                        + sig(cell,2,2)*(-F(cell,0,1)*F(cell,1,0) + F(cell,0,0)*F(cell,1,1));

           break;

           case 2:
           P(cell,0,0) = sig(cell,0,0)*F(cell,1,1) - sig(cell,0,1)*F(cell,0,1);
           P(cell,0,1) = -sig(cell,0,0)*F(cell,1,0) + sig(cell,0,1)*F(cell,0,0);

           P(cell,1,0) = sig(cell,1,0)*F(cell,1,1) - sig(cell,1,1)*F(cell,0,1);
           P(cell,1,1) = -sig(cell,1,0)*F(cell,1,0) + sig(cell,1,1)*F(cell,0,0);
            break;
         }
       }
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            first_pk_stress_(cell,pt,i,j) = P(cell,i, j);
          }
        }
      }
*/

   ScalarT sig[3][3], F[3][3], P[3][3];

       for (int pt = 0; pt < num_pts_; ++pt) {
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            F[i][j]=def_grad_(cell, pt,i,j);
            sig[i][j]=stress_(cell, pt,i,j);
          }
        }

       //piola(P, F, sig);
      {
          switch (num_dims_) {

           default:
             Kokkos::abort("Error(LCM FirstPK): piola function is defined only for rank-2 or 3 .");
           break;

           case 3:

           P[0][0] = sig[0][0]*(-F[1][2]*F[2][1] + F[1][1]*F[2][2]) + sig[0][1]*( F[0][2]*F[2][1] - F[0][1]*F[2][2])
                        + sig[0][2]*(-F[0][2]*F[1][1] + F[0][1]*F[1][2]);
           P[0][1] = sig[0][0]*( F[1][2]*F[2][0] - F[1][0]*F[2][2]) + sig[0][1]*(-F[0][2]*F[2][0] + F[0][0]*F[2][2])
                        + sig[0][2]*( F[0][2]*F[1][0] - F[0][0]*F[1][2]);
           P[0][2] = sig[0][0]*(-F[1][1]*F[2][0] + F[1][0]*F[2][1]) + sig[0][1]*( F[0][1]*F[2][0] - F[0][0]*F[2][1])
                        + sig[0][2]*(-F[0][1]*F[1][0] + F[0][0]*F[1][1]);

           P[1][0] = sig[1][0]*(-F[1][2]*F[2][1] + F[1][1]*F[2][2]) + sig[1][1]*( F[0][2]*F[2][1] - F[0][1]*F[2][2])
                       + sig[1][2]*(-F[0][2]*F[1][1] + F[0][1]*F[1][2]);
           P[1][1] = sig[1][0]*( F[1][2]*F[2][0] - F[1][0]*F[2][2]) + sig[1][1]*(-F[0][2]*F[2][0] + F[0][0]*F[2][2]) 
                        + sig[1][2]*( F[0][2]*F[1][0] - F[0][0]*F[1][2]);
           P[1][2] = sig[1][0]*(-F[1][1]*F[2][0] + F[1][0]*F[2][1]) + sig[1][1]*( F[0][1]*F[2][0] - F[0][0]*F[2][1])
                        + sig[1][2]*(-F[0][1]*F[1][0] + F[0][0]*F[1][1]);

           P[2][0] = sig[2][0]*(-F[1][2]*F[2][1] + F[1][1]*F[2][2])+ sig[2][1]*( F[0][2]*F[2][1] - F[0][1]*F[2][2])
                        + sig[2][2]*(-F[0][2]*F[1][1] + F[0][1]*F[1][2]);
           P[2][1] = sig[2][0]*( F[1][2]*F[2][0] - F[1][0]*F[2][2])+ sig[2][1]*(-F[0][2]*F[2][0] + F[0][0]*F[2][2]) 
                        + sig[2][2]*( F[0][2]*F[1][0] - F[0][0]*F[1][2]);
           P[2][2] = sig[2][0]*(-F[1][1]*F[2][0] + F[1][0]*F[2][1])+ sig[2][1]*( F[0][1]*F[2][0] - F[0][0]*F[2][1]) 
                        + sig[2][2]*(-F[0][1]*F[1][0] + F[0][0]*F[1][1]);

           break;

           case 2:
           P[0][0] = sig[0][0]*F[1][1] - sig[0][1]*F[0][1];
           P[0][1] = -sig[0][0]*F[1][0] + sig[0][1]*F[0][0];

           P[1][0] = sig[1][0]*F[1][1] - sig[1][1]*F[0][1];
           P[1][1] = -sig[1][0]*F[1][0] + sig[1][1]*F[0][0];
            break;
         }
       }
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            first_pk_stress_(cell,pt,i,j) = P[i][j];
          }
        }
      }

}
#endif
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void FirstPK<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //std::cout.precision(15);
  // initilize Tensors
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Intrepid::Tensor<ScalarT> F(num_dims_), P(num_dims_), sig(num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

  if (have_stab_pressure_) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        sig.fill(stress_,cell, pt,0,0);
        sig += stab_pressure_(cell,pt)*I - (1.0/3.0)*Intrepid::trace(sig)*I;

        for (int i = 0; i < num_dims_; i++) {
          for (int j = 0; j < num_dims_; j++) {
            stress_(cell, pt, i, j) = sig(i, j);
          }
        }
      }
    }
  }

  if (have_pore_pressure_) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {

        // Effective Stress theory
        sig.fill(stress_,cell, pt,0,0);
        sig -= biot_coeff_(cell, pt) * pore_pressure_(cell, pt) * I;

        for (int i = 0; i < num_dims_; i++) {
          for (int j = 0; j < num_dims_; j++) {
            stress_(cell, pt, i, j) = sig(i, j);
          }
        }
      }
    }
  }

  // for small deformation, trivially copy Cauchy stress into first PK
  if (small_strain_) { 
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        sig.fill(stress_,cell,pt,0,0);
        for (int dim0 = 0; dim0 < num_dims_; ++dim0) {
          for (int dim1 = 0; dim1 < num_dims_; ++dim1) {
            first_pk_stress_(cell,pt,dim0,dim1) = stress_(cell,pt,dim0,dim1);
          }
        }
      }
    }
  } else {
    // for large deformation, map Cauchy stress to 1st PK stress
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        F.fill(def_grad_,cell, pt,0,0);
        sig.fill(stress_,cell, pt,0,0);


        // map Cauchy stress to 1st PK
        P = Intrepid::piola(F, sig);

        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            first_pk_stress_(cell,pt,i,j) = P(i, j);
          }
        }
      }
    }
  }
#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

   if (have_stab_pressure_) 
     Kokkos::parallel_for(have_stab_pressure_Policy(0,workset.numCells),*this);
   if (have_pore_pressure_)
     Kokkos::parallel_for(have_pore_pressure_Policy(0,workset.numCells),*this);
   if (small_strain_)
     Kokkos::parallel_for(small_strain_Policy(0,workset.numCells),*this);
   else
     Kokkos::parallel_for(no_small_strain_Policy(0,workset.numCells),*this);
#ifdef ALBANY_TIMER
 PHX::Device::fence();
 auto elapsed = std::chrono::high_resolution_clock::now() - start;
 long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
 long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
 std::cout<< "First_PK time = "  << millisec << "  "  << microseconds << std::endl;
#endif
#endif
}
//------------------------------------------------------------------------------
}

