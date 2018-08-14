//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <PHAL_Utilities.hpp>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"
#ifdef ALBANY_TIMER
#include <chrono>
#endif
#include <typeinfo>

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
Kinematics<EvalT, Traits>::Kinematics(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : grad_u_(p.get<std::string>("Gradient QP Variable Name"), dl->qp_tensor),
      weights_(p.get<std::string>("Weights Name"), dl->qp_scalar),
      def_grad_(p.get<std::string>("DefGrad Name"), dl->qp_tensor),
      j_(p.get<std::string>("DetDefGrad Name"), dl->qp_scalar),
      weighted_average_(p.get<bool>("Weighted Volume Average J", false)),
      alpha_(p.get<RealType>("Average J Stabilization Parameter", 0.0)),
      needs_vel_grad_(false),
      needs_strain_(false)
{
  if (p.isType<bool>("Velocity Gradient Flag"))
    needs_vel_grad_ = p.get<bool>("Velocity Gradient Flag");
  if (p.isType<std::string>("Strain Name")) {
    needs_strain_ = true;
    strain_ =
        decltype(strain_)(p.get<std::string>("Strain Name"), dl->qp_tensor);
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
    vel_grad_ = decltype(vel_grad_)(
        p.get<std::string>("Velocity Gradient Name"), dl->qp_tensor);
    this->addEvaluatedField(vel_grad_);
  }

  this->setName("Kinematics" + PHX::typeAsString<EvalT>());

  if (def_grad_rc_.init(p, p.get<std::string>("DefGrad Name")))
    this->addDependentField(def_grad_rc_());
  if (def_grad_rc_) {
    u_ = decltype(u_)(p.get<std::string>("Displacement Name"), dl->node_vector);
    this->addDependentField(u_);
  }
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
Kinematics<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights_, fm);
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(j_, fm);
  this->utils.setFieldData(grad_u_, fm);
  if (needs_strain_) this->utils.setFieldData(strain_, fm);
  if (needs_vel_grad_) this->utils.setFieldData(vel_grad_, fm);
  if (def_grad_rc_) this->utils.setFieldData(def_grad_rc_(), fm);
  if (def_grad_rc_) this->utils.setFieldData(u_, fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
bool
Kinematics<EvalT, Traits>::check_det(
    typename Traits::EvalData workset,
    int                       cell,
    int                       pt)
{
  minitensor::Tensor<ScalarT> F(num_dims_);
  F.fill(def_grad_, cell, pt, 0, 0);
  j_(cell, pt) = minitensor::det(F);
  bool neg_det = false;
  if (pt == 0 && j_(cell, pt) < 1e-16) {
    neg_det = true;
    std::cout << "amb: (neg det) rcu Kinematics check_det " << j_(cell, pt)
              << " " << cell << " " << pt << "\nF_incr = [" << F << "];\n";
    const Teuchos::ArrayRCP<GO>& gid = workset.wsElNodeID[cell];
    std::cout << "gid_matlab = [";
    for (int i = 0; i < gid.size(); ++i) std::cout << " " << gid[i] + 1;
    std::cout << "];\n";
  }
  return neg_det;
}

template <typename EvalT, typename Traits>
void
Kinematics<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  minitensor::Tensor<ScalarT> F(num_dims_), strain(num_dims_), gradu(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  // Compute DefGrad tensor from displacement gradient
  if (!def_grad_rc_) {
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        gradu.fill(grad_u_, cell, pt, 0, 0);
        F            = I + gradu;
        j_(cell, pt) = minitensor::det(F);
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            def_grad_(cell, pt, i, j) = F(i, j);
          }
        }
      }
    }
  } else {
    bool first = true;
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        gradu.fill(grad_u_, cell, pt, 0, 0);
        F = I + gradu;
        for (int i = 0; i < num_dims_; ++i)
          for (int j = 0; j < num_dims_; ++j)
            def_grad_(cell, pt, i, j) = F(i, j);
        if (first && check_det(workset, cell, pt)) first = false;
        // F[n,0] = F[n,n-1] F[n-1,0].
        def_grad_rc_.multiplyInto<ScalarT>(def_grad_, cell, pt);
        F.fill(def_grad_, cell, pt, 0, 0);
        j_(cell, pt) = minitensor::det(F);
      }
    }
  }

  if (weighted_average_) {
    ScalarT jbar, weighted_jbar, volume;
    for (int cell(0); cell < workset.numCells; ++cell) {
      jbar   = 0.0;
      volume = 0.0;
      for (int pt(0); pt < num_pts_; ++pt) {
        jbar += weights_(cell, pt) * j_(cell, pt);
        volume += weights_(cell, pt);
      }
      jbar /= volume;

      for (int pt(0); pt < num_pts_; ++pt) {
        weighted_jbar = (1 - alpha_) * jbar + alpha_ * j_(cell, pt);
        F.fill(def_grad_, cell, pt, 0, 0);
        const ScalarT p = std::pow((weighted_jbar / j_(cell, pt)), 1. / 3.);
        F *= p;
        j_(cell, pt) = weighted_jbar;
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            def_grad_(cell, pt, i, j) = F(i, j);
          }
        }
      }
    }
  }

  if (needs_strain_) {
    if (!def_grad_rc_) {
      for (int cell(0); cell < workset.numCells; ++cell) {
        for (int pt(0); pt < num_pts_; ++pt) {
          gradu.fill(grad_u_, cell, pt, 0, 0);
          strain = 0.5 * (gradu + minitensor::transpose(gradu));
          for (int i(0); i < num_dims_; ++i) {
            for (int j(0); j < num_dims_; ++j) {
              strain_(cell, pt, i, j) = strain(i, j);
            }
          }
        }
      }
    } else {
      for (int cell = 0; cell < workset.numCells; ++cell) {
        for (int pt = 0; pt < num_pts_; ++pt) {
          F.fill(def_grad_, cell, pt, 0, 0);
          gradu = F - I;
          // dU/dx[0] = dx[n]/dx[0] - dx[0]/dx[0] = F[n,0] - I.
          // strain = 1/2 (dU/dx[0] + dU/dx[0]^T).
          strain = 0.5 * (gradu + minitensor::transpose(gradu));
          for (int i = 0; i < num_dims_; ++i)
            for (int j = 0; j < num_dims_; ++j)
              strain_(cell, pt, i, j) = strain(i, j);
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
}  // namespace LCM
