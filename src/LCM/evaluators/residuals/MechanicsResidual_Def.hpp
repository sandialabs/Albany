//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <MiniTensor_Mechanics.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>

#include "Albany_config.h"

#ifdef ALBANY_TIMER
#include <chrono>
#endif

// IKT: uncomment to turn on debug output
//#define DEBUG_OUTPUT

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
MechanicsResidual<EvalT, Traits>::MechanicsResidual(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : stress_(p.get<std::string>("Stress Name"), dl->qp_tensor),
      w_grad_bf_(
          p.get<std::string>("Weighted Gradient BF Name"),
          dl->node_qp_vector),
      w_bf_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      residual_(p.get<std::string>("Residual Name"), dl->node_vector),
      mass_(p.get<std::string>("Analytic Mass Name"), dl->node_vector),
      have_body_force_(p.isType<bool>("Has Body Force")),
      density_(p.get<RealType>("Density", 1.0))
{
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
  this->addDependentField(stress_);
  this->addDependentField(w_grad_bf_);
  this->addDependentField(w_bf_);

  this->addEvaluatedField(residual_);

  if (p.isType<bool>("Disable Dynamics"))
    enable_dynamics_ = !p.get<bool>("Disable Dynamics");
  else
    enable_dynamics_ = true;

  use_analytic_mass_ = p.get<bool>("Use Analytic Mass");
#ifdef DEBUG_OUTPUT
  *out << "IKT use_analytic_mass_ = " << use_analytic_mass_ << "\n";
#endif
  if (enable_dynamics_) {
    acceleration_ = decltype(acceleration_)(
        p.get<std::string>("Acceleration Name"), dl->qp_vector);
    this->addDependentField(acceleration_);
    if (use_analytic_mass_) this->addDependentField(mass_);
  }

  this->setName("MechanicsResidual" + PHX::typeAsString<EvalT>());

  if (have_body_force_) {
    body_force_ = decltype(body_force_)(
        p.get<std::string>("Body Force Name"), dl->qp_vector);
    this->addDependentField(body_force_);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library");

  if (def_grad_rc_.init(p, "F")) this->addDependentField(def_grad_rc_());
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
MechanicsResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(w_grad_bf_, fm);
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(residual_, fm);
  if (have_body_force_) { this->utils.setFieldData(body_force_, fm); }
  if (enable_dynamics_) {
    this->utils.setFieldData(acceleration_, fm);
    if (use_analytic_mass_) this->utils.setFieldData(mass_, fm);
  }
  if (def_grad_rc_) this->utils.setFieldData(def_grad_rc_(), fm);
}

// ***************************************************************************
// Kokkos kernels
//
template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
MechanicsResidual<EvalT, Traits>::compute_Stress(const int i) const
{
  for (int node = 0; node < num_nodes_; ++node) {
    for (int dim = 0; dim < num_dims_; ++dim) {
      residual_(i, node, dim) = typename EvalT::ScalarT(0.0);
    }
  }
  for (int pt = 0; pt < num_pts_; ++pt) {
    for (int node = 0; node < num_nodes_; ++node) {
      for (int dim = 0; dim < num_dims_; ++dim) {
        for (int j = 0; j < num_dims_; ++j) {
          residual_(i, node, dim) +=
              stress_(i, pt, dim, j) * w_grad_bf_(i, node, pt, j);
        }
      }
    }
  }
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
MechanicsResidual<EvalT, Traits>::compute_BodyForce(const int i) const
{
  for (int node = 0; node < num_nodes_; ++node) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      for (int dim = 0; dim < num_dims_; ++dim) {
        residual_(i, node, dim) -= w_bf_(i, node, pt) * body_force_(i, pt, dim);
      }
    }
  }
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
MechanicsResidual<EvalT, Traits>::compute_Acceleration(const int i) const
{
  for (int node = 0; node < num_nodes_; ++node) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      for (int dim = 0; dim < num_dims_; ++dim) {
        residual_(i, node, dim) +=
            density_ * acceleration_(i, pt, dim) * w_bf_(i, node, pt);
      }
    }
  }
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
MechanicsResidual<EvalT, Traits>::operator()(
    const residual_Tag& tag,
    const int&          i) const
{
  this->compute_Stress(i);
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
MechanicsResidual<EvalT, Traits>::operator()(
    const residual_haveBodyForce_Tag& tag,
    const int&                        i) const
{
  this->compute_Stress(i);
  this->compute_BodyForce(i);
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
MechanicsResidual<EvalT, Traits>::operator()(
    const residual_haveBodyForce_and_dynamic_Tag& tag,
    const int&                                    i) const
{
  this->compute_Stress(i);
  this->compute_BodyForce(i);
  this->compute_Acceleration(i);
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
MechanicsResidual<EvalT, Traits>::operator()(
    const residual_have_dynamic_Tag& tag,
    const int&                       i) const
{
  this->compute_Stress(i);
  this->compute_Acceleration(i);
}

// ***************************************************************************
template <typename EvalT, typename Traits>
void
MechanicsResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < num_nodes_; ++node)
      for (int dim = 0; dim < num_dims_; ++dim)
        residual_(cell, node, dim) = ScalarT(0);
    if (def_grad_rc_) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        minitensor::Tensor<RealType> F(num_dims_);
        F.fill(def_grad_rc_(), cell, pt, 0, 0);
        const RealType F_det = minitensor::det(F);
        for (int node = 0; node < num_nodes_; ++node) {
          MeshScalarT w[3];
          AAdapt::rc::transformWeightedGradientBF(
              def_grad_rc_, F_det, w_grad_bf_, cell, pt, node, w);
          for (int i = 0; i < num_dims_; ++i)
            for (int j = 0; j < num_dims_; ++j)
              residual_(cell, node, i) += stress_(cell, pt, i, j) * w[j];
        }
      }
    } else {
      for (int pt = 0; pt < num_pts_; ++pt) {
        for (int node = 0; node < num_nodes_; ++node) {
          for (int i = 0; i < num_dims_; ++i)
            for (int j = 0; j < num_dims_; ++j)
              residual_(cell, node, i) +=
                  stress_(cell, pt, i, j) * w_grad_bf_(cell, node, pt, j);
        }
      }
    }
  }

  // optional body force
  if (have_body_force_) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < num_nodes_; ++node) {
        for (int pt = 0; pt < num_pts_; ++pt) {
          for (int dim = 0; dim < num_dims_; ++dim) {
            residual_(cell, node, dim) -=
                w_bf_(cell, node, pt) * body_force_(cell, pt, dim);
          }
        }
      }
    }
  }

  // dynamic term
  if (workset.transientTerms && enable_dynamics_) {
    // If transient problem and not using analytic mass, enable acceleration
    // terms. This is similar to what is done in Peridigm when mass is passed
    // from peridigm rather than computed in Albany; see, e.g.,
    // albanyIsCreatingMassMatrix-based logic in PeridigmForce_Def.hpp
    if (!use_analytic_mass_) {  // not using analytic mass
      for (int cell = 0; cell < workset.numCells; ++cell) {
        for (int node = 0; node < num_nodes_; ++node) {
          for (int pt = 0; pt < num_pts_; ++pt) {
            for (int dim = 0; dim < num_dims_; ++dim) {
              residual_(cell, node, dim) += density_ *
                                            acceleration_(cell, pt, dim) *
                                            w_bf_(cell, node, pt);
            }
          }
        }
      }
    } else {  // using analytic mass: add contribution from analytic mass
              // evaluator
      for (int cell = 0; cell < workset.numCells; ++cell) {
        for (int node = 0; node < num_nodes_; ++node) {
          for (int dim = 0; dim < num_dims_; ++dim) {
            residual_(cell, node, dim) += mass_(cell, node, dim);
          }
        }
      }
    }
  }
#ifdef DEBUG_OUTPUT
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
  for (int cell = 0; cell < workset.numCells; ++cell) {
    if (cell == 0) {
      for (int node = 0; node < this->num_nodes_; ++node) {
        for (int dim = 0; dim < this->num_dims_; ++dim) {
          *out << "IKT node, dim, residual = " << node << ", " << dim << ", "
               << residual_(cell, node, dim) << "\n";
        }
      }
    }
  }
#endif
}
//------------------------------------------------------------------------------
}  // namespace LCM
