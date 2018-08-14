//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>
#include <typeinfo>
#include "PHAL_Utilities.hpp"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
TransportResidual<EvalT, Traits>::TransportResidual(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : scalar_(p.get<std::string>("Scalar Variable Name"), dl->qp_scalar),
      scalar_grad_(
          p.get<std::string>("Scalar Gradient Variable Name"),
          dl->qp_vector),
      weights_(p.get<std::string>("Weights Name"), dl->qp_scalar),
      w_bf_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      w_grad_bf_(
          p.get<std::string>("Weighted Gradient BF Name"),
          dl->node_qp_vector),
      residual_(p.get<std::string>("Residual Name"), dl->node_scalar),
      have_source_(p.get<bool>("Have Source", false)),
      have_second_source_(p.get<bool>("Have Second Source", false)),
      have_transient_(p.get<bool>("Have Transient", false)),
      have_diffusion_(p.get<bool>("Have Diffusion", false)),
      have_convection_(p.get<bool>("Have Convection", false)),
      have_species_coupling_(p.get<bool>("Have Species Coupling", false)),
      have_stabilization_(p.get<bool>("Have Stabilization", false)),
      have_contact_(p.get<bool>("Have Contact", false)),
      have_mechanics_(p.get<bool>("Have Mechanics", false)),
      SolutionType_(p.get<std::string>("Solution Method Type")),
      num_nodes_(0),
      num_pts_(0),
      num_dims_(0)
{
  this->addDependentField(scalar_);
  this->addDependentField(scalar_grad_);
  this->addDependentField(weights_);
  this->addDependentField(w_bf_);
  this->addDependentField(w_grad_bf_);

  if (have_source_) {
    source_ =
        decltype(source_)(p.get<std::string>("Source Name"), dl->qp_scalar);
    this->addDependentField(source_);
  }

  if (have_second_source_) {
    second_source_ = decltype(second_source_)(
        p.get<std::string>("Second Source Name"), dl->qp_scalar);
    this->addDependentField(second_source_);
  }

  if (have_transient_) {
    scalar_dot_ = decltype(scalar_dot_)(
        p.get<std::string>("Scalar Dot Name"), dl->qp_scalar);
    this->addDependentField(scalar_dot_);

    transient_coeff_ = decltype(transient_coeff_)(
        p.get<std::string>("Transient Coefficient Name"), dl->qp_scalar);
    this->addDependentField(transient_coeff_);

    delta_time_ = decltype(delta_time_)(
        p.get<std::string>("Delta Time Name"), dl->workset_scalar);
    this->addDependentField(delta_time_);
  }

  if (have_diffusion_) {
    diffusivity_ = decltype(diffusivity_)(
        p.get<std::string>("Diffusivity Name"), dl->qp_tensor);
    this->addDependentField(diffusivity_);
  }

  if (have_convection_) {
    convection_vector_ = decltype(convection_vector_)(
        p.get<std::string>("Convection Vector Name"), dl->qp_vector);
    this->addDependentField(convection_vector_);
  }

  if (have_species_coupling_) {
    species_coupling_ = decltype(species_coupling_)(
        p.get<std::string>("Species Coupling Name"), dl->qp_scalar);
    this->addDependentField(species_coupling_);
  }

  if (have_stabilization_) {
    stabilization_ = decltype(stabilization_)(
        p.get<std::string>("Stabilization Name"), dl->qp_scalar);
    this->addDependentField(stabilization_);
  }

  if (have_contact_) {
    M_operator_ =
        decltype(M_operator_)(p.get<std::string>("M Name"), dl->qp_scalar);
    this->addDependentField(M_operator_);
  }

  if (have_transient_ && have_mechanics_ && (SolutionType_ != "Continuation")) {
    stress_ =
        decltype(stress_)(p.get<std::string>("Stress Name"), dl->qp_tensor);

    vel_grad_ = decltype(vel_grad_)(
        p.get<std::string>("Velocity Gradient Variable Name"), dl->qp_tensor);

    this->addDependentField(stress_);
    this->addDependentField(vel_grad_);
  }

  this->addEvaluatedField(residual_);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  num_cells_ = dims[0];
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];

  scalar_name_ = p.get<std::string>("Scalar Variable Name") + "_old";

  this->setName("TransportResidual" + PHX::typeAsString<EvalT>());
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
TransportResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(scalar_, fm);
  this->utils.setFieldData(scalar_grad_, fm);
  this->utils.setFieldData(weights_, fm);
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(w_grad_bf_, fm);

  if (have_source_) { this->utils.setFieldData(source_, fm); }

  if (have_second_source_) { this->utils.setFieldData(second_source_, fm); }

  if (have_transient_) {
    this->utils.setFieldData(transient_coeff_, fm);
    this->utils.setFieldData(scalar_dot_, fm);
    this->utils.setFieldData(delta_time_, fm);
  }

  if (have_diffusion_) { this->utils.setFieldData(diffusivity_, fm); }

  if (have_convection_) { this->utils.setFieldData(convection_vector_, fm); }

  if (have_species_coupling_) {
    this->utils.setFieldData(species_coupling_, fm);
  }

  if (have_stabilization_) { this->utils.setFieldData(stabilization_, fm); }

  if (have_contact_) { this->utils.setFieldData(M_operator_, fm); }

  if (have_transient_ && have_mechanics_ && (SolutionType_ != "Continuation")) {
    this->utils.setFieldData(stress_, fm);
    this->utils.setFieldData(vel_grad_, fm);
  }

  this->utils.setFieldData(residual_, fm);

  // initialize term1_
  term1_ = Kokkos::createDynRankView(
      scalar_.get_view(), "XXX", num_cells_, num_pts_);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
TransportResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // zero out residual
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < num_nodes_; ++node) {
      residual_(cell, node) = 0.0;
    }
  }

  // transient term
  if (have_transient_ && delta_time_(0) > 0.0) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        for (int node = 0; node < num_nodes_; ++node) {
          residual_(cell, node) += transient_coeff_(cell, pt) *
                                   w_bf_(cell, node, pt) *
                                   scalar_dot_(cell, pt);
        }
      }
    }
  }

  // term ==> P : F_dot
  if (have_transient_ && have_mechanics_ && (SolutionType_ != "Continuation")) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        ScalarT sum(0.0);
        // This is dumb, but I want to be sure I am initializing the arrays with
        // zeros.
        term1_(cell, pt) = ScalarT(0.0);
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            sum += stress_(cell, pt, i, j) * vel_grad_(cell, pt, i, j);
          }
        }
        term1_(cell, pt) = sum;
      }
    }

    // add to residual
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        for (int node = 0; node < num_nodes_; ++node) {
          residual_(cell, node) -= w_bf_(cell, node, pt) * term1_(cell, pt);
        }
      }
    }
  }

  // diffusive term
  if (have_diffusion_ && !(have_transient_ && delta_time_(0) == 0.0)) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        for (int node = 0; node < num_nodes_; ++node) {
          for (int i = 0; i < num_dims_; ++i) {
            for (int j = 0; j < num_dims_; ++j) {
              residual_(cell, node) += w_grad_bf_(cell, node, pt, i) *
                                       diffusivity_(cell, pt, i, j) *
                                       scalar_grad_(cell, pt, j);
            }
          }
        }
      }
    }
  }

  // source term
  if (have_source_) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        for (int node = 0; node < num_nodes_; ++node) {
          residual_(cell, node) -= w_bf_(cell, node, pt) * source_(cell, pt);
        }
      }
    }
  }

  // second source term
  if (have_second_source_) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < num_pts_; ++pt) {
        for (std::size_t node = 0; node < num_nodes_; ++node) {
          residual_(cell, node) -=
              w_bf_(cell, node, pt) * second_source_(cell, pt);
        }
      }
    }
  }

  // convection term
  if (have_convection_) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < num_pts_; ++pt) {
        for (int node = 0; node < num_nodes_; ++node) {
          for (int dim = 0; dim < num_dims_; ++dim) {
            residual_(cell, node) += w_bf_(cell, node, pt) *
                                     convection_vector_(cell, pt, dim) *
                                     scalar_grad_(cell, pt, dim);
          }
        }
      }
    }
  }
}
}  // namespace LCM
