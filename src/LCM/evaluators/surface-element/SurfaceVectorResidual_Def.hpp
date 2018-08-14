//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

//#define PRINT_DEBUG

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
SurfaceVectorResidual<EvalT, Traits>::SurfaceVectorResidual(
    Teuchos::ParameterList&              p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : thickness_(p.get<double>("thickness")),

      cubature_(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),

      intrepid_basis_(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis")),

      stress_(p.get<std::string>("Stress Name"), dl->qp_tensor),

      current_basis_(p.get<std::string>("Current Basis Name"), dl->qp_tensor),

      ref_dual_basis_(
          p.get<std::string>("Reference Dual Basis Name"),
          dl->qp_tensor),

      ref_normal_(p.get<std::string>("Reference Normal Name"), dl->qp_vector),

      ref_area_(p.get<std::string>("Reference Area Name"), dl->qp_scalar),

      force_(
          p.get<std::string>("Surface Vector Residual Name"),
          dl->node_vector),

      use_cohesive_traction_(p.get<bool>("Use Cohesive Traction", false)),

      compute_membrane_forces_(p.get<bool>("Compute Membrane Forces", false)),

      have_topmod_adaptation_(p.get<bool>("Use Adaptive Insertion", false))
{
  this->addDependentField(current_basis_);
  this->addDependentField(ref_dual_basis_);
  this->addDependentField(ref_normal_);
  this->addDependentField(ref_area_);

  this->addEvaluatedField(force_);

  this->setName("Surface Vector Residual" + PHX::typeAsString<EvalT>());

  // if enabled grab the cohesive tractions
  if (use_cohesive_traction_) {
    traction_ = decltype(traction_)(
        p.get<std::string>("Cohesive Traction Name"), dl->qp_vector);

    this->addDependentField(traction_);
  } else {
    this->addDependentField(stress_);
  }

  if (have_topmod_adaptation_ == true) {
    detF_ = decltype(detF_)(p.get<std::string>("Jacobian Name"), dl->qp_scalar);

    this->addDependentField(detF_);

    cauchy_stress_ = decltype(cauchy_stress_)(
        p.get<std::string>("Cauchy Stress Name"), dl->qp_tensor);

    this->addEvaluatedField(cauchy_stress_);
  }

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_vector->dimensions(dims);
  workset_size_ = dims[0];
  num_nodes_    = dims[1];
  num_dims_     = dims[2];

  num_qps_ = cubature_->getNumPoints();

  num_surf_nodes_ = num_nodes_ / 2;
  num_plane_dims_ = num_dims_ - 1;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
SurfaceVectorResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(current_basis_, fm);
  this->utils.setFieldData(ref_dual_basis_, fm);
  this->utils.setFieldData(ref_normal_, fm);
  this->utils.setFieldData(ref_area_, fm);
  this->utils.setFieldData(force_, fm);

  if (use_cohesive_traction_) {
    this->utils.setFieldData(traction_, fm);
  } else {
    this->utils.setFieldData(stress_, fm);
  }

  if (have_topmod_adaptation_) this->utils.setFieldData(cauchy_stress_, fm);

  // Allocate Temporary Views
  ref_values_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_surf_nodes_, num_qps_);
  ref_grads_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_surf_nodes_, num_qps_, num_plane_dims_);
  ref_points_ = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", num_qps_, num_plane_dims_);
  ref_weights_ = Kokkos::DynRankView<RealType, PHX::Device>("XXX", num_qps_);

  // Pre-Calculate reference element quantitites
  cubature_->getCubature(ref_points_, ref_weights_);
  intrepid_basis_->getValues(
      ref_values_, ref_points_, Intrepid2::OPERATOR_VALUE);

  intrepid_basis_->getValues(ref_grads_, ref_points_, Intrepid2::OPERATOR_GRAD);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
SurfaceVectorResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // define and initialize tensors/vectors
  minitensor::Vector<ScalarT> f_plus(0, 0, 0), f_minus(0, 0, 0);

  ScalarT dgapdxN, tmp1, tmp2, dndxbar, dFdx_plus, dFdx_minus;

  // 2nd-order identity tensor
  minitensor::Tensor<MeshScalarT> const I =
      minitensor::identity<MeshScalarT>(3);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int bottom_node(0); bottom_node < num_surf_nodes_; ++bottom_node) {
      force_(cell, bottom_node, 0) = 0.0;
      force_(cell, bottom_node, 1) = 0.0;
      force_(cell, bottom_node, 2) = 0.0;

      int top_node = bottom_node + num_surf_nodes_;

      force_(cell, top_node, 0) = 0.0;
      force_(cell, top_node, 1) = 0.0;
      force_(cell, top_node, 2) = 0.0;

      for (int pt(0); pt < num_qps_; ++pt) {
        // deformed bases
        minitensor::Vector<ScalarT> g_0(
            minitensor::Source::ARRAY, 3, current_basis_, cell, pt, 0, 0);
        minitensor::Vector<ScalarT> g_1(
            minitensor::Source::ARRAY, 3, current_basis_, cell, pt, 1, 0);
        minitensor::Vector<ScalarT> n(
            minitensor::Source::ARRAY, 3, current_basis_, cell, pt, 2, 0);
        // ref bases
        minitensor::Vector<MeshScalarT> G0(
            minitensor::Source::ARRAY, 3, ref_dual_basis_, cell, pt, 0, 0);
        minitensor::Vector<MeshScalarT> G1(
            minitensor::Source::ARRAY, 3, ref_dual_basis_, cell, pt, 1, 0);
        minitensor::Vector<MeshScalarT> G2(
            minitensor::Source::ARRAY, 3, ref_dual_basis_, cell, pt, 2, 0);
        // ref normal
        minitensor::Vector<MeshScalarT> N(
            minitensor::Source::ARRAY, 3, ref_normal_, cell, pt, 0);

        // compute dFdx_plus_or_minus
        f_plus.fill(minitensor::Filler::ZEROS);
        f_minus.fill(minitensor::Filler::ZEROS);

        // h * P * dFperpdx --> +/- \lambda * P * N
        if (use_cohesive_traction_) {
          minitensor::Vector<ScalarT> T(
              minitensor::Source::ARRAY, 3, traction_, cell, pt, 0);

          f_plus  = ref_values_(bottom_node, pt) * T;
          f_minus = -ref_values_(bottom_node, pt) * T;
        } else {
          minitensor::Tensor<ScalarT> P(
              minitensor::Source::ARRAY, 3, stress_, cell, pt, 0, 0);

          f_plus  = ref_values_(bottom_node, pt) * P * N;
          f_minus = -ref_values_(bottom_node, pt) * P * N;

          if (compute_membrane_forces_) {
            for (int m(0); m < num_dims_; ++m) {
              for (int i(0); i < num_dims_; ++i) {
                for (int L(0); L < num_dims_; ++L) {
                  // tmp1 = (1/2) * delta * lambda_{,alpha} * G^{alpha L}
                  tmp1 = 0.5 * I(m, i) *
                         (ref_grads_(bottom_node, pt, 0) * G0(L) +
                          ref_grads_(bottom_node, pt, 1) * G1(L));

                  // tmp2 = (1/2) * dndxbar * G^{3}
                  dndxbar = 0.0;
                  for (int r(0); r < num_dims_; ++r) {
                    for (int s(0); s < num_dims_; ++s) {
                      dndxbar += minitensor::levi_civita<MeshScalarT>(i, r, s) *
                                 (g_1(r) * ref_grads_(bottom_node, pt, 0) -
                                  g_0(r) * ref_grads_(bottom_node, pt, 1)) *
                                 (I(m, s) - n(m) * n(s)) /
                                 minitensor::norm(minitensor::cross(g_0, g_1));
                    }
                  }
                  tmp2 = 0.5 * dndxbar * G2(L);

                  // dFdx_plus
                  dFdx_plus = tmp1 + tmp2;

                  // dFdx_minus
                  dFdx_minus = tmp1 + tmp2;

                  // F = h * P:dFdx
                  f_plus(i) += thickness_ * P(m, L) * dFdx_plus;
                  f_minus(i) += thickness_ * P(m, L) * dFdx_minus;
                }
              }
            }
          }
        }

        // area (Reference) = |Jacobian| * weights
        force_(cell, top_node, 0) += f_plus(0) * ref_area_(cell, pt);
        force_(cell, top_node, 1) += f_plus(1) * ref_area_(cell, pt);
        force_(cell, top_node, 2) += f_plus(2) * ref_area_(cell, pt);

        force_(cell, bottom_node, 0) += f_minus(0) * ref_area_(cell, pt);
        force_(cell, bottom_node, 1) += f_minus(1) * ref_area_(cell, pt);
        force_(cell, bottom_node, 2) += f_minus(2) * ref_area_(cell, pt);

      }  // end of pt

#if defined(PRINT_DEBUG)
      std::cout << "\nCELL: " << cell << " TOP NODE: " << top_node;
      std::cout << " BOTTOM NODE: " << bottom_node << '\n';
      std::cout << "force(0) +:" << force_(cell, top_node, 0) << '\n';
      std::cout << "force(1) +:" << force_(cell, top_node, 1) << '\n';
      std::cout << "force(2) +:" << force_(cell, top_node, 2) << '\n';
      std::cout << "force(0) -:" << force_(cell, bottom_node, 0) << '\n';
      std::cout << "force(1) -:" << force_(cell, bottom_node, 1) << '\n';
      std::cout << "force(2) -:" << force_(cell, bottom_node, 2) << '\n';
#endif  // PRINT_DEBUG

    }  // end of numPlaneNodes
  }    // end of cell

  // This is here just to satisfy projection operators from QPs to nodes
  if (have_topmod_adaptation_ == true) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < num_qps_; ++pt) {
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            if (use_cohesive_traction_) {
              cauchy_stress_(cell, pt, i, j) =
                  traction_(cell, pt, i) * ref_normal_(cell, pt, j);
            } else {
              cauchy_stress_(cell, pt, i, j) = stress_(cell, pt, i, j);
            }
          }
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
}  // namespace LCM
