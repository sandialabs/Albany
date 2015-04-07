//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

//#define PRINT_DEBUG

namespace LCM {

//----------------------------------------------------------------------------
template<typename EvalT, typename Traits>
SurfaceVectorResidual<EvalT, Traits>::
SurfaceVectorResidual(Teuchos::ParameterList & p,
    Teuchos::RCP<Albany::Layouts> const & dl) :
    thickness_
    (p.get<double>("thickness")),

    cubature_
    (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),

    intrepid_basis_
    (p.get<Teuchos::RCP<Intrepid::Basis<RealType,
        Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),

    stress_
    (p.get<std::string>("Stress Name"), dl->qp_tensor),

    current_basis_
    (p.get<std::string>("Current Basis Name"), dl->qp_tensor),

    ref_dual_basis_
    (p.get<std::string>("Reference Dual Basis Name"), dl->qp_tensor),

    ref_normal_
    (p.get<std::string>("Reference Normal Name"), dl->qp_vector),

    ref_area_
    (p.get<std::string>("Reference Area Name"), dl->qp_scalar),

    force_
    (p.get<std::string>("Surface Vector Residual Name"), dl->node_vector),

    use_cohesive_traction_
    (p.get<bool>("Use Cohesive Traction", false)),

    compute_membrane_forces_
    (p.get<bool>("Compute Membrane Forces", false)),

    have_topmod_adaptation_
    (p.get<bool>("Use Adaptive Insertion", false))
{
  this->addDependentField(current_basis_);
  this->addDependentField(ref_dual_basis_);
  this->addDependentField(ref_normal_);
  this->addDependentField(ref_area_);

  this->addEvaluatedField(force_);

  this->setName("Surface Vector Residual" + PHX::typeAsString<EvalT>());

  // if enabled grab the cohesive tractions
  if (use_cohesive_traction_) {

    PHX::MDField<ScalarT, Cell, QuadPoint, Dim>
    ct(p.get<std::string>("Cohesive Traction Name"), dl->qp_vector);

    traction_ = ct;
    this->addDependentField(traction_);
  } else {
    this->addDependentField(stress_);
  }

  if (have_topmod_adaptation_ == true) {
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim>
    J(p.get<std::string>("Jacobian Name"), dl->qp_scalar);

    detF_ = J;
    this->addDependentField(detF_);

    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim>
    sigma(p.get<std::string>("Cauchy Stress Name"), dl->qp_tensor);

    cauchy_stress_ = sigma;

    this->addEvaluatedField(cauchy_stress_);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numDims = dims[2];

  numQPs = cubature_->getNumPoints();

  numPlaneNodes = numNodes / 2;
  numPlaneDims = numDims - 1;

  // Allocate Temporary FieldContainers
  ref_values_.resize(numPlaneNodes, numQPs);
  ref_grads_.resize(numPlaneNodes, numQPs, numPlaneDims);
  ref_points_.resize(numQPs, numPlaneDims);
  ref_weights_.resize(numQPs);

  // Pre-Calculate reference element quantitites
  cubature_->getCubature(ref_points_, ref_weights_);
  intrepid_basis_->getValues(ref_values_, ref_points_, Intrepid::OPERATOR_VALUE);
  intrepid_basis_->getValues(ref_grads_, ref_points_, Intrepid::OPERATOR_GRAD);
}

//----------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void SurfaceVectorResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits> & fm)
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

  if (have_topmod_adaptation_)
    this->utils.setFieldData(cauchy_stress_, fm);
}

//----------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void SurfaceVectorResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // define and initialize tensors/vectors
  Intrepid::Vector<ScalarT> f_plus(0, 0, 0), f_minus(0, 0, 0);
  ScalarT dgapdxN, tmp1, tmp2, dndxbar, dFdx_plus, dFdx_minus;

  // manually fill the permutation tensor
  Intrepid::Tensor3<MeshScalarT> e(3, Intrepid::ZEROS);
  e(0, 1, 2) = e(1, 2, 0) = e(2, 0, 1) = 1.0;
  e(0, 2, 1) = e(1, 0, 2) = e(2, 1, 0) = -1.0;

  // 2nd-order identity tensor
  const Intrepid::Tensor<MeshScalarT> I = Intrepid::identity<MeshScalarT>(3);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {

      force_(cell, node, 0) = 0.0;
      force_(cell, node, 1) = 0.0;
      force_(cell, node, 2) = 0.0;
      int topNode = node + numPlaneNodes;
      force_(cell, topNode, 0) = 0.0;
      force_(cell, topNode, 1) = 0.0;
      force_(cell, topNode, 2) = 0.0;

      for (int pt(0); pt < numQPs; ++pt) {
        // deformed bases
        Intrepid::Vector<ScalarT> g_0(3, current_basis_, cell, pt, 0, 0);
        Intrepid::Vector<ScalarT> g_1(3, current_basis_, cell, pt, 1, 0);
        Intrepid::Vector<ScalarT> n(3, current_basis_, cell, pt, 2, 0);
        // ref bases
        Intrepid::Vector<MeshScalarT> G0(3, ref_dual_basis_, cell, pt, 0, 0);
        Intrepid::Vector<MeshScalarT> G1(3, ref_dual_basis_, cell, pt, 1, 0);
        Intrepid::Vector<MeshScalarT> G2(3, ref_dual_basis_, cell, pt, 2, 0);
        // ref normal
        Intrepid::Vector<MeshScalarT> N(3, ref_normal_, cell, pt, 0);

        // compute dFdx_plus_or_minus
        f_plus.clear();
        f_minus.clear();

        // h * P * dFperpdx --> +/- \lambda * P * N
        if (use_cohesive_traction_) {
          Intrepid::Vector<ScalarT> T(3, traction_, cell, pt, 0);
          f_plus = ref_values_(node, pt) * T;
          f_minus = -ref_values_(node, pt) * T;
        } else {
          Intrepid::Tensor<ScalarT> P(3, stress_, cell, pt, 0, 0);

          f_plus = ref_values_(node, pt) * P * N;
          f_minus = -ref_values_(node, pt) * P * N;

          if (compute_membrane_forces_) {
            for (int m(0); m < numDims; ++m) {
              for (int i(0); i < numDims; ++i) {
                for (int L(0); L < numDims; ++L) {

                  // tmp1 = (1/2) * delta * lambda_{,alpha} * G^{alpha L}
                  tmp1 = 0.5 * I(m, i) * (ref_grads_(node, pt, 0) * G0(L) +
                      ref_grads_(node, pt, 1) * G1(L));

                  // tmp2 = (1/2) * dndxbar * G^{3}
                  dndxbar = 0.0;
                  for (int r(0); r < numDims; ++r) {
                    for (int s(0); s < numDims; ++s) {
                      //dndxbar(m, i) += e(i, r, s)
                      dndxbar += e(i, r, s)
                          * (g_1(r) * ref_grads_(node, pt, 0) -
                              g_0(r) * ref_grads_(node, pt, 1))
                          * (I(m, s) - n(m) * n(s)) /
                          Intrepid::norm(Intrepid::cross(g_0, g_1));
                    }
                  }
                  tmp2 = 0.5 * dndxbar * G2(L);

                  // dFdx_plus
                  dFdx_plus = tmp1 + tmp2;

                  // dFdx_minus
                  dFdx_minus = tmp1 + tmp2;

                  //F = h * P:dFdx
                  f_plus(i) += thickness_ * P(m, L) * dFdx_plus;
                  f_minus(i) += thickness_ * P(m, L) * dFdx_minus;

                }
              }
            }
          }
        }

        // area (Reference) = |Jacobian| * weights
        force_(cell, topNode, 0) += f_plus(0) * ref_area_(cell, pt);
        force_(cell, topNode, 1) += f_plus(1) * ref_area_(cell, pt);
        force_(cell, topNode, 2) += f_plus(2) * ref_area_(cell, pt);

        force_(cell, node, 0) += f_minus(0) * ref_area_(cell, pt);
        force_(cell, node, 1) += f_minus(1) * ref_area_(cell, pt);
        force_(cell, node, 2) += f_minus(2) * ref_area_(cell, pt);

      } // end of pt

#if defined(PRINT_DEBUG)
      std::cout << "\nCELL: " << cell << " TOP NODE: " << topNode;
      std::cout << " BOTTOM NODE: " << node << '\n';
      std::cout << "force(0) +:" << force_(cell, topNode, 0) << '\n';
      std::cout << "force(1) +:" << force_(cell, topNode, 1) << '\n';
      std::cout << "force(2) +:" << force_(cell, topNode, 2) << '\n';
      std::cout << "force(0) -:" << force_(cell, node, 0) << '\n';
      std::cout << "force(1) -:" << force_(cell, node, 1) << '\n';
      std::cout << "force(2) -:" << force_(cell, node, 2) << '\n';
#endif //PRINT_DEBUG

    } // end of numPlaneNodes
  } // end of cell

  // This is here just to satisfy projection operators from QPs to nodes
  if (have_topmod_adaptation_ == true) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t pt = 0; pt < numQPs; ++pt) {
        for (int i = 0; i < numDims; ++i) {
          for (int j = 0; j < numDims; ++j) {
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
}
