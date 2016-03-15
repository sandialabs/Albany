//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid2_MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>

namespace LCM
{

//
//
//
template<typename EvalT, typename Traits>
ConstitutiveModel<EvalT, Traits>::
ConstitutiveModel(
    Teuchos::ParameterList * p,
    Teuchos::RCP<Albany::Layouts> const & dl) :
    AbstractModel<EvalT, Traits>::AbstractModel(p, dl)
{
  if (p->isType<bool>("Have Temperature") == true) {

    have_temperature_ = p->get<bool>("Have Temperature");
    expansion_coeff_ = p->get<RealType>("Thermal Expansion Coefficient", 0.0);
    ref_temperature_ = p->get<RealType>("Reference Temperature", 0.0);
    heat_capacity_ = p->get<RealType>("Heat Capacity", 1.0);
    density_ = p->get<RealType>("Density", 1.0);
  }

  if (p->isType<bool>("Have Damage") == true) {
    have_damage_ = p->get<bool>("Have Damage");
  }

  if (p->isType<bool>("Have Total Concentration")) {
    have_total_concentration_ = p->get<bool>("Have Total Concentration");
  }

  if (p->isType<bool>("Have Total Bubble Density")) {
    have_total_bubble_density_ = p->get<bool>("Have Total Bubble Density");
  }

  if (p->isType<bool>("Have Bubble Volume Fraction")) {
    have_bubble_volume_fraction_ = p->get<bool>("Have Bubble Volume Fraction");
  }

  if (p->isType<bool>("Compute Tangent")) {
    compute_tangent_ = p->get<bool>("Compute Tangent");
  }

}

//
// Kokkos Kernel for computeVolumeAverage
//
template <typename ScalarT, class ArrayStress, class ArrayWeights, class ArrayJ>
class computeVolumeAverageKernel {

  ArrayStress
  stress;

  ArrayWeights const
  weights_;

  ArrayJ const
  j_;

  int
  num_pts_;

  int
  num_dims_;


public:

  using device_type = PHX::Device;

  computeVolumeAverageKernel(
      ArrayStress & stress_,
      ArrayWeights const & weights,
      ArrayJ const & j,
      int const num_pts,
      int const num_dims) :
        stress(stress_),
        weights_(weights),
        j_(j),
        num_pts_(num_pts),
        num_dims_(num_dims)
  {
    return;
  }

  KOKKOS_INLINE_FUNCTION
  void operator ()(const int cell) const
  {
#ifndef PHX_KOKKOS_DEVICE_TYPE_CUDA
    ScalarT volume, pbar, p;
    Intrepid2::Tensor<ScalarT> sig(num_dims_);
    Intrepid2::Tensor<ScalarT> I(Intrepid2::eye<ScalarT>(num_dims_));

    volume = pbar = 0.0;

    for (int pt(0); pt < num_pts_; ++pt) {

      for (int i = 0; i < num_dims_; ++i)
        for (int j = 0; j < num_dims_; ++j)
          sig(i, j) = stress(cell, pt, i, j);

      pbar += weights_(cell, pt) * (1. / num_dims_) * Intrepid2::trace(sig);
      volume += weights_(cell, pt) * j_(cell, pt);
    }

    pbar /= volume;

    for (int pt(0); pt < num_pts_; ++pt) {

      for (int i = 0; i < num_dims_; ++i)
        for (int j = 0; j < num_dims_; ++j)
          sig(i, j) = stress(cell, pt, i, j);

      p = (1. / num_dims_) * Intrepid2::trace(sig);
      sig += (pbar - p) * I;

      for (int i = 0; i < num_dims_; ++i) {
        stress(cell, pt, i, i) = sig(i, i);
      }
    }
#else
    ScalarT volume, pbar, p;
    ScalarT sig[3][3];
    ScalarT I[3][3];

    ScalarT trace_sig=0.0;

    if (num_dims_>3)
    Kokkos::abort( "Error: ConstitutiveModel::computeVolumeAverage: size of temorary array is smaller then it should be");

    for (int i=0; i<num_dims_; i++) {
      for (int j=0; j<num_dims_; j++) {
        I[i][j]=ScalarT(0.0);
        if (i==j)
        I[i][j]=ScalarT(1.0);
      }
    }

    volume =0.0;
    pbar = 0.0;

    for (int pt(0); pt < num_pts_; ++pt) {

      for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j)
      sig[i][j]=stress(cell,pt,i,j);

      trace_sig=0.0;

      for (int i = 0; i < num_dims_; ++i) {
        trace_sig += sig[i][i];

        pbar += weights_(cell,pt) * (1./num_dims_) * trace_sig;
        volume += weights_(cell,pt) * j_(cell,pt);
      }
    }

    pbar /= volume;

    for (int pt(0); pt < num_pts_; ++pt) {

      for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j)
      sig[i][j]=stress(cell,pt,i,j);

      trace_sig=0.0;

      for (int i = 0; i < num_dims_; ++i) {
        trace_sig += sig[i][i];

        p = (1./num_dims_) * trace_sig;

        for (int i = 0; i < num_dims_; ++i)
        for (int j = 0; j < num_dims_; ++j)
        sig[i][j]+=(pbar - p)*I[i][j];
        //sig += (pbar - p)*I;

        for (int i = 0; i < num_dims_; ++i) {
          stress(cell,pt,i,i) = sig[i][i];
        }
      }

    }
#endif
  }
};

//
//
//
template<typename EvalT, typename Traits>
void ConstitutiveModel<EvalT, Traits>::
computeVolumeAverage(
    Workset workset,
    FieldMap dep_fields,
    FieldMap eval_fields)
{
  int const &
  num_dims = this->num_dims_;

  int const &
  num_pts = this->num_pts_;

  std::string cauchy = (*this->field_name_map_)["Cauchy_Stress"];
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy];
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Intrepid2::Tensor<ScalarT> sig(num_dims);
  Intrepid2::Tensor<ScalarT> I(Intrepid2::eye<ScalarT>(num_dims));

  ScalarT volume, pbar, p;

  for (int cell(0); cell < workset.numCells; ++cell) {
    volume = pbar = 0.0;
    for (int pt(0); pt < num_pts; ++pt) {
      sig.fill(stress, cell, pt, 0, 0);
      pbar += weights_(cell, pt) * (1. / num_dims) * Intrepid2::trace(sig);
      volume += weights_(cell, pt) * j_(cell, pt);
    }

    pbar /= volume;

    for (int pt(0); pt < num_pts; ++pt) {
      sig.fill(stress, cell, pt, 0, 0);
      p = (1. / num_dims) * Intrepid2::trace(sig);
      sig += (pbar - p) * I;

      for (int i = 0; i < num_dims; ++i) {
        stress(cell, pt, i, i) = sig(i, i);
      }
    }
  }
#else
  Kokkos::parallel_for(workset.numCells, computeVolumeAverageKernel<ScalarT, PHX::MDField<ScalarT>, PHX::MDField<MeshScalarT, Cell, QuadPoint>, PHX::MDField<ScalarT, Cell, QuadPoint>>(stress, weights_, j_, num_pts_, num_dims_));
#endif
}

} // namespace LCM
