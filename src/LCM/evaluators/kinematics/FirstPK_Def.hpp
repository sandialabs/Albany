//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <PHAL_Utilities.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>
#ifdef ALBANY_TIMER
#include <chrono>
#endif

namespace LCM {
template <typename EvalT, typename Traits>
FirstPK<EvalT, Traits>::FirstPK(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : stress_(p.get<std::string>("Stress Name"), dl->qp_tensor),
      def_grad_(p.get<std::string>("DefGrad Name"), dl->qp_tensor),
      first_pk_stress_(
          p.get<std::string>("First PK Stress Name"),
          dl->qp_tensor),
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
    pore_pressure_ = decltype(pore_pressure_)(
        p.get<std::string>("Pore Pressure Name"), dl->qp_scalar);
    // grab Biot's coefficient
    biot_coeff_ = decltype(biot_coeff_)(
        p.get<std::string>("Biot Coefficient Name"), dl->qp_scalar);
    this->addDependentField(pore_pressure_);
    this->addDependentField(biot_coeff_);
  }

  // deal with stabilized pressure
  if (have_stab_pressure_) {
    stab_pressure_ = decltype(stab_pressure_)(
        p.get<std::string>("Pressure Name"), dl->qp_scalar);
    this->addDependentField(stab_pressure_);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  stress_.fieldTag().dataLayout().dimensions(dims);
  num_pts_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
}

template <typename EvalT, typename Traits>
void
FirstPK<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(first_pk_stress_, fm);
  if (have_pore_pressure_) {
    this->utils.setFieldData(pore_pressure_, fm);
    this->utils.setFieldData(biot_coeff_, fm);
  }
  if (have_stab_pressure_) this->utils.setFieldData(stab_pressure_, fm);
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
FirstPK<EvalT, Traits>::operator()(const small_strain_Tag& tag, const int& cell)
    const
{
  for (int pt = 0; pt < num_pts_; ++pt)
    for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j)
        first_pk_stress_(cell, pt, i, j) = stress_(cell, pt, i, j);
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
FirstPK<EvalT, Traits>::operator()(
    const have_stab_pressure_Tag& tag,
    const int&                    cell) const
{
  for (int pt = 0; pt < num_pts_; ++pt) {
    ScalarT pressure = first_pk_stress_(cell, pt, 0, 0);
    for (int k = 1; k < num_dims_; ++k)
      pressure += first_pk_stress_(cell, pt, k, k);
    pressure /= num_dims_;
    for (int k = 0; k < num_dims_; ++k)
      first_pk_stress_(cell, pt, k, k) += stab_pressure_(cell, pt) - pressure;
  }
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
FirstPK<EvalT, Traits>::operator()(
    const have_pore_pressure_Tag& tag,
    const int&                    cell) const
{
  for (int pt = 0; pt < num_pts_; ++pt)
    for (int k = 0; k < num_dims_; ++k) {
      // Effective Stress theory
      first_pk_stress_(cell, pt, k, k) -=
          biot_coeff_(cell, pt) * pore_pressure_(cell, pt);
    }
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
FirstPK<EvalT, Traits>::operator()(
    const no_small_strain_Tag& tag,
    const int&                 cell) const
{
  ScalarT sig[3][3], F[3][3], P[3][3];
  for (int pt = 0; pt < num_pts_; ++pt) {
    for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j) {
        F[i][j]   = def_grad_(cell, pt, i, j);
        sig[i][j] = first_pk_stress_(cell, pt, i, j);
      }

    // Replacement for piola(P, F, sig) for GPU, I think.
    // todo Can we go back to the MiniTensor function?
    switch (num_dims_) {
      default:
        Kokkos::abort(
            "Error(LCM FirstPK): piola function is defined only for rank-2 or "
            "3.");
        break;

      case 3:
        P[0][0] = sig[0][0] * (-F[1][2] * F[2][1] + F[1][1] * F[2][2]) +
                  sig[0][1] * (F[0][2] * F[2][1] - F[0][1] * F[2][2]) +
                  sig[0][2] * (-F[0][2] * F[1][1] + F[0][1] * F[1][2]);
        P[0][1] = sig[0][0] * (F[1][2] * F[2][0] - F[1][0] * F[2][2]) +
                  sig[0][1] * (-F[0][2] * F[2][0] + F[0][0] * F[2][2]) +
                  sig[0][2] * (F[0][2] * F[1][0] - F[0][0] * F[1][2]);
        P[0][2] = sig[0][0] * (-F[1][1] * F[2][0] + F[1][0] * F[2][1]) +
                  sig[0][1] * (F[0][1] * F[2][0] - F[0][0] * F[2][1]) +
                  sig[0][2] * (-F[0][1] * F[1][0] + F[0][0] * F[1][1]);

        P[1][0] = sig[1][0] * (-F[1][2] * F[2][1] + F[1][1] * F[2][2]) +
                  sig[1][1] * (F[0][2] * F[2][1] - F[0][1] * F[2][2]) +
                  sig[1][2] * (-F[0][2] * F[1][1] + F[0][1] * F[1][2]);
        P[1][1] = sig[1][0] * (F[1][2] * F[2][0] - F[1][0] * F[2][2]) +
                  sig[1][1] * (-F[0][2] * F[2][0] + F[0][0] * F[2][2]) +
                  sig[1][2] * (F[0][2] * F[1][0] - F[0][0] * F[1][2]);
        P[1][2] = sig[1][0] * (-F[1][1] * F[2][0] + F[1][0] * F[2][1]) +
                  sig[1][1] * (F[0][1] * F[2][0] - F[0][0] * F[2][1]) +
                  sig[1][2] * (-F[0][1] * F[1][0] + F[0][0] * F[1][1]);

        P[2][0] = sig[2][0] * (-F[1][2] * F[2][1] + F[1][1] * F[2][2]) +
                  sig[2][1] * (F[0][2] * F[2][1] - F[0][1] * F[2][2]) +
                  sig[2][2] * (-F[0][2] * F[1][1] + F[0][1] * F[1][2]);
        P[2][1] = sig[2][0] * (F[1][2] * F[2][0] - F[1][0] * F[2][2]) +
                  sig[2][1] * (-F[0][2] * F[2][0] + F[0][0] * F[2][2]) +
                  sig[2][2] * (F[0][2] * F[1][0] - F[0][0] * F[1][2]);
        P[2][2] = sig[2][0] * (-F[1][1] * F[2][0] + F[1][0] * F[2][1]) +
                  sig[2][1] * (F[0][1] * F[2][0] - F[0][0] * F[2][1]) +
                  sig[2][2] * (-F[0][1] * F[1][0] + F[0][0] * F[1][1]);
        break;

      case 2:
        P[0][0] = sig[0][0] * F[1][1] - sig[0][1] * F[0][1];
        P[0][1] = -sig[0][0] * F[1][0] + sig[0][1] * F[0][0];

        P[1][0] = sig[1][0] * F[1][1] - sig[1][1] * F[0][1];
        P[1][1] = -sig[1][0] * F[1][0] + sig[1][1] * F[0][0];
        break;
    }

    for (int i = 0; i < num_dims_; ++i)
      for (int j = 0; j < num_dims_; ++j)
        first_pk_stress_(cell, pt, i, j) = P[i][j];
  }
}

template <typename EvalT, typename Traits>
void
FirstPK<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Copy stress_ to first_pk_stress_.
  Kokkos::parallel_for(small_strain_Policy(0, workset.numCells), *this);
  // Optionally modify the stress tensor by pressure terms.
  if (have_stab_pressure_)
    Kokkos::parallel_for(have_stab_pressure_Policy(0, workset.numCells), *this);
  if (have_pore_pressure_)
    Kokkos::parallel_for(have_pore_pressure_Policy(0, workset.numCells), *this);
  if (!small_strain_) {
    // For large deformation, map Cauchy stress to 1st PK stress. In the
    // small-strain case, this transformation is Identity.
    Kokkos::parallel_for(no_small_strain_Policy(0, workset.numCells), *this);
  }
#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto      elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "First_PK time = " << millisec << "  " << microseconds
            << std::endl;
#endif
}
}  // namespace LCM
