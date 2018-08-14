//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ParallelNeohookeanModel_Def_hpp)
#define LCM_ParallelNeohookeanModel_Def_hpp

#include <MiniTensor.h>
#include <iostream>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"
#include "utility/math/Tensor.hpp"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
NeohookeanKernel<EvalT, Traits>::NeohookeanKernel(
    ConstitutiveModel<EvalT, Traits>&    model,
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : BaseKernel(model)
{
  std::string F_string = field_name_map_["F"];
  std::string J_string = field_name_map_["J"];
  std::string cauchy   = field_name_map_["Cauchy_Stress"];

  // define the dependent fields
  setDependentField(F_string, dl->qp_tensor);
  setDependentField(J_string, dl->qp_scalar);
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);

  // define the evaluated fields
  setEvaluatedField(cauchy, dl->qp_tensor);
  setEvaluatedField("Energy", dl->qp_scalar);
  setEvaluatedField("Material Tangent", dl->qp_tensor4);

  // define the state variables
  addStateVariable(
      cauchy,
      dl->qp_tensor,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output Cauchy Stress", false));
}

template <typename EvalT, typename Traits>
void
NeohookeanKernel<EvalT, Traits>::init(
    Workset&                 workset,
    FieldMap<const ScalarT>& dep_fields,
    FieldMap<ScalarT>&       eval_fields)
{
  std::string F_string = field_name_map_["F"];
  std::string J_string = field_name_map_["J"];
  std::string cauchy   = field_name_map_["Cauchy_Stress"];

  // extract dependent MDFields
  def_grad        = *dep_fields[F_string];
  J               = *dep_fields[J_string];
  poissons_ratio  = *dep_fields["Poissons Ratio"];
  elastic_modulus = *dep_fields["Elastic Modulus"];
  // extract evaluated MDFields
  stress  = *eval_fields[cauchy];
  energy  = *eval_fields["Energy"];
  tangent = *eval_fields["Material Tangent"];

  // std::cout << "field dimension: " << kern.def_grad.dimension(2) <<
  // std::endl; std::cout << "num_dims: " << num_dims_ << std::endl; std::cout <<
  // "num_pts: " << num_pts_ << std::endl;
}

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
NeohookeanKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
#if !defined(ALBANY_NIGHTLY_TEST)
  ScalarT kappa;
  ScalarT mu, mubar;
  ScalarT Jm53, Jm23;
  ScalarT smag;

  util::Tensor2<ScalarT> F(num_dims_), b(num_dims_), sigma(num_dims_);
  util::Tensor2<ScalarT> I(util::identity<ScalarT>(num_dims_));
  util::Tensor2<ScalarT> s(num_dims_), n(num_dims_);

  util::Tensor4<ScalarT> dsigmadb;
  util::Tensor4<ScalarT> I1(util::identity_1<ScalarT>(num_dims_));
  util::Tensor4<ScalarT> I3(util::identity_3<ScalarT>(num_dims_));
  kappa =
      elastic_modulus(cell, pt) / (3. * (1. - 2. * poissons_ratio(cell, pt)));
  mu   = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
  Jm53 = std::pow(J(cell, pt), -5. / 3.);
  Jm23 = Jm53 * J(cell, pt);

  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < num_dims_; ++j) { F(i, j) = def_grad(cell, pt, i, j); }
  }
  b     = F * util::transpose(F);
  mubar = (1.0 / 3.0) * mu * Jm23 * util::trace(b);

  sigma = 0.5 * kappa * (J(cell, pt) - 1. / J(cell, pt)) * I +
          mu * Jm53 * util::dev(b);

  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < num_dims_; ++j) {
      stress(cell, pt, i, j) = sigma(i, j);
    }
  }

  if (compute_energy_) {  // compute energy
    energy(cell, pt) =
        0.5 * kappa *
            (0.5 * (J(cell, pt) * J(cell, pt) - 1.0) - std::log(J(cell, pt))) +
        0.5 * mu * (Jm23 * util::trace(b) - 3.0);
  }

  if (compute_tangent_) {  // compute tangent

    s    = util::dev(sigma);
    smag = util::norm(s);
    n    = s / smag;

    dsigmadb = kappa * J(cell, pt) * J(cell, pt) * I3 -
               kappa * (J(cell, pt) * J(cell, pt) - 1.0) * I1 +
               2.0 * mubar * (I1 - (1.0 / 3.0) * I3) -
               2.0 / 3.0 * smag * (util::tensor(n, I) + util::tensor(I, n));

    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < num_dims_; ++j) {
        for (int k = 0; k < num_dims_; ++k) {
          for (int l = 0; l < num_dims_; ++l) {
            tangent(cell, pt, i, j, k, l) = dsigmadb(i, j, k, l);
          }
        }
      }
    }
  }
#endif
}

}  // namespace LCM

#endif
