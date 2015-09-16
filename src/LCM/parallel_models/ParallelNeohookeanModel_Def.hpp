//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ParallelNeohookeanModel_Def_hpp)
#define LCM_ParallelNeohookeanModel_Def_hpp

#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "utility/math/Tensor.hpp"
#include <iostream>

namespace LCM
{

//----------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ParallelNeohookeanModel<EvalT, Traits>::
ParallelNeohookeanModel(Teuchos::ParameterList* p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  Parent(p, dl)
{
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair("Energy", dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair("Material Tangent", dl->qp_tensor4));

  // define the state variables
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Cauchy Stress", false));
}


template<typename EvalT, typename Traits>
typename ParallelNeohookeanModel<EvalT, Traits>::EvalKernel
ParallelNeohookeanModel<EvalT, Traits>::
createEvalKernel(FieldMap &dep_fields,
                 FieldMap &eval_fields,
                 int numCells)
{
  EvalKernel kern;
  
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string cauchy = (*field_name_map_)["Cauchy_Stress"];

  // extract dependent MDFields
  kern.def_grad = *dep_fields[F_string];
  kern.J = *dep_fields[J_string];
  kern.poissons_ratio = *dep_fields["Poissons Ratio"];
  kern.elastic_modulus = *dep_fields["Elastic Modulus"];
  // extract evaluated MDFields
  kern.stress = *eval_fields[cauchy];
  kern.energy = *eval_fields["Energy"];
  kern.tangent = *eval_fields["Material Tangent"];
  
  //std::cout << "field dimension: " << kern.def_grad.dimension(2) << std::endl;
  //std::cout << "num_dims: " << num_dims_ << std::endl;
  //std::cout << "num_pts: " << num_pts_ << std::endl;
  
  kern.num_dims = num_dims_;
  kern.num_pts = num_pts_;
  kern.compute_energy = compute_energy_;
  kern.compute_tangent = compute_tangent_;
  
  return kern;
}

namespace detail
{
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
NeohookeanKernel<EvalT, Traits>::
operator()(int cell) const
{
  ScalarT kappa;
  ScalarT mu, mubar;
  ScalarT Jm53, Jm23;
  ScalarT smag;

  util::Tensor2<ScalarT> F(num_dims), b(num_dims), sigma(num_dims);
  util::Tensor2<ScalarT> I(util::identity<ScalarT>(num_dims));
  util::Tensor2<ScalarT> s(num_dims), n(num_dims);

  util::Tensor4<ScalarT> dsigmadb;
  util::Tensor4<ScalarT> I1(util::identity_1<ScalarT>(num_dims));
  util::Tensor4<ScalarT> I3(util::identity_3<ScalarT>(num_dims));
  
  for (int pt = 0; pt < num_pts; ++pt) {
    kappa =
        elastic_modulus(cell, pt)
            / (3. * (1. - 2. * poissons_ratio(cell, pt)));
    mu =
        elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
    Jm53 = std::pow(J(cell, pt), -5. / 3.);
    Jm23 = Jm53 * J(cell, pt);
    
    for (int i = 0; i < num_dims; ++i) {
      for (int j = 0; j < num_dims; ++j) {
        F(i, j) = def_grad(cell, pt, i, j);
      }
    }
    b = F * util::transpose(F);
    mubar = (1.0 / 3.0) * mu * Jm23 * util::trace(b);

    sigma = 0.5 * kappa * (J(cell, pt) - 1. / J(cell, pt)) * I
        + mu * Jm53 * util::dev(b);
    
    for (int i = 0; i < num_dims; ++i) {
      for (int j = 0; j < num_dims; ++j) {
        stress(cell, pt, i, j) = sigma(i, j);
      }
    }
    
    if (compute_energy) { // compute energy
      energy(cell, pt) =
          0.5 * kappa
              * (0.5 * (J(cell, pt) * J(cell, pt) - 1.0)
                  - std::log(J(cell, pt)))
              + 0.5 * mu * (Jm23 * util::trace(b) - 3.0);
    }

    if (compute_tangent) { // compute tangent

      s = util::dev(sigma);
      smag = util::norm(s);
      n = s / smag;

      dsigmadb =
          kappa * J(cell, pt) * J(cell, pt) * I3
              - kappa * (J(cell, pt) * J(cell, pt) - 1.0) * I1
              + 2.0 * mubar * (I1 - (1.0 / 3.0) * I3)
              - 2.0 / 3.0 * smag
                  * (util::tensor(n, I) + util::tensor(I, n));

      for (int i = 0; i < num_dims; ++i) {
        for (int j = 0; j < num_dims; ++j) {
          for (int k = 0; k < num_dims; ++k) {
            for (int l = 0; l < num_dims; ++l) {
              tangent(cell, pt, i, j, k, l) = dsigmadb(i, j, k, l);
            }
          }
        }
      }
    }
  }
}
}

}

#endif

