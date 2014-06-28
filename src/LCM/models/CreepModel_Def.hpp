//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
CreepModel<EvalT, Traits>::
CreepModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    creep_initial_guess(p->get<RealType>("Initial Creep Guess", 1.1e-4))
{
  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Yield Strength", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Hardening Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  if (have_temperature_) {
    this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));
  }

  // define the state variables
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Cauchy Stress", false));
  //
  // Fp
  this->num_state_variables_++;
  this->state_var_names_.push_back(Fp_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Fp", false));
  //
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output eqps", false));
  //
  // mechanical source
  if (have_temperature_) {
    this->num_state_variables_++;
    this->state_var_names_.push_back(source_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(p->get<bool>("Output Mechanical Source", false));
  }
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void CreepModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string     = (*field_name_map_)["Fp"];
  std::string eqps_string   = (*field_name_map_)["eqps"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string      = (*field_name_map_)["F"];
  std::string J_string      = (*field_name_map_)["J"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad         = *dep_fields[F_string];
  PHX::MDField<ScalarT> J                = *dep_fields[J_string];
  PHX::MDField<ScalarT> poissons_ratio   = *dep_fields["Poissons Ratio"];
  PHX::MDField<ScalarT> elastic_modulus  = *dep_fields["Elastic Modulus"];
  PHX::MDField<ScalarT> yieldStrength    = *dep_fields["Yield Strength"];
  PHX::MDField<ScalarT> hardeningModulus = *dep_fields["Hardening Modulus"];
  PHX::MDField<ScalarT> delta_time       = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> Fp     = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> eqps   = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> source;
  if (have_temperature_) {
    source = *eval_fields[source_string];
  }

  // get State Variables
  Albany::MDArray Fpold = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqps_string + "_old"];

  ScalarT kappa, mu, mubar, K, Y;
  ScalarT Jm23, trace, p, dgam, a0, a1;
  ScalarT sq23(std::sqrt(2. / 3.));

  Intrepid::Tensor<ScalarT> F(num_dims_), be(num_dims_), s(num_dims_), sigma(
      num_dims_);
  Intrepid::Tensor<ScalarT> N(num_dims_), A(num_dims_), expA(num_dims_), Fpnew(
      num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
  Intrepid::Tensor<ScalarT> Fpn(num_dims_), Fpinv(num_dims_), Cpinv(num_dims_);

  
  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {
      kappa = elastic_modulus(cell, pt)
          / (3. * (1. - 2. * poissons_ratio(cell, pt)));
      mu = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      K = hardeningModulus(cell, pt);
      Y = yieldStrength(cell, pt);
      Jm23 = std::pow(J(cell, pt), -2. / 3.);

      // fill local tensors
      F.fill(&def_grad(cell, pt, 0, 0));
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
        }
      }

      // compute trial state
      Fpinv = Intrepid::inverse(Fpn);
      Cpinv = Fpinv * Intrepid::transpose(Fpinv);
      be = Jm23 * F * Cpinv * Intrepid::transpose(F);

      a0 = Intrepid::norm(Intrepid::dev(be));
      a1 = Intrepid::trace(be); 

      s = mu * Intrepid::dev(be);
      mubar = Intrepid::trace(be) * mu / (num_dims_);

      // check yield condition
      if (a0 > 1E-12) {
        // return mapping algorithm
        bool converged = false;
        ScalarT alpha = 0.0;
        ScalarT res = 0.0;
        int count = 0;
        dgam = 0.0;

        LocalNonlinearSolver<EvalT, Traits> solver;

        std::vector<ScalarT> F(1);
        std::vector<ScalarT> dFdX(1);
        std::vector<ScalarT> X(1);

        X[0] = creep_initial_guess;
	F[0] =
	  std::pow(X[0], 2./K) 
	  - std::pow(Y, 2./K) * std::pow(mu, 2.) * std::pow(delta_time(0), 2./K)
	  * ( std::pow(a0, 2.) + 4./9. * std::pow(X[0], 2.) * std::pow(a1, 2.)
	      - 4./3. * X[0] * a0 * a1 
	      )	;

	dFdX[0] =
	  2./K * std::pow(X[0], (2./K - 1.)) 
	  - std::pow(Y, 2./K) * std::pow(mu, 2.) * std::pow(delta_time(0), 2./K)
	  * (8./9. * X[0] * std::pow(a1, 2.) - 4./3. * a0 * a1);

        while (!converged && count <= 30)
        {
          count++;
          solver.solve(dFdX, X, F);
          alpha = eqpsold(cell, pt) + sq23 * X[0];

	  F[0] =
	    std::pow(X[0], 2./K) 
	    - std::pow(Y, 2./K) * std::pow(mu, 2.) 
	    * std::pow(delta_time(0), 2./K)
	    * ( std::pow(a0, 2.) + 4./9. * std::pow(X[0], 2.) * std::pow(a1, 2.)
		- 4./3. * X[0] * a0 * a1 
		)	;

	  dFdX[0] =
	    2./K * std::pow(X[0], (2./K - 1.)) 
	    - std::pow(Y, 2./K) * std::pow(mu, 2.) 
	    * std::pow(delta_time(0), 2./K)
	    * (8./9. * X[0] * std::pow(a1, 2.) - 4./3. * a0 * a1);
	  
          res = std::abs(F[0]);
          if (res < 1.e-11 )
            converged = true;

          TEUCHOS_TEST_FOR_EXCEPTION(count == 30, std::runtime_error,
              std::endl <<
              "Error in return mapping, count = " <<
              count <<
              "\nres = " << res <<
              "\ng = " << F[0] <<
              "\ndg = " << dFdX[0] <<
              "\nalpha = " << alpha << std::endl);
        }
        solver.computeFadInfo(dFdX, X, F);
        dgam = X[0];

        // plastic direction
	N =  s / Intrepid::norm(s);

        // update s
        s -= 2 * mubar * dgam * N;

        // update eqps
        eqps(cell, pt) = alpha;

        // mechanical source
        if (have_temperature_ && delta_time(0) > 0) {
          source(cell, pt) = (sq23 * dgam / delta_time(0)
            * (Y + temperature_(cell,pt))) / (density_ * heat_capacity_);
        }

        // exponential map to get Fpnew
        A = dgam * N;
        expA = Intrepid::exp(A);
        Fpnew = expA * Fpn;
        for (std::size_t i(0); i < num_dims_; ++i) {
          for (std::size_t j(0); j < num_dims_; ++j) {
            Fp(cell, pt, i, j) = Fpnew(i, j);
          }
        }
      } else {
        std::cout << "hit alternate condition" << std::endl;
        eqps(cell, pt) = eqpsold(cell, pt);
        if (have_temperature_) source(cell, pt) = 0.0;
        for (std::size_t i(0); i < num_dims_; ++i) {
          for (std::size_t j(0); j < num_dims_; ++j) {
            Fp(cell, pt, i, j) = Fpn(i, j);
          }
        }
      }

      // compute pressure
      p = 0.5 * kappa * (J(cell, pt) - 1. / (J(cell, pt)));

      // compute stress
      sigma = p * I + s / J(cell, pt);
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }
    }
  }

  if (have_temperature_) {
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        F.fill(&def_grad(cell,pt,0,0));
        ScalarT J = Intrepid::det(F);
        sigma.fill(&stress(cell,pt,0,0));
        sigma -= 3.0 * expansion_coeff_ * (1.0 + 1.0 / (J*J))
          * (temperature_(cell,pt) - ref_temperature_) * I;
        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            stress(cell, pt, i, j) = sigma(i, j);
          }
        }
      }
    }
  }

}
//------------------------------------------------------------------------------
}

