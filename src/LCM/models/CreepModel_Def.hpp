//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#define DEBUG_FREQ 100000000000
#include <MiniTensor.h>
#include "LocalNonlinearSolver.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

// static void aprintd(double x)
//{
//  if (-1e-5 < x && x < 0)
//    x = 0;
//  fprintf(stderr, "%.5f", x);
//}

// static void aprints(double x)
//{
//  aprintd(x);
//  fprintf(stderr,"\n");
//}

// static void aprints(FadType const& x)
//{
//  aprintd(x.val());
//  fprintf(stderr," [");
//  for (int i = 0; i < x.size(); ++i) {
//    fprintf(stderr," ");
//    aprintd(x.dx(i));
//  }
//  fprintf(stderr,"]\n");
//}

// static void stripDeriv(double& x)
//{
//  (void)x;
//}

// static void stripDeriv(FadType& x)
//{
//  x.resize(0);
//}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
CreepModel<EvalT, Traits>::CreepModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      creep_initial_guess_(p->get<RealType>("Initial Creep Guess", 1.1e-4)),

      // sat_mod_(p->get<RealType>("Saturation Modulus", 0.0)),
      // sat_exp_(p->get<RealType>("Saturation Exponent", 0.0)),

      // below is what we called C_2 in the functions
      strain_rate_expo_(p->get<RealType>("Strain Rate Exponent", 1.0)),
      // below is what we called A in the functions
      relaxation_para_(
          p->get<RealType>("Relaxation Parameter of Material_A", 0.1)),
      // below is what we called Q/R in the functions, users can give them
      // values here
      activation_para_(
          p->get<RealType>("Activation Parameter of Material_Q/R", 500.0))

{
  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string     = (*field_name_map_)["Fp"];
  std::string eqps_string   = (*field_name_map_)["eqps"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string      = (*field_name_map_)["F"];
  std::string J_string      = (*field_name_map_)["J"];

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
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cauchy Stress", false));
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
    this->state_var_output_flags_.push_back(
        p->get<bool>("Output Mechanical Source", false));
  }
}

// void creepprint(double x)
//{
//  fprintf(stderr, "%a\n", x);
//}

// void creepprint(FadType const& x)
//{
//  fprintf(stderr, "%a [", x.val());
//  for (int i = 0; i < x.size(); ++i)
//    fprintf(stderr, " %a", x.dx(i));
//  fprintf(stderr, "\n");
//}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
CreepModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  static int  times_called  = 0;
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string     = (*field_name_map_)["Fp"];
  std::string eqps_string   = (*field_name_map_)["eqps"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string      = (*field_name_map_)["F"];
  std::string J_string      = (*field_name_map_)["J"];

  // extract dependent MDFields
  auto def_grad          = *dep_fields[F_string];
  auto J                 = *dep_fields[J_string];
  auto poissons_ratio    = *dep_fields["Poissons Ratio"];
  auto elastic_modulus   = *dep_fields["Elastic Modulus"];
  auto yield_strength    = *dep_fields["Yield Strength"];
  auto hardening_modulus = *dep_fields["Hardening Modulus"];
  auto delta_time        = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  auto                  stress = *eval_fields[cauchy_string];
  auto                  Fp     = *eval_fields[Fp_string];
  auto                  eqps   = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> source;
  if (have_temperature_) { source = *eval_fields[source_string]; }

  // get State Variables
  Albany::MDArray Fpold   = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqps_string + "_old"];

  ScalarT kappa, mu, mubar, K, Y;
  // new parameters introduced here for being the temperature dependent, they
  // are the last two listed below
  ScalarT Jm23, p, dgam, dgam_plastic, a0, a1, f, smag,
      temp_adj_relaxation_para_;
  ScalarT sq23(std::sqrt(2. / 3.));

  minitensor::Tensor<ScalarT> F(num_dims_), be(num_dims_), s(num_dims_),
      sigma(num_dims_);
  minitensor::Tensor<ScalarT> N(num_dims_), A(num_dims_), expA(num_dims_),
      Fpnew(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));
  minitensor::Tensor<ScalarT> Fpn(num_dims_), Fpinv(num_dims_),
      Cpinv(num_dims_);

  long int debug_output_counter = 0;

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      debug_output_counter++;
      kappa = elastic_modulus(cell, pt) /
              (3. * (1. - 2. * poissons_ratio(cell, pt)));
      mu   = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      K    = hardening_modulus(cell, pt);
      Y    = yield_strength(cell, pt);
      Jm23 = std::pow(J(cell, pt), -2. / 3.);

      // ----------------------------  temperature dependent coefficient
      // ------------------------

      // the effective 'B' we had before in the previous models, with mu
      if (have_temperature_) {
        temp_adj_relaxation_para_ =
            relaxation_para_ *
            std::exp(-activation_para_ / temperature_(cell, pt));
      } else {
        temp_adj_relaxation_para_ =
            relaxation_para_ * std::exp(-activation_para_ / 303.0);
      }

      if (debug_output_counter % DEBUG_FREQ == 0)
        std::cout << "B = " << temp_adj_relaxation_para_ << std::endl;

      // fill local tensors
      F.fill(def_grad, cell, pt, 0, 0);

      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
        }
      }

      // compute trial state
      Fpinv = minitensor::inverse(Fpn);
      Cpinv = Fpinv * minitensor::transpose(Fpinv);
      be    = Jm23 * F * Cpinv * minitensor::transpose(F);

      a0 = minitensor::norm(minitensor::dev(be));
      a1 = minitensor::trace(be);

      s = mu * minitensor::dev(be);

      mubar = minitensor::trace(be) * mu / (num_dims_);

      smag = minitensor::norm(s);

      f = smag - sq23 * (Y + K * eqpsold(cell, pt));

      // check yield condition
      if (f <= 0.0) {
        if (a0 > 1.0E-12) {
          // return mapping algorithm
          bool      converged = false;
          ScalarT   alpha     = 0.0;
          ScalarT   res       = 0.0;
          int       count     = 0;
          int const max_count = 100;
          // ScalarT H = 0.0;
          dgam = 0.0;
          ScalarT debug_X[max_count + 1];
          ScalarT debug_F[max_count + 1];
          ScalarT debug_dFdX[max_count + 1];
          ScalarT debug_res[max_count + 1];

          LocalNonlinearSolver<EvalT, Traits> solver;

          std::vector<ScalarT> F(1);
          std::vector<ScalarT> dFdX(1);
          std::vector<ScalarT> X(1);

          X[0] = creep_initial_guess_;

          F[0] = X[0] - delta_time(0) * temp_adj_relaxation_para_ *
                            std::pow(mu, strain_rate_expo_) *
                            std::pow(
                                (a0 - 2. / 3. * X[0] * a1) *
                                    (a0 - 2. / 3. * X[0] * a1),
                                strain_rate_expo_ / 2.);

          dFdX[0] =
              1. -
              delta_time(0) * temp_adj_relaxation_para_ *
                  std::pow(mu, strain_rate_expo_) * (strain_rate_expo_ / 2.) *
                  std::pow(
                      (a0 - 2. / 3. * X[0] * a1) * (a0 - 2. / 3. * X[0] * a1),
                      strain_rate_expo_ / 2. - 1.) *
                  (8. / 9. * X[0] * a1 * a1 - 4. / 3. * a0 * a1);

          if ((typeid(ScalarT) == typeid(double)) && (F[0] != F[0])) {
            std::cerr << "F[0] is NaN, here are some contributing values:n";
            std::cerr << "Fpinv is " << Fpinv << 'n';
            std::cerr << "Cpinv is " << Fpinv << 'n';
            std::cerr << "a0 is " << a0 << 'n';
            std::cerr << "a1 is " << a1 << 'n';
            std::cerr << "mu is " << mu << 'n';
            std::cerr << "strain_rate_expo_ is " << strain_rate_expo_ << 'n';
            std::cerr << "temp_adj_relaxation_para_ is "
                      << temp_adj_relaxation_para_ << 'n';
            std::cerr << "dt is " << delta_time(0) << 'n';
          }

          debug_X[0]    = X[0];
          debug_F[0]    = F[0];
          debug_dFdX[0] = dFdX[0];
          debug_res[0]  = 0.0;

          while (!converged && count <= max_count) {
            count++;
            solver.solve(dFdX, X, F);

            F[0] = X[0] - delta_time(0) * temp_adj_relaxation_para_ *
                              std::pow(mu, strain_rate_expo_) *
                              std::pow(
                                  (a0 - 2. / 3. * X[0] * a1) *
                                      (a0 - 2. / 3. * X[0] * a1),
                                  strain_rate_expo_ / 2.);

            dFdX[0] =
                1. -
                delta_time(0) * temp_adj_relaxation_para_ *
                    std::pow(mu, strain_rate_expo_) * (strain_rate_expo_ / 2.) *
                    std::pow(
                        (a0 - 2. / 3. * X[0] * a1) * (a0 - 2. / 3. * X[0] * a1),
                        strain_rate_expo_ / 2. - 1.) *
                    (8. / 9. * X[0] * a1 * a1 - 4. / 3. * a0 * a1);

            if (debug_output_counter % DEBUG_FREQ == 0)
              std::cout << "Creep Solver count = " << count << std::endl;
            if (debug_output_counter % DEBUG_FREQ == 0)
              std::cout << "X[0] = " << X[0] << std::endl;
            if (debug_output_counter % DEBUG_FREQ == 0)
              std::cout << "F[0] = " << F[0] << std::endl;
            if (debug_output_counter % DEBUG_FREQ == 0)
              std::cout << "dFdX[0] = " << dFdX[0] << std::endl;

            debug_X[count]    = X[0];
            debug_F[count]    = F[0];
            debug_dFdX[count] = dFdX[0];

            res              = std::abs(F[0]);
            debug_res[count] = res;
            if (res < 1.e-10) { converged = true; }

            if (count == max_count) {
              std::cerr << "detected NaN, here are the X, F, dfdX values at "
                           "each iteration:\n";
              for (int i = 0; i < max_count; ++i) {
                std::cout << "i = " << i << std::endl;
                std::cout << "debug_X =" << debug_X[i] << std::endl;
                std::cout << "debug_F =" << debug_F[i] << std::endl;
                std::cout << "debug_dFdX =" << debug_dFdX[i] << std::endl;
                std::cout << "debug_res =" << debug_res[i] << std::endl;
              }
            }

            TEUCHOS_TEST_FOR_EXCEPTION(
                count == max_count,
                std::runtime_error,
                std::endl
                    << "Error in return mapping, count = " << count
                    << "\nres = " << res << "\ng = " << F[0] << "\ndg = "
                    << dFdX[0] << "\nalpha = " << alpha << std::endl);
          }
          solver.computeFadInfo(dFdX, X, F);

          dgam = X[0];

          // plastic direction
          N = s / minitensor::norm(s);

          // update s
          s -= 2.0 * mubar * dgam * N;

          // mechanical source
          /* The below source heat calculation is not correct.
           *  It is not correct because the yield strength (Y)
           *  is being added to the temperature (temperature_)
           *  which is dimensionally wrong.
           *
           *  if (have_temperature_ && delta_time(0) > 0)
           *  {
           *  source(cell, pt) = 0.0 * (sq23 * dgam / delta_time(0)
           *    * (Y + temperature_(cell,pt))) / (density_ * heat_capacity_);
           *  }
           */

          // exponential map to get Fpnew
          A              = dgam * N;
          eqps(cell, pt) = eqpsold(cell, pt);
          expA           = minitensor::exp(A);
          Fpnew          = expA * Fpn;
          for (int i(0); i < num_dims_; ++i) {
            for (int j(0); j < num_dims_; ++j) {
              Fp(cell, pt, i, j) = Fpnew(i, j);
            }
          }
        } else {
          eqps(cell, pt) = eqpsold(cell, pt);
          for (int i(0); i < num_dims_; ++i) {
            for (int j(0); j < num_dims_; ++j) {
              Fp(cell, pt, i, j) = Fpn(i, j);
            }
          }
        }
      } else {
        bool    converged = false;
        ScalarT H         = 0.0;
        ScalarT dH        = 0.0;
        ScalarT alpha     = 0.0;
        ScalarT res       = 0.0;
        int     count     = 0;
        dgam              = 0.0;
        // smag_new = 0.0;
        dgam_plastic = 0.0;

        LocalNonlinearSolver<EvalT, Traits> solver;

        std::vector<ScalarT> F(1);
        std::vector<ScalarT> dFdX(1);
        std::vector<ScalarT> X(1);

        F[0]    = f;
        X[0]    = 0.0;
        dFdX[0] = (-2. * mubar) * (1. + H / (3. * mubar));

        while (!converged) {
          count++;
          solver.solve(dFdX, X, F);
          H = 2. * mubar * delta_time(0) * temp_adj_relaxation_para_ *
              std::pow(
                  (smag + 2. / 3. * (K * X[0]) - f) *
                      (smag + 2. / 3. * (K * X[0]) - f),
                  strain_rate_expo_ / 2.);
          dH = strain_rate_expo_ * 2. * mubar * delta_time(0) *
               temp_adj_relaxation_para_ * (2. * K) / 3. *
               std::pow(
                   (smag + 2. / 3. * (K * X[0]) - f) *
                       (smag + 2. / 3. * (K * X[0]) - f),
                   (strain_rate_expo_ - 1.) / 2.);
          F[0]    = f - 2. * mubar * (1. + K / (3. * mubar)) * X[0] - H;
          dFdX[0] = -2. * mubar * (1. + K / (3. * mubar)) - dH;

          res = std::abs(F[0]);
          if (res < 1.e-10 || res / f < 1.E-11) converged = true;

          TEUCHOS_TEST_FOR_EXCEPTION(
              count > 30,
              std::runtime_error,
              std::endl
                  << "Error in return mapping, count = " << count
                  << "\nres = " << res << "\nrelres = " << res / f
                  << "\ng = " << F[0] << "\ndg = " << dFdX[0] << std::endl);
        }
        solver.computeFadInfo(dFdX, X, F);

        dgam_plastic = X[0];

        // plastic direction
        N = s / minitensor::norm(s);

        // update s

        s -= 2.0 * mubar * dgam_plastic * N + f * N -
             2. * mubar * (1. + K / (3. * mubar)) * dgam_plastic * N;

        dgam =
            dgam_plastic + delta_time(0) * temp_adj_relaxation_para_ *
                               std::pow(minitensor::norm(s), strain_rate_expo_);

        alpha = eqpsold(cell, pt) + sq23 * dgam_plastic;

        // plastic direction
        N = s / minitensor::norm(s);

        // update eqps
        eqps(cell, pt) = alpha;

        // mechanical source
        if (have_temperature_ && delta_time(0) > 0) {
          source(cell, pt) =
              0.0 *
              (sq23 * dgam / delta_time(0) * (Y + H + temperature_(cell, pt))) /
              (density_ * heat_capacity_);
        }

        // exponential map to get Fpnew
        A     = dgam * N;
        expA  = minitensor::exp(A);
        Fpnew = expA * Fpn;
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            Fp(cell, pt, i, j) = Fpnew(i, j);
          }
        }
      }

      p = 0.5 * kappa * (J(cell, pt) - 1. / (J(cell, pt)));

      // compute stress
      sigma = p * I + s / J(cell, pt);
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }
    }
  }

  if (have_temperature_) {
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        ScalarT three_kappa =
            elastic_modulus(cell, pt) / (1.0 - 2.0 * poissons_ratio(cell, pt));
        F.fill(def_grad, cell, pt, 0, 0);
        ScalarT J = minitensor::det(F);
        sigma.fill(stress, cell, pt, 0, 0);
        sigma -= three_kappa * expansion_coeff_ * (1.0 + 1.0 / (J * J)) *
                 (temperature_(cell, pt) - ref_temperature_) * I;
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            stress(cell, pt, i, j) = sigma(i, j);
          }
        }
      }
    }
  }
}

}  // namespace LCM
