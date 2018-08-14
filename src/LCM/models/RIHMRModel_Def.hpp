//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
RIHMRModel<EvalT, Traits>::RIHMRModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      sat_mod_(p->get<RealType>("Saturation Modulus", 0.0)),
      sat_exp_(p->get<RealType>("Saturation Exponent", 0.0))
{
  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("F", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("J", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Yield Strength", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Hardening Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Recovery Modulus", dl->qp_scalar));

  // retrieve appropriate field name strings
  std::string cauchy_string       = (*field_name_map_)["Cauchy_Stress"];
  std::string logFp_string        = (*field_name_map_)["logFp"];
  std::string eqps_string         = (*field_name_map_)["eqps"];
  std::string isoHardening_string = (*field_name_map_)["isoHardening"];

  // define evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(logFp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair(isoHardening_string, dl->qp_scalar));

  // define the state variables
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(true);
  //
  // logFp
  this->num_state_variables_++;
  this->state_var_names_.push_back(logFp_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(false);
  //
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // isoHardening
  this->num_state_variables_++;
  this->state_var_names_.push_back(isoHardening_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
}
//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
RIHMRModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // extract dependent MDFields
  auto def_grad          = *dep_fields["F"];
  auto J                 = *dep_fields["J"];
  auto poissons_ratio    = *dep_fields["Poissons Ratio"];
  auto elastic_modulus   = *dep_fields["Elastic Modulus"];
  auto yield_strength    = *dep_fields["Yield Strength"];
  auto hardening_modulus = *dep_fields["Hardening Modulus"];
  auto recovery_modulus  = *dep_fields["Recovery Modulus"];

  // retrieve appropriate field name strings
  std::string cauchy_string       = (*field_name_map_)["Cauchy_Stress"];
  std::string logFp_string        = (*field_name_map_)["logFp"];
  std::string eqps_string         = (*field_name_map_)["eqps"];
  std::string isoHardening_string = (*field_name_map_)["isoHardening"];

  // extract evaluated MDFields
  auto stress       = *eval_fields[cauchy_string];
  auto logFp        = *eval_fields[logFp_string];
  auto eqps         = *eval_fields[eqps_string];
  auto isoHardening = *eval_fields[isoHardening_string];

  // get State Variables
  Albany::MDArray logFp_old = (*workset.stateArrayPtr)[logFp_string + "_old"];
  Albany::MDArray eqps_old  = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray isoHardening_old =
      (*workset.stateArrayPtr)[isoHardening_string + "_old"];

  // scratch space FCs
  minitensor::Tensor<ScalarT> be(num_dims_);
  minitensor::Tensor<ScalarT> s(num_dims_);
  minitensor::Tensor<ScalarT> n(num_dims_);
  minitensor::Tensor<ScalarT> A(num_dims_);
  minitensor::Tensor<ScalarT> expA(num_dims_);

  minitensor::Tensor<ScalarT> logFp_n(num_dims_);
  minitensor::Tensor<ScalarT> Fp(num_dims_);
  minitensor::Tensor<ScalarT> Fpold(num_dims_);
  minitensor::Tensor<ScalarT> Cpinv(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  ScalarT kappa, Rd;
  ScalarT mu, mubar;
  ScalarT K, Y;
  ScalarT Jm23;
  ScalarT trd3, smag;
  ScalarT Phi, p, dgam, isoH;
  ScalarT sq23 = std::sqrt(2.0 / 3.0);

  // local unknowns and residual vectors
  std::vector<ScalarT>                R(2);
  std::vector<ScalarT>                X(2);
  std::vector<ScalarT>                dRdX(4);
  ScalarT                             normR0(0.0), normR(0.0), conv(0.0);
  LocalNonlinearSolver<EvalT, Traits> solver;

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      // logFp_n.fill(&logFp_old(cell, pt, int(0), int(0)) );
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          logFp_n(i, j) = static_cast<ScalarT>(logFp_old(cell, pt, i, j));
          // std::cout << "logFp_n(" << cell << ", " << pt << ", " << i << ", "
          // << j << " ) = " << logFp_n(i, j) << std::endl;
        }
      }

      Fp    = minitensor::exp(logFp_n);
      Fpold = Fp;
      Cpinv = minitensor::dot(
          minitensor::inverse(Fp),
          minitensor::transpose(minitensor::inverse(Fp)));

      kappa = elastic_modulus(cell, pt) /
              (3.0 * (1.0 - 2.0 * poissons_ratio(cell, pt)));
      mu = elastic_modulus(cell, pt) / (2.0 * (1.0 + poissons_ratio(cell, pt)));
      K  = hardening_modulus(cell, pt);
      Y  = yield_strength(cell, pt);
      Rd = recovery_modulus(cell, pt);
      Jm23 = std::pow(J(cell, pt), -2.0 / 3.0);

      // Compute Trial State
      be.clear();
      for (int i = 0; i < num_dims_; ++i)
        for (int j = 0; j < num_dims_; ++j)
          for (int p = 0; p < num_dims_; ++p)
            for (int q = 0; q < num_dims_; ++q)
              be(i, j) += Jm23 * def_grad(cell, pt, i, p) * Cpinv(p, q) *
                          def_grad(cell, pt, j, q);

      trd3  = minitensor::trace(be) / 3.;
      mubar = trd3 * mu;
      s     = mu * (be - trd3 * I);

      isoH = isoHardening_old(cell, pt);

      // check for yielding
      smag = minitensor::norm(s);
      Phi  = smag - sq23 * (Y + isoH);

      if (Phi > 1e-11) {  // plastic yielding

        // return mapping algorithm
        bool converged = false;
        int  iter      = 0;
        dgam           = 0.0;

        // initialize local unknown vector
        X[0] = dgam;
        X[1] = isoH;

        while (!converged) {
          ResidualJacobian(X, R, dRdX, isoH, smag, mubar, mu, kappa, K, Y, Rd);

          normR = R[0] * R[0] + R[1] * R[1];
          normR = std::sqrt(normR);

          if (iter == 0) normR0 = normR;
          if (normR0 != 0)
            conv = normR / normR0;
          else
            conv = normR0;

          //              std::cout << iter << " " << normR << " " << conv <<
          //              std::endl;
          if (conv < 1.e-11 || normR < 1.e-11) break;
          if (iter > 20) break;

          //            TEUCHOS_TEST_FOR_EXCEPTION( iter > 20,
          //            std::runtime_error,
          //                std::endl << "Error in return mapping, iter = " <<
          //                iter << "\nres = " << normR << "\nrelres = " << conv
          //                << std::endl);

          solver.solve(dRdX, X, R);
          iter++;
        }

        // compute sensitivity information w.r.t system parameters, and pack
        // back to X
        solver.computeFadInfo(dRdX, X, R);

        // update
        dgam = X[0];
        isoH = X[1];

        // plastic direction
        n = ScalarT(1. / smag) * s;

        // updated deviatoric stress
        s -= ScalarT(2. * mubar * dgam) * n;

        // update isoHardening
        isoHardening(cell, pt) = isoH;

        // update eqps
        eqps(cell, pt) = eqps_old(cell, pt) + sq23 * dgam;

        // exponential map to get Fp
        A    = dgam * n;
        expA = minitensor::exp<ScalarT>(A);

        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            Fp(i, j) = 0.0;
            for (int p = 0; p < num_dims_; ++p) {
              Fp(i, j) += expA(i, p) * Fpold(p, j);
            }
          }
        }

      } else {
        // set state variables to old values
        isoHardening(cell, pt) = isoHardening_old(cell, pt);

        eqps(cell, pt) = eqps_old(cell, pt);
        Fp             = Fpold;
      }

      // store logFp as the state variable
      logFp_n = minitensor::log(Fp);
      for (int i = 0; i < num_dims_; ++i)
        for (int j = 0; j < num_dims_; ++j)
          logFp(cell, pt, i, j) = logFp_n(i, j);

      // compute pressure
      p = 0.5 * kappa * (J(cell, pt) - 1 / (J(cell, pt)));

      // compute stress
      for (int i = 0; i < num_dims_; ++i) {
        for (int j = 0; j < num_dims_; ++j) {
          stress(cell, pt, i, j) = s(i, j) / J(cell, pt);
        }
        stress(cell, pt, i, i) += p;
      }
    }
  }

}  // end of compute state
//----------------------------------------------------------------------------
// all local functions for compute state
template <typename EvalT, typename Traits>
void
RIHMRModel<EvalT, Traits>::ResidualJacobian(
    std::vector<ScalarT>& X,
    std::vector<ScalarT>& R,
    std::vector<ScalarT>& dRdX,
    const ScalarT&        isoH,
    const ScalarT&        smag,
    const ScalarT&        mubar,
    ScalarT&              mu,
    ScalarT&              kappa,
    ScalarT&              K,
    ScalarT&              Y,
    ScalarT&              Rd)
{
  ScalarT               sq23 = std::sqrt(2.0 / 3.0);
  std::vector<DFadType> Rfad(2);
  std::vector<DFadType> Xfad(2);
  std::vector<ScalarT>  Xval(2);

  // initialize DFadType local unknown vector Xfad
  // Note that since Xfad is a temporary variable
  // that gets changed within local iterations.
  // Therefore, when we initialize Xfad, we only pass in the values of X
  // NOT the system sensitivity information
  Xval[0] = Sacado::ScalarValue<ScalarT>::eval(X[0]);
  Xval[1] = Sacado::ScalarValue<ScalarT>::eval(X[1]);

  Xfad[0] = DFadType(2, 0, Xval[0]);
  Xfad[1] = DFadType(2, 1, Xval[1]);

  DFadType smagfad, Yfad, d_isoH;

  DFadType dgam = Xfad[0], isoHfad = Xfad[1];

  // I have to break down these equations, to avoid compile error
  // Q.Chen.
  // smagfad = smag - 2.0 * mubar * dgam;
  smagfad = mubar * dgam;
  smagfad = 2.0 * smagfad;
  smagfad = smag - smagfad;

  // Yfad = sq23 * (Y + isoHfad);
  Yfad = Y + isoHfad;
  Yfad = sq23 * Yfad;

  // d_isoH = (K - Rd * isoHfad) * sq23 * dgam;
  d_isoH = Rd * isoHfad;
  d_isoH = K - d_isoH;
  d_isoH = d_isoH * dgam;
  d_isoH = d_isoH * sq23;

  // local nonlinear sys of equations
  Rfad[0] =
      smagfad - Yfad;  // Phi = smag - 2.* mubar * dgam - sq23 * (Y + isoHfad);
  Rfad[1] = isoHfad - isoH - d_isoH;

  // get ScalarT residual
  R[0] = Rfad[0].val();
  R[1] = Rfad[1].val();

  // get local Jacobian
  dRdX[0 + 2 * 0] = Rfad[0].dx(0);
  dRdX[0 + 2 * 1] = Rfad[0].dx(1);
  dRdX[1 + 2 * 0] = Rfad[1].dx(0);
  dRdX[1 + 2 * 1] = Rfad[1].dx(1);
}
//----------------------------------------------------------------------------
}  // namespace LCM
