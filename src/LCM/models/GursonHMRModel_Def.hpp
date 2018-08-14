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

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
GursonHMRModel<EvalT, Traits>::GursonHMRModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      sat_mod_(p->get<RealType>("Saturation Modulus", 0.0)),
      sat_exp_(p->get<RealType>("Saturation Exponent", 0.0)),
      f0_(p->get<RealType>("Initial Void Volume", 0.0)),
      kw_(p->get<RealType>("Shear Damage Parameter", 0.0)),
      eN_(p->get<RealType>("Void Nucleation Parameter eN", 0.0)),
      sN_(p->get<RealType>("Void Nucleation Parameter sN", 0.1)),
      fN_(p->get<RealType>("Void Nucleation Parameter fN", 0.0)),
      fc_(p->get<RealType>("Critical Void Volume", 1.0)),
      ff_(p->get<RealType>("Failure Void Volume", 1.0)),
      q1_(p->get<RealType>("Yield Parameter q1", 1.0)),
      q2_(p->get<RealType>("Yield Parameter q2", 1.0)),
      q3_(p->get<RealType>("Yield Parameter q3", 1.0))
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
  std::string Fp_string           = (*field_name_map_)["Fp"];
  std::string eqps_string         = (*field_name_map_)["eqps"];
  std::string ess_string          = (*field_name_map_)["ess"];
  std::string isoHardening_string = (*field_name_map_)["isoHardening"];
  std::string void_string         = (*field_name_map_)["Void_Volume"];

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(ess_string, dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair(isoHardening_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(void_string, dl->qp_scalar));

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
  // Fp
  this->num_state_variables_++;
  this->state_var_names_.push_back(Fp_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(1.0);
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
  // ess
  this->num_state_variables_++;
  this->state_var_names_.push_back(ess_string);
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
  //
  // void volume
  this->num_state_variables_++;
  this->state_var_names_.push_back(void_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(f0_);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
}
//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
GursonHMRModel<EvalT, Traits>::computeState(
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
  std::string Fp_string           = (*field_name_map_)["Fp"];
  std::string eqps_string         = (*field_name_map_)["eqps"];
  std::string ess_string          = (*field_name_map_)["ess"];
  std::string isoHardening_string = (*field_name_map_)["isoHardening"];
  std::string void_string         = (*field_name_map_)["Void_Volume"];

  // extract evaluated MDFields
  auto stress       = *eval_fields[cauchy_string];
  auto Fp           = *eval_fields[Fp_string];
  auto eqps         = *eval_fields[eqps_string];
  auto ess          = *eval_fields[ess_string];
  auto isoHardening = *eval_fields[isoHardening_string];
  auto void_volume  = *eval_fields[void_string];

  // get State Variables
  Albany::MDArray Fp_old   = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray eqps_old = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray ess_old  = (*workset.stateArrayPtr)[ess_string + "_old"];
  Albany::MDArray isoHardening_old =
      (*workset.stateArrayPtr)[isoHardening_string + "_old"];
  Albany::MDArray void_volume_old =
      (*workset.stateArrayPtr)[void_string + "_old"];

  minitensor::Tensor<ScalarT> F(num_dims_), be(num_dims_), logbe(num_dims_);
  minitensor::Tensor<ScalarT> s(num_dims_), sigma(num_dims_), N(num_dims_);
  minitensor::Tensor<ScalarT> A(num_dims_), expA(num_dims_), Fpnew(num_dims_);
  minitensor::Tensor<ScalarT> Fpn(num_dims_), Fpinv(num_dims_),
      Cpinv(num_dims_);
  minitensor::Tensor<ScalarT> dPhi(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  ScalarT kappa, mu, H, Y, Rd;
  ScalarT p, trlogbeby3, detbe;
  ScalarT fvoid, eq, es, isoH, Phi, dgam, Ybar;

  // local unknowns and residual vectors
  std::vector<ScalarT> X(4);
  std::vector<ScalarT> R(4);
  std::vector<ScalarT> dRdX(16);
  ScalarT norm_residual0(0.0), norm_residual(0.0), relative_residual(0.0);
  LocalNonlinearSolver<EvalT, Traits> solver;

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      kappa = elastic_modulus(cell, pt) /
              (3.0 * (1.0 - 2.0 * poissons_ratio(cell, pt)));
      mu = elastic_modulus(cell, pt) / (2.0 * (1.0 + poissons_ratio(cell, pt)));
      H  = hardening_modulus(cell, pt);
      Y  = yield_strength(cell, pt);
      Rd = recovery_modulus(cell, pt);

      // fill local tensors
      F.fill(def_grad, cell, pt, 0, 0);
      // Fpn.fill( &Fpold(cell,pt,int(0),int(0)) );
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fpn(i, j) = static_cast<ScalarT>(Fp_old(cell, pt, i, j));
        }
      }

      // compute trial state
      Fpinv      = minitensor::inverse(Fpn);
      Cpinv      = Fpinv * minitensor::transpose(Fpinv);
      be         = F * Cpinv * minitensor::transpose(F);
      logbe      = minitensor::log_sym<ScalarT>(be);
      trlogbeby3 = minitensor::trace(logbe) / 3.0;
      detbe      = minitensor::det<ScalarT>(be);
      s          = mu * (logbe - trlogbeby3 * I);
      p          = 0.5 * kappa * std::log(detbe);
      fvoid      = void_volume_old(cell, pt);
      eq         = eqps_old(cell, pt);
      es         = ess_old(cell, pt);
      isoH       = isoHardening_old(cell, pt);

      // check yield condition
      Phi = YieldFunction(s, p, fvoid, Y, isoH, J(cell, pt));

      dgam = 0.0;
      if (Phi > 0.0) {  // plastic yielding

        // initialize local unknown vector
        X[0] = dgam;
        X[1] = p;
        X[2] = fvoid;
        X[3] = es;

        int iter          = 0;
        norm_residual0    = 0.0;
        norm_residual     = 0.0;
        relative_residual = 0.0;

        // local N-R loop
        while (true) {
          ResidualJacobian(
              X, R, dRdX, p, fvoid, es, s, mu, kappa, H, Y, Rd, J(cell, pt));

          norm_residual = 0.0;
          for (int i = 0; i < 4; i++) norm_residual += R[i] * R[i];

          norm_residual = std::sqrt(norm_residual);

          if (iter == 0) norm_residual0 = norm_residual;

          if (norm_residual0 != 0)
            relative_residual = norm_residual / norm_residual0;
          else
            relative_residual = norm_residual0;

          // std::cout << iter << " "
          //<< norm_residual << " " << relative_residual << std::endl;

          if (relative_residual < 1.0e-11 || norm_residual < 1.0e-11) break;

          if (iter > 20) break;

          // call local nonlinear solver
          solver.solve(dRdX, X, R);

          iter++;
        }  // end of local N-R loop

        // compute sensitivity information w.r.t. system parameters
        // and pack the sensitivity back to X
        solver.computeFadInfo(dRdX, X, R);

        // update
        dgam  = X[0];
        p     = X[1];
        fvoid = X[2];
        es    = X[3];

        isoH = 2.0 * mu * es;

        // accounts for void coalescence
        //          fvoid_star = fvoid;
        //          if ((fvoid > fc_) && (fvoid < ff_)) {
        //            if ((ff_ - fc_) != 0.0) {
        //              fvoid_star = fc_ + (fvoid - fc_) * (1.0 / q1_ - fc_) /
        //              (ff_ - fc_);
        //            }
        //          }
        //          else if (fvoid >= ff_) {
        //            fvoid_star = 1.0 / q1_;
        //            if (fvoid_star > 1.0)
        //              fvoid_star = 1.0;
        //          }

        // deviatoric stress tensor
        s = (1.0 / (1.0 + 2.0 * mu * dgam)) * s;

        // hardening
        Ybar = Y + isoH;

        // Kirchhoff_yield_stress = Cauchy_yield_stress * J
        // Ybar = Ybar * J(cell, pt);

        // dPhi w.r.t. dKirchhoff_stress
        ScalarT tmp = 1.5 * q2_ * p / Ybar;
        ScalarT deq = dgam / Ybar / (1.0 - fvoid) *
                      (minitensor::dotdot(s, s) +
                       q1_ * q2_ * p * Ybar * fvoid * std::sinh(tmp));
        eq = eq + deq;

        dPhi = s + 1.0 / 3.0 * q1_ * q2_ * Ybar * fvoid * std::sinh(tmp) * I;

        expA = minitensor::exp(dgam * dPhi);

        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            Fp(cell, pt, i, j) = 0.0;
            for (int k(0); k < num_dims_; ++k) {
              Fp(cell, pt, i, j) += expA(i, k) * Fpn(k, j);
            }
          }
        }

        eqps(cell, pt)         = eq;
        ess(cell, pt)          = es;
        isoHardening(cell, pt) = isoH;
        void_volume(cell, pt)  = fvoid;

      }       // end of plastic loading
      else {  // elasticity, set state variables to previous values

        eqps(cell, pt)         = eqps_old(cell, pt);
        ess(cell, pt)          = ess_old(cell, pt);
        isoHardening(cell, pt) = isoHardening_old(cell, pt);
        void_volume(cell, pt)  = void_volume_old(cell, pt);

        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            Fp(cell, pt, i, j) = Fp_old(cell, pt, i, j);
          }
        }

      }  // end of elasticity

      // compute Cauchy stress tensor
      // note that p also has to be divided by J
      // because the one computed from return mapping is the Kirchhoff pressure
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = s(i, j) / J(cell, pt);
        }
        stress(cell, pt, i, i) += p / J(cell, pt);
      }

    }  // end of loop over Gauss points
  }    // end of loop over cells

}  // end of compute state

//------------------------------------------------------------------------------
// all local functions for compute state
template <typename EvalT, typename Traits>
typename EvalT::ScalarT
GursonHMRModel<EvalT, Traits>::YieldFunction(
    minitensor::Tensor<ScalarT> const& s,
    ScalarT const&                     p,
    ScalarT const&                     fvoid,
    ScalarT const&                     Y,
    ScalarT const&                     isoH,
    ScalarT const&                     jacobian)
{
  // yield strength
  ScalarT Ybar = Y + isoH;

  // Kirchhoff yield stress
  // Ybar = Ybar * jacobian;

  ScalarT tmp = 1.5 * q2_ * p / Ybar;

  // acounts for void coalescence
  //    ScalarT fvoid_star = fvoid;
  //    if ((fvoid > fc_) && (fvoid < ff_)) {
  //      if ((ff_ - fc_) != 0.0) {
  //        fvoid_star = fc_ + (fvoid - fc_) * (1. / q1_ - fc_) / (ff_ - fc_);
  //      }
  //    }
  //    else if (fvoid >= ff_) {
  //      fvoid_star = 1.0 / q1_;
  //      if (fvoid_star > 1.0)
  //        fvoid_star = 1.0;
  //    }

  //    ScalarT psi = 1.0 + q3_ * fvoid_star * fvoid_star
  //        - 2.0 * q1_ * fvoid_star * std::cosh(tmp);

  ScalarT psi = 1.0 + q3_ * fvoid * fvoid - 2.0 * q1_ * fvoid * std::cosh(tmp);

  // a quadratic representation will look like:
  ScalarT Phi = 0.5 * minitensor::dotdot(s, s) - psi * Ybar * Ybar / 3.0;

  // linear form
  // ScalarT smag = minitensor::dotdot(s,s);
  // smag = std::sqrt(smag);
  // ScalarT sq23 = std::sqrt(2./3.);
  // ScalarT Phi = smag - sq23 * std::sqrt(psi) * psi_sign * Ybar

  return Phi;
}  // end of YieldFunction

template <typename EvalT, typename Traits>
void
GursonHMRModel<EvalT, Traits>::ResidualJacobian(
    std::vector<ScalarT>&        X,
    std::vector<ScalarT>&        R,
    std::vector<ScalarT>&        dRdX,
    const ScalarT&               p,
    const ScalarT&               fvoid,
    const ScalarT&               es,
    minitensor::Tensor<ScalarT>& s,
    const ScalarT&               mu,
    const ScalarT&               kappa,
    const ScalarT&               H,
    const ScalarT&               Y,
    const ScalarT&               Rd,
    const ScalarT&               jacobian)
{
  ScalarT               sq32 = std::sqrt(3.0 / 2.0);
  ScalarT               sq23 = std::sqrt(2.0 / 3.0);
  std::vector<DFadType> Rfad(4);
  std::vector<DFadType> Xfad(4);
  // initialize DFadType local unknown vector Xfad
  // Note that since Xfad is a temporary variable
  // that gets changed within local iterations
  // when we initialize Xfad, we only pass in the values of X,
  // NOT the system sensitivity information
  std::vector<ScalarT> Xval(4);
  for (int i = 0; i < 4; ++i) {
    Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
    Xfad[i] = DFadType(4, i, Xval[i]);
  }

  DFadType dgam     = Xfad[0];
  DFadType pFad     = Xfad[1];
  DFadType fvoidFad = Xfad[2];
  DFadType esFad    = Xfad[3];

  // accounts for void coalescence
  //    DFadType fvoidFad_star = fvoidFad;
  //
  //    if ((fvoidFad > fc_) && (fvoidFad < ff_)) {
  //      if ((ff_ - fc_) != 0.0) {
  //        fvoidFad_star = fc_ + (fvoidFad - fc_) * (1. / q1_ - fc_) / (ff_ -
  //        fc_);
  //      }
  //    }
  //    else if (fvoidFad >= ff_) {
  //      fvoidFad_star = 1.0 / q1_;
  //      if (fvoidFad_star > 1.0)
  //        fvoidFad_star = 1.0;
  //    }

  // yield strength
  DFadType Ybar;  // Ybar = Y + 2.0 * mu * esFad;
  Ybar = mu * esFad;
  Ybar = Y + 2.0 * Ybar;

  // Kirchhoff yield stress
  // Ybar = Ybar * jacobian;

  DFadType tmp = 1.5 * q2_ * pFad / Ybar;

  DFadType psi =
      1.0 + q3_ * fvoidFad * fvoidFad - 2.0 * q1_ * fvoidFad * std::cosh(tmp);

  DFadType factor = 1.0 / (1.0 + (2.0 * (mu * dgam)));

  // valid for assumption Ntr = N;
  minitensor::Tensor<DFadType> sfad(num_dims_);
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < num_dims_; ++j) { sfad(i, j) = factor * s(i, j); }
  }

  // currently complaining error in promotion tensor type
  // sfad = factor * s;

  // shear-dependent term in void growth
  DFadType omega(0.0), J3(0.0), taue(0.0), smag2, smag;
  J3    = minitensor::det(sfad);
  smag2 = minitensor::dotdot(sfad, sfad);
  if (smag2 > 0.0) {
    smag = std::sqrt(smag2);
    taue = sq32 * smag;
  }

  if (taue > 0.0)
    omega = 1.0 - (27.0 * J3 / 2.0 / taue / taue / taue) *
                      (27.0 * J3 / 2.0 / taue / taue / taue);

  DFadType deq(0.0);
  if (smag != 0.0) {
    deq = dgam * (smag2 + q1_ * q2_ * pFad * Ybar * fvoidFad * std::sinh(tmp)) /
          (1.0 - fvoidFad) / Ybar;
  } else {
    deq = dgam * (q1_ * q2_ * pFad * Ybar * fvoidFad * std::sinh(tmp)) /
          (1.0 - fvoidFad) / Ybar;
  }

  DFadType des = (H - Rd * esFad) * deq;

  // void nucleation (to be added later)
  DFadType dfn(0.0);
  //    DFadType An(0.0), eratio(0.0);
  //    eratio = -0.5 * (eqFad - eN_) * (eqFad - eN_) / sN_ / sN_;
  //    const double pi = acos(-1.0);
  //    if (pFad >= 0.0) {
  //      An = fN_ / sN_ / (std::sqrt(2.0 * pi)) * std::exp(eratio);
  //    }
  //
  //    dfn = An * deq;

  // void growth
  // fvoidFad or fvoidFad_star
  DFadType dfg(0.0);
  if (taue > 0.0) {
    dfg =
        dgam * q1_ * q2_ * (1.0 - fvoidFad) * fvoidFad * Ybar * std::sinh(tmp) +
        sq23 * dgam * kw_ * fvoidFad * omega * smag;
  } else {
    dfg =
        dgam * q1_ * q2_ * (1.0 - fvoidFad) * fvoidFad * Ybar * std::sinh(tmp);
  }

  DFadType Phi;
  Phi = 0.5 * smag2 - psi * Ybar * Ybar / 3.0;

  // local system of equations
  Rfad[0] = Phi;
  Rfad[1] =
      pFad - p + dgam * q1_ * q2_ * kappa * Ybar * fvoidFad * std::sinh(tmp);
  Rfad[2] = fvoidFad - fvoid - dfg - dfn;
  Rfad[3] = esFad - es - des;

  // get ScalarT Residual
  for (int i = 0; i < 4; i++) R[i] = Rfad[i].val();

  // get local Jacobian
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) dRdX[i + 4 * j] = Rfad[i].dx(j);

}  // end of ResidualJacobian
//------------------------------------------------------------------------------
}  // namespace LCM
