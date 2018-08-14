//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <typeinfo>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
DruckerPragerModel<EvalT, Traits>::DruckerPragerModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      a0_(p->get<RealType>("Initial Friction Parameter a0", 0.0)),
      a1_(p->get<RealType>("Hardening Parameter a1", 0.0)),
      a2_(p->get<RealType>("Hardening Parameter a2", 1.0)),
      a3_(p->get<RealType>("Hardening Parameter a3", 1.0)),
      b0_(p->get<RealType>("Critical Friction Coefficient b0", 0.0)),
      Cf_(p->get<RealType>("Cohesion Parameter Cf", 0.0)),
      Cg_(p->get<RealType>("Plastic Potential Parameter Cg", 0.0))
{
  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("Strain", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));

  // retrieve appropriate field name strings
  std::string cauchy_string   = (*field_name_map_)["Cauchy_Stress"];
  std::string strain_string   = (*field_name_map_)["Strain"];
  std::string eqps_string     = (*field_name_map_)["eqps"];
  std::string friction_string = (*field_name_map_)["Friction_Parameter"];

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(friction_string, dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair("Material Tangent", dl->qp_tensor4));

  // define the state variables
  // strain
  this->num_state_variables_++;
  this->state_var_names_.push_back(strain_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
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
  // alpha (friction parameter)
  this->num_state_variables_++;
  this->state_var_names_.push_back(friction_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(a0_);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
}
//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
DruckerPragerModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // extract dependent MDFields
  auto strain          = *dep_fields["Strain"];
  auto poissons_ratio  = *dep_fields["Poissons Ratio"];
  auto elastic_modulus = *dep_fields["Elastic Modulus"];

  // retrieve appropriate field name strings
  std::string cauchy_string   = (*field_name_map_)["Cauchy_Stress"];
  std::string strain_string   = (*field_name_map_)["Strain"];
  std::string eqps_string     = (*field_name_map_)["eqps"];
  std::string friction_string = (*field_name_map_)["Friction_Parameter"];

  // extract evaluated MDFields
  auto stress   = *eval_fields[cauchy_string];
  auto eqps     = *eval_fields[eqps_string];
  auto friction = *eval_fields[friction_string];
  auto tangent  = *eval_fields["Material Tangent"];

  // get State Variables
  Albany::MDArray strainold = (*workset.stateArrayPtr)[strain_string + "_old"];
  Albany::MDArray stressold = (*workset.stateArrayPtr)[cauchy_string + "_old"];
  Albany::MDArray eqpsold   = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray frictionold =
      (*workset.stateArrayPtr)[friction_string + "_old"];

  minitensor::Tensor<ScalarT>  id(minitensor::eye<ScalarT>(num_dims_));
  minitensor::Tensor4<ScalarT> id1(minitensor::identity_1<ScalarT>(num_dims_));
  minitensor::Tensor4<ScalarT> id2(minitensor::identity_2<ScalarT>(num_dims_));
  minitensor::Tensor4<ScalarT> id3(minitensor::identity_3<ScalarT>(num_dims_));

  minitensor::Tensor4<ScalarT> Celastic(num_dims_);
  minitensor::Tensor<ScalarT> sigma(num_dims_), sigmaN(num_dims_), s(num_dims_);
  minitensor::Tensor<ScalarT> epsilon(num_dims_), epsilonN(num_dims_);
  minitensor::Tensor<ScalarT> depsilon(num_dims_);
  minitensor::Tensor<ScalarT> nhat(num_dims_);

  ScalarT lambda, mu, kappa;
  ScalarT alpha, alphaN;
  ScalarT p, q, ptr, qtr;
  ScalarT eq, eqN, deq;
  ScalarT snorm;
  ScalarT Phi;

  // local unknowns and residual vectors
  std::vector<ScalarT> X(4);
  std::vector<ScalarT> R(4);
  std::vector<ScalarT> dRdX(16);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      lambda =
          (elastic_modulus(cell, pt) * poissons_ratio(cell, pt)) /
          ((1 + poissons_ratio(cell, pt)) * (1 - 2 * poissons_ratio(cell, pt)));
      mu    = elastic_modulus(cell, pt) / (2 * (1 + poissons_ratio(cell, pt)));
      kappa = lambda + 2.0 * mu / 3.0;

      // 4-th order elasticity tensor
      Celastic = lambda * id3 + mu * (id1 + id2);

      // previous state (the fill doesn't work for state virable)
      // sigmaN.fill( &stressold(cell,pt,0,0) );
      // epsilonN.fill( &strainold(cell,pt,0,0) );

      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          sigmaN(i, j)   = stressold(cell, pt, i, j);
          epsilonN(i, j) = strainold(cell, pt, i, j);
          // epsilon(i,j) = strain(cell,pt,i,j);
        }
      }

      epsilon.fill(strain, cell, pt, 0, 0);
      depsilon = epsilon - epsilonN;

      alphaN = frictionold(cell, pt);
      eqN    = eqpsold(cell, pt);

      // trial state
      sigma = sigmaN + minitensor::dotdot(Celastic, depsilon);
      ptr   = minitensor::trace(sigma) / 3.0;
      s     = sigma - ptr * id;
      snorm = minitensor::dotdot(s, s);
      if (snorm > 0) snorm = std::sqrt(snorm);
      qtr = sqrt(3.0 / 2.0) * snorm;

      // unit deviatoric tensor
      if (snorm > 0) {
        nhat = s / snorm;
      } else {
        nhat = id;
      }

      // check yielding
      Phi = qtr + alphaN * ptr - Cf_;

      alpha = alphaN;
      p     = ptr;
      q     = qtr;
      deq   = 0.0;
      if (Phi > 1.0e-12) {  // plastic yielding

        // initialize local unknown vector
        X[0] = ptr;
        X[1] = qtr;
        X[2] = alpha;
        X[3] = deq;

        LocalNonlinearSolver<EvalT, Traits> solver;
        int                                 iter = 0;
        ScalarT norm_residual0(0.0), norm_residual(0.0), relative_residual(0.0);

        // local N-R loop
        while (true) {
          ResidualJacobian(X, R, dRdX, ptr, qtr, eqN, mu, kappa);

          norm_residual = 0.0;
          for (int i = 0; i < 4; i++) norm_residual += R[i] * R[i];
          norm_residual = std::sqrt(norm_residual);

          if (iter == 0) norm_residual0 = norm_residual;

          if (norm_residual0 != 0)
            relative_residual = norm_residual / norm_residual0;
          else
            relative_residual = norm_residual0;

          // std::cout << iter << " "
          //<< Sacado::ScalarValue<ScalarT>::eval(norm_residual)
          //<< " " << Sacado::ScalarValue<ScalarT>::eval(relative_residual)
          //<< std::endl;

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
        p     = X[0];
        q     = X[1];
        alpha = X[2];
        deq   = X[3];

      }  // end plastic yielding

      eq = eqN + deq;

      s     = sqrt(2.0 / 3.0) * q * nhat;
      sigma = s + p * id;

      eqps(cell, pt)     = eq;
      friction(cell, pt) = alpha;

      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }

    }  // end loop over pt
  }    //  end loop over cell
}
//----------------------------------------------------------------------------
// all local functions for compute state
template <typename EvalT, typename Traits>
void
DruckerPragerModel<EvalT, Traits>::ResidualJacobian(
    std::vector<ScalarT>& X,
    std::vector<ScalarT>& R,
    std::vector<ScalarT>& dRdX,
    const ScalarT         ptr,
    const ScalarT         qtr,
    const ScalarT         eqN,
    const ScalarT         mu,
    const ScalarT         kappa)
{
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

  DFadType pFad     = Xfad[0];
  DFadType qFad     = Xfad[1];
  DFadType alphaFad = Xfad[2];
  DFadType deqFad   = Xfad[3];

  DFadType betaFad = alphaFad - b0_;

  // check this
  DFadType eqFad = eqN + deqFad;  // (eqFad + deqFad??)

  // have to break down 3.0 * mu * deqFad;
  // other wise there wil be compiling error
  DFadType dq = deqFad;
  dq          = mu * dq;
  dq          = 3.0 * dq;

  // local system of equations
  Rfad[0] = pFad - ptr + kappa * betaFad * deqFad;
  Rfad[1] = qFad - qtr + dq;
  Rfad[2] = alphaFad - (a0_ + a1_ * eqFad * std::exp(a2_ * pFad - a3_ * eqFad));
  Rfad[3] = qFad + alphaFad * pFad - Cf_;

  // get ScalarT Residual
  for (int i = 0; i < 4; i++) R[i] = Rfad[i].val();

  // get local Jacobian
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) dRdX[i + 4 * j] = Rfad[i].dx(j);

}  // end of ResidualJacobian
//------------------------------------------------------------------------------
}  // namespace LCM
