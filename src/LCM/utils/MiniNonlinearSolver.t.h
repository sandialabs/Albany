//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// Specializations for MiniNonlinear solver
//

//
// Residual
//
template<typename Residual, Intrepid::Index N>
void
MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, Residual, N>::
solve(Residual & residual, Intrepid::Vector<ScalarT, N> & soln)
{
  this->nonlinear_method_.solve(residual, soln);
  return;
}

//
// Jacobian
//

//
// Tangent
//

//
// DistParamDeriv
//

#ifdef ALBANY_SG
//
// SGResidual
//

//
// SGJacobian
//

//
// SGTangent
//
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//

//
// MPJacobian
//

//
// MPTangent
//

#endif // ALBANY_ENSEMBLE

//
// Specializations for Newton solver
//

//
// Residual
//
template<typename Residual, Intrepid::Index N>
void
NewtonSolver<PHAL::AlbanyTraits::Residual, Residual, N>::
solve(
    Residual & residual,
    Intrepid::Vector<FadT, N> & soln)
{
  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<FadT, N>
  resi = residual.compute(soln);

  Intrepid::Vector<ValueT, N>
  resi_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi);

  ValueT const
  initial_norm = Intrepid::norm(resi_val);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    resi = residual.compute(soln);

    resi_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi);

    this->abs_error_ = Intrepid::norm(resi_val);

    this->rel_error_ = this->abs_error_ / initial_norm;

    bool const
    converged_relative = this->rel_error_ <= this->rel_tol_;

    bool const
    converged_absolute = this->abs_error_ <= this->abs_tol_;

    this->converged_ = converged_relative || converged_absolute;

    bool const
    is_max_iter = this->num_iter_ >= this->max_num_iter_;

    bool const
    end_solve = this->converged_ || is_max_iter;

    if (end_solve == true) break;

    Intrepid::Tensor<ValueT, N>
    Hessian(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = resi(i).dx(j);
      }
    }

    Intrepid::Vector<ValueT, N> const
    soln_incr = - Intrepid::solve(Hessian, resi_val);

    soln += soln_incr;

    ++this->num_iter_;
  }

  return;
}

//
// Jacobian
//

//
// Tangent
//

//
// DistParamDeriv
//

#ifdef ALBANY_SG
//
// SGResidual
//

//
// SGJacobian
//

//
// SGTangent
//
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//

//
// MPJacobian
//

//
// MPTangent
//

#endif // ALBANY_ENSEMBLE

//
// Specializations for trust-region solver
//

//
// Residual
//
template<typename Residual, Intrepid::Index N>
void
TrustRegionSolver<PHAL::AlbanyTraits::Residual, Residual, N>::
solve(
    Residual & residual,
    Intrepid::Vector<FadT, N> & soln)
{
  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<FadT, N>
  resi = residual.compute(soln);

  Intrepid::Vector<ValueT, N>
  resi_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi);

  ValueT const
  initial_norm = Intrepid::norm(resi_val);

  ValueT
  step_length = this->initial_step_length_;

  Intrepid::Tensor<ValueT, N>
  I = Intrepid::identity<ValueT, N>(dimension);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  // Outer solution loop
  while (this->converged_ == false) {

    resi = residual.compute(soln);

    resi_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi);

    this->abs_error_ = Intrepid::norm(resi_val);

    this->rel_error_ = this->abs_error_ / initial_norm;

    bool const
    converged_relative = this->rel_error_ <= this->rel_tol_;

    bool const
    converged_absolute = this->abs_error_ <= this->abs_tol_;

    this->converged_ = converged_relative || converged_absolute;

    bool const
    is_max_iter = this->num_iter_ >= this->max_num_iter_;

    bool const
    end_solve = this->converged_ || is_max_iter;

    if (end_solve == true) break;

    Intrepid::Tensor<ValueT, N>
    Hessian(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = resi(i).dx(j);
      }
    }

    // Restrict step to size of trust region. Exact algorithm, Nocedal 4.4
    ValueT
    lambda = 0.0;

    Intrepid::Tensor<ValueT, N>
    K(dimension);

    Intrepid::Tensor<ValueT, N>
    L(dimension);

    Intrepid::Vector<ValueT, N>
    step;

    Intrepid::Vector<ValueT, N>
    q;

    for (Intrepid::Index i{0}; i < this->max_num_restrict_iter_; ++i) {

      K = Hessian + lambda * I;

      L = Intrepid::cholesky(K).first;

      step = - Intrepid::solve(K, resi_val);

      q = Intrepid::solve(L, step);

      ValueT const
      nps = Intrepid::norm_square(step);

      ValueT const
      nqs = Intrepid::norm_square(q);

      ValueT const
      lambda_incr = nps * (std::sqrt(nps) - step_length) / nqs / step_length;

      lambda += std::max(lambda_incr, 0.0);

    }

    Intrepid::Vector<FadT, N> const
    soln_next = soln + step;

    Intrepid::Vector<FadT, N> const
    resi_next = residual.compute(soln_next);

    // Compute reduction factor \rho_k in Nocedal's algorithm 11.5
    Intrepid::Vector<ValueT, N>
    resi_next_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi_next);

    ValueT const
    nr = Intrepid::norm_square(resi_val);

    ValueT const
    nrp = Intrepid::norm_square(resi_next_val);

    ValueT const
    nrKp = Intrepid::norm_square(resi_val + Intrepid::dot(Hessian, step));

    ValueT const
    reduction = (nr - nrp) / (nr - nrKp);

    // Determine whether the trust region should be increased, decreased
    // or left the same.
    ValueT const
    computed_length = Intrepid::norm(step);

    if (reduction < 0.25) {

      step_length = 0.25 * computed_length;

    } else {

      bool const
      at_boundary_region = std::abs(computed_length - step_length) <= 1.0e-3;

      bool const
      increase_step_length = reduction > 0.75 && at_boundary_region;

      if (increase_step_length == true) {
        step_length = std::min(2.0 * step_length, this->max_step_length_);
      }

    }

    if (reduction > this->min_reduction_) {
      soln = soln_next;
    }

    ++this->num_iter_;
  }

  return;
}

//
// Jacobian
//

//
// Tangent
//

//
// DistParamDeriv
//

#ifdef ALBANY_SG
//
// SGResidual
//

//
// SGJacobian
//

//
// SGTangent
//
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//

//
// MPJacobian
//

//
// MPTangent
//

#endif // ALBANY_ENSEMBLE

//
// Specializations for Conjugate Gradient solver
//

//
// Residual
//
template<typename Residual, Intrepid::Index N>
void
ConjugateGradientSolver<PHAL::AlbanyTraits::Residual, Residual, N>::
solve(
    Residual & residual,
    Intrepid::Vector<FadT, N> & soln)
{
  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<FadT, N>
  resi = - residual.compute(soln);

  Intrepid::Vector<ValueT, N>
  resi_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi);

  Intrepid::Tensor<ValueT, N>
  Hessian(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    for (Intrepid::Index j{0}; j < dimension; ++j) {
      Hessian(i, j) = resi(i).dx(j);
    }
  }

  Intrepid::Vector<ValueT, N>
  precon_resi_val = Intrepid::solve(Hessian, resi_val);

  Intrepid::Vector<ValueT, N>
  search_direction = precon_resi_val;

  ValueT
  projection_new = Intrepid::dot(resi_val, search_direction);

  ValueT const
  initial_norm = Intrepid::norm(resi_val);

  this->num_iter_ = 0;

  Intrepid::Index
  restart_directions_counter = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    resi = - residual.compute(soln);

    resi_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi);

    this->abs_error_ = Intrepid::norm(resi_val);

    this->rel_error_ = this->abs_error_ / initial_norm;

    bool const
    converged_relative = this->rel_error_ <= this->rel_tol_;

    bool const
    converged_absolute = this->abs_error_ <= this->abs_tol_;

    this->converged_ = converged_relative || converged_absolute;

    bool const
    is_max_iter = this->num_iter_ >= this->max_num_iter_;

    bool const
    end_solve = this->converged_ || is_max_iter;

    if (end_solve == true) break;

    ValueT
    projection_search = Intrepid::dot(search_direction, search_direction);

    ValueT
    step_length = - this->initial_secant_step_length_;

    Intrepid::Vector<FadT, N> const
    trial_soln = soln + this->initial_secant_step_length_ * search_direction;

    Intrepid::Vector<FadT, N> const
    trial_gradient = residual.compute(trial_soln);

    Intrepid::Vector<ValueT, N> const
    trial_gradient_val =
        Sacado::Value<Intrepid::Vector<FadT, N>>::eval(trial_gradient);

    ValueT
    projection_prev = Intrepid::dot(trial_gradient_val, search_direction);

    // Secant line search.

    for (Intrepid::Index i{0}; i < this->max_num_secant_iter_; ++i) {

      Intrepid::Vector<FadT, N> const
      gradient = residual.compute(soln);

      Intrepid::Vector<ValueT, N> const
      gradient_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(gradient);

      ValueT const
      projection = Intrepid::dot(gradient_val, search_direction);

      step_length *= projection / (projection_prev - projection);

      soln += step_length * search_direction;

      projection_prev = projection;

      bool const
      secant_converged = step_length * step_length * projection_search <=
        this->secant_tol_ * this->secant_tol_;

      if (secant_converged == true) break;

    }

    resi = - residual.compute(soln);

    resi_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = resi(i).dx(j);
      }
    }

    ValueT const
    projection_old = projection_new;

    ValueT const
    projection_mid = Intrepid::dot(resi_val, precon_resi_val);

    precon_resi_val = Intrepid::solve(Hessian, resi_val);

    projection_new = Intrepid::dot(resi_val, precon_resi_val);

    ValueT const
    gram_schmidt_factor = (projection_new - projection_mid) / projection_old;

    ++restart_directions_counter;

    bool const
    restart_directions =
        restart_directions_counter == this->restart_directions_interval_ ||
        gram_schmidt_factor <= 0.0;

    if (restart_directions == true) {

      search_direction = precon_resi_val;
      restart_directions_counter = 0;

    } else {

      search_direction =
          precon_resi_val + gram_schmidt_factor * search_direction;

    }

    ++this->num_iter_;
  }

  return;
}

//
// Jacobian
//

//
// Tangent
//

//
// DistParamDeriv
//

#ifdef ALBANY_SG
//
// SGResidual
//

//
// SGJacobian
//

//
// SGTangent
//
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//

//
// MPJacobian
//

//
// MPTangent
//

#endif // ALBANY_ENSEMBLE

} // namespace LCM
