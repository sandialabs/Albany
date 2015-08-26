//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

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
    Intrepid::Vector<FadT, N> & soln_prev)
{
  Intrepid::Index const
  dimension = soln_prev.get_dimension();

  Intrepid::Vector<FadT, N>
  resi_prev = residual.compute(soln_prev);

  Intrepid::Vector<ValueT, N>
  resi_prev_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi_prev);

  ValueT const
  initial_norm = Intrepid::norm(resi_prev_val);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    resi_prev = residual.compute(soln_prev);

    resi_prev_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi_prev);

    this->abs_error_ = Intrepid::norm(resi_prev_val);

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
        Hessian(i, j) = resi_prev(i).dx(j);
      }
    }

    Intrepid::Vector<ValueT, N> const
    soln_incr = - Intrepid::solve(Hessian, resi_prev_val);

    soln_prev += soln_incr;

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
    Intrepid::Vector<FadT, N> & soln_prev)
{
  Intrepid::Index const
  dimension = soln_prev.get_dimension();

  Intrepid::Vector<FadT, N>
  resi_prev = residual.compute(soln_prev);

  Intrepid::Vector<ValueT, N>
  resi_prev_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi_prev);

  ValueT const
  initial_norm = Intrepid::norm(resi_prev_val);

  ValueT
  step_length = this->initial_step_length_;

  Intrepid::Tensor<ValueT, N>
  I = Intrepid::identity<ValueT, N>(dimension);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  // Outer solution loop
  while (this->converged_ == false) {

    resi_prev = residual.compute(soln_prev);

    resi_prev_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi_prev);

    this->abs_error_ = Intrepid::norm(resi_prev_val);

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
        Hessian(i, j) = resi_prev(i).dx(j);
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

      step = - Intrepid::solve(K, resi_prev_val);

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
    soln_next = soln_prev + step;

    Intrepid::Vector<FadT, N> const
    resi_next = residual.compute(soln_next);

    // Compute reduction factor \rho_k in Nocedal's algorithm 11.5
    Intrepid::Vector<ValueT, N>
    resi_next_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(resi_next);

    ValueT const
    nr = Intrepid::norm_square(resi_prev_val);

    ValueT const
    nrp = Intrepid::norm_square(resi_next_val);

    ValueT const
    nrKp = Intrepid::norm_square(resi_prev_val + Intrepid::dot(Hessian, step));

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
      soln_prev = soln_next;
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
