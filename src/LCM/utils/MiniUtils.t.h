//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
//
//
template<typename NLS, typename T, Intrepid::Index N>
std::unique_ptr<NonlinearMethod_Base<NLS, T, N>>
nonlinearMethodFactory(NonlinearMethod const method_type)
{
  std::unique_ptr<NonlinearMethod_Base<NLS, T, N>>
  method = nullptr;

  switch (method_type) {

  default:
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << std::endl;
    std::cerr << "Unknown nonlinear method.";
    std::cerr << std::endl;
    exit(1);
    break;

  case NonlinearMethod::NEWTON:
    method = new NewtonMethod<NLS, T, N>();
    break;

  case NonlinearMethod::TRUST_REGION:
    method = new TrustRegionMethod<NLS, T, N>();
    break;

  case NonlinearMethod::CONJUGATE_GRADIENT:
    method = new ConjugateGradientMethod<NLS, T, N>();
    break;

  }

  return method;
}

//
//
//
template<typename T, typename S, Intrepid::Index N>
void
computeFADInfo(
    Intrepid::Vector<T, N> const & r,
    Intrepid::Tensor<S, N> const & DrDx,
    Intrepid::Vector<T, N> & x)
{
  // Check whether dealing with AD type.
  if (Sacado::IsADType<T>::value == false) return;

  //Deal with derivative information
  auto const
  dimension = r.get_dimension();

  assert(dimension > 0);

  auto const
  order = r[0].size();

  assert(order > 0);

  // Extract sensitivities of r wrt p
  Intrepid::Matrix<S>
  DrDp(dimension, order);

  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DrDp(i, j) = r(i).dx(j);
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<S>
  DxDp = Intrepid::solve(DrDx, DrDp);

  // Pack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

}

//
//
//
template<typename NLS, typename T, Intrepid::Index N>
void
NewtonMethod<NLS, T, N>::
solve(NLS & nls, Intrepid::Vector<T, N> & soln)
{
  using AD = typename Sacado::Fad::DFad<T>;

  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<T, N>
  resi = nls.compute(soln);

  Intrepid::Vector<AD, N>
  soln_ad(dimension);

  Intrepid::Vector<AD, N>
  resi_ad(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    soln_ad(i) = AD(dimension, i, soln(i));
  }

  T const
  initial_norm = Intrepid::norm(resi);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    resi_ad = nls.compute(soln_ad);

    resi = Sacado::Value<Intrepid::Vector<AD, N>>::eval(resi_ad);

    this->abs_error_ = Intrepid::norm(resi);

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

    Intrepid::Tensor<T, N>
    Hessian(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = resi_ad(i).dx(j);
      }
    }

    Intrepid::Vector<T, N> const
    soln_incr = - Intrepid::solve(Hessian, resi);

    soln_ad += soln_incr;

    ++this->num_iter_;
  }

  soln = Sacado::Value<Intrepid::Vector<AD, N>>::eval(soln_ad);

  return;

}

//
// Trust Region method.  See Nocedal's algorithm 11.5.
//
template<typename NLS, typename T, Intrepid::Index N>
void
TrustRegionMethod<NLS, T, N>::
solve(NLS & nls, Intrepid::Vector<T, N> & soln)
{
  using AD = typename Sacado::Fad::DFad<T>;

  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<T, N>
  resi = nls.compute(soln);

  Intrepid::Vector<AD, N>
  soln_ad(dimension);

  Intrepid::Vector<AD, N>
  resi_ad(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    soln_ad(i) = AD(dimension, i, soln(i));
  }

  T const
  initial_norm = Intrepid::norm(resi);

  T
  step_length = this->initial_step_length_;

  Intrepid::Tensor<T, N>
  I = Intrepid::identity<T, N>(dimension);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  // Outer solution loop
  while (this->converged_ == false) {

    resi_ad = nls.compute(soln_ad);

    resi = Sacado::Value<Intrepid::Vector<AD, N>>::eval(resi_ad);

    this->abs_error_ = Intrepid::norm(resi);

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

    Intrepid::Tensor<T, N>
    Hessian(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = resi_ad(i).dx(j);
      }
    }

    // Trust region subproblem. Exact algorithm, Nocedal 2nd Ed 4.3
    T
    lambda = 0.0;

    Intrepid::Tensor<T, N>
    K(dimension);

    Intrepid::Tensor<T, N>
    L(dimension);

    Intrepid::Vector<T, N>
    step;

    Intrepid::Vector<T, N>
    q;

    for (Intrepid::Index i{0}; i < this->max_num_restrict_iter_; ++i) {

      K = Hessian + lambda * I;

      L = Intrepid::cholesky(K).first;

      step = - Intrepid::solve(K, resi);

      q = Intrepid::solve(L, step);

      T const
      nps = Intrepid::norm_square(step);

      T const
      nqs = Intrepid::norm_square(q);

      T const
      lambda_incr = nps * (std::sqrt(nps) - step_length) / nqs / step_length;

      lambda += std::max(lambda_incr, 0.0);

    }

    Intrepid::Vector<AD, N> const
    soln_ad_next = soln_ad + step;

    Intrepid::Vector<AD, N> const
    resi_ad_next = nls.compute(soln_ad_next);

    // Compute reduction factor \rho_k in Nocedal's algorithm 11.5
    Intrepid::Vector<T, N>
    resi_next = Sacado::Value<Intrepid::Vector<AD, N>>::eval(resi_ad_next);

    T const
    nr = Intrepid::norm_square(resi);

    T const
    nrp = Intrepid::norm_square(resi_next);

    T const
    nrKp = Intrepid::norm_square(resi + Intrepid::dot(Hessian, step));

    T const
    reduction = (nr - nrp) / (nr - nrKp);

    // Determine whether the trust region should be increased, decreased
    // or left the same.
    T const
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
      soln_ad = soln_ad_next;
    }

    ++this->num_iter_;
  }

  soln = Sacado::Value<Intrepid::Vector<AD, N>>::eval(soln_ad);

  return;
}

//
//
//
template<typename NLS, typename T, Intrepid::Index N>
void
ConjugateGradientMethod<NLS, T, N>::
solve(NLS & nls, Intrepid::Vector<T, N> & soln)
{
  using AD = typename Sacado::Fad::DFad<T>;

  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<AD, N>
  soln_ad(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    soln_ad(i) = AD(dimension, i, soln(i));
  }

  Intrepid::Vector<AD, N>
  gradient_ad = nls.compute(soln_ad);

  Intrepid::Vector<AD, N>
  resi_ad = - gradient_ad;

  Intrepid::Tensor<T, N>
  Hessian(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    for (Intrepid::Index j{0}; j < dimension; ++j) {
      Hessian(i, j) = gradient_ad(i).dx(j);
    }
  }

  Intrepid::Vector<T, N>
  resi = Sacado::Value<Intrepid::Vector<AD, N>>::eval(resi_ad);

  Intrepid::Vector<T, N>
  precon_resi = Intrepid::solve(Hessian, resi);

  Intrepid::Vector<T, N>
  search_direction = precon_resi;

  T
  projection_new = Intrepid::dot(resi, search_direction);

  T const
  initial_norm = Intrepid::norm(resi);

  this->num_iter_ = 0;

  Intrepid::Index
  restart_directions_counter = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    gradient_ad = nls.compute(soln_ad);

    resi_ad = - gradient_ad;

    resi = Sacado::Value<Intrepid::Vector<AD, N>>::eval(resi_ad);

    this->abs_error_ = Intrepid::norm(resi);

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

    T
    projection_search = Intrepid::dot(search_direction, search_direction);

    T
    step_length = - this->initial_secant_step_length_;

    Intrepid::Vector<AD, N> const
    trial_soln_ad =
        soln_ad + this->initial_secant_step_length_ * search_direction;

    Intrepid::Vector<AD, N> const
    trial_gradient_ad = nls.compute(trial_soln_ad);

    Intrepid::Vector<T, N> const
    trial_gradient =
        Sacado::Value<Intrepid::Vector<AD, N>>::eval(trial_gradient_ad);

    T
    projection_prev = Intrepid::dot(trial_gradient, search_direction);

    // Secant line search.

    for (Intrepid::Index i{0}; i < this->max_num_secant_iter_; ++i) {

      gradient_ad = nls.compute(soln_ad);

      Intrepid::Vector<T, N> const
      gradient = Sacado::Value<Intrepid::Vector<AD, N>>::eval(gradient_ad);

      T const
      projection = Intrepid::dot(gradient, search_direction);

      step_length *= projection / (projection_prev - projection);

      soln_ad += step_length * search_direction;

      projection_prev = projection;

      bool const
      secant_converged = step_length * step_length * projection_search <=
        this->secant_tol_ * this->secant_tol_;

      if (secant_converged == true) break;

    }

    gradient_ad = nls.compute(soln_ad);

    resi_ad = - gradient_ad;

    resi = Sacado::Value<Intrepid::Vector<AD, N>>::eval(resi_ad);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = gradient_ad(i).dx(j);
      }
    }

    T const
    projection_old = projection_new;

    T const
    projection_mid = Intrepid::dot(resi, precon_resi);

    precon_resi = Intrepid::solve(Hessian, resi);

    projection_new = Intrepid::dot(resi, precon_resi);

    T const
    gram_schmidt_factor = (projection_new - projection_mid) / projection_old;

    ++restart_directions_counter;

    bool const
    restart_directions =
        restart_directions_counter == this->restart_directions_interval_ ||
        gram_schmidt_factor <= 0.0;

    if (restart_directions == true) {

      search_direction = precon_resi;
      restart_directions_counter = 0;

    } else {

      search_direction = precon_resi + gram_schmidt_factor * search_direction;

    }

    ++this->num_iter_;
  }

  soln = Sacado::Value<Intrepid::Vector<AD, N>>::eval(soln_ad);

  return;
}

} // namespace LCM
