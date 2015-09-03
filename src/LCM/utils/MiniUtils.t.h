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
template<typename Residual, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
std::unique_ptr<NonlinearMethod_Base<Residual, T, N>>
nonlinearMethodFactory(NonlinearMethod const method_type)
{
  std::unique_ptr<NonlinearMethod_Base<Residual, T, N>>
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
    method = new NewtonMethod<Residual, T, N>();
    break;

  case NonlinearMethod::TRUST_REGION:
    method = new TrustRegionMethod<Residual, T, N>();
    break;

  case NonlinearMethod::CONJUGATE_GRADIENT:
    method = new ConjugateGradientMethod<Residual, T, N>();
    break;

  }

  return method;
}

//
//
//
template<typename Residual, typename T, Intrepid::Index N>
void
NewtonMethod<Residual, T, N>::
solve(Residual & residual, Intrepid::Vector<T, N> & soln)
{
  using AD = typename Sacado::Fad::DFad<T>;

  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<T, N>
  resi = residual.compute(soln);

  Intrepid::Vector<AD, N>
  soln_ad(dimension), resi_ad(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    soln_ad(i) = AD(dimension, i, soln(i));
  }

  T const
  initial_norm = Intrepid::norm(resi);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    resi_ad = residual.compute(soln_ad);

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
//
//
template<typename Residual, typename T, Intrepid::Index N>
void
TrustRegionMethod<Residual, T, N>::
solve(Residual & residual, Intrepid::Vector<T, N> & soln)
{
  using AD = typename Sacado::Fad::DFad<T>;

  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<T, N>
  resi = residual.compute(soln);

  Intrepid::Vector<AD, N>
  soln_ad(dimension), resi_ad(dimension);

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

    resi_ad = residual.compute(soln_ad);

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

    // Restrict step to size of trust region. Exact algorithm, Nocedal 4.4
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
    resi_ad_next = residual.compute(soln_ad_next);

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

} // namespace LCM
