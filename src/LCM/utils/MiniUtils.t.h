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

  case NonlinearMethod::LINE_SEARCH_REGULARIZED:
    method = new LineSearchRegularizedMethod<NLS, T, N>();
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
// Hessian of nonlinear system
//
template<typename NLS, typename T, Intrepid::Index N>
Intrepid::Tensor<typename Sacado::ValueType<T>::type, N>
computeHessian(NLS const & nls, Intrepid::Vector<T, N> const & x)
{
  using S = typename Sacado::ValueType<T>::type;
  using AD = typename Sacado::Fad::DFad<S>;

  Intrepid::Index const
  dimension = x.get_dimension();

  Intrepid::Vector<S, N>
  x_val = Sacado::Value<Intrepid::Vector<T, N>>::eval(x);

  Intrepid::Vector<AD, N>
  x_ad(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    x_ad(i) = AD(dimension, i, x_val(i));
  }

  Intrepid::Vector<AD, N>
  r_ad = nls.compute(x_ad);

  Intrepid::Tensor<S, N>
  Hessian(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    for (Intrepid::Index j{0}; j < dimension; ++j) {
      Hessian(i, j) = r_ad(i).dx(j);
    }
  }

  return Hessian;
}

//
//
//
template<typename NLS, typename T, Intrepid::Index N>
void
NewtonMethod<NLS, T, N>::
solve(NLS const & nls, Intrepid::Vector<T, N> & soln)
{
  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Tensor<T, N>
  Hessian(dimension);

  Intrepid::Vector<T, N>
  soln_incr(dimension);

  Intrepid::Vector<T, N>
  resi = nls.compute(soln);

  T const
  initial_norm = Intrepid::norm(resi);

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  while (this->continueSolve() == true) {

    Hessian = computeHessian(nls, soln);

    soln_incr = - Intrepid::solve(Hessian, resi);

    soln += soln_incr;

    resi = nls.compute(soln);

    T const
    norm_resi = Intrepid::norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  return;
}

//
// Trust Region method.  See Nocedal's algorithm 11.5.
//
template<typename NLS, typename T, Intrepid::Index N>
void
TrustRegionMethod<NLS, T, N>::
solve(NLS const & nls, Intrepid::Vector<T, N> & soln)
{
  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Tensor<T, N>
  Hessian(dimension);

  Intrepid::Tensor<T, N>
  K(dimension);

  Intrepid::Tensor<T, N>
  L(dimension);

  Intrepid::Vector<T, N>
  step(dimension);

  Intrepid::Vector<T, N>
  q(dimension);

  Intrepid::Vector<T, N>
  soln_next(dimension);

  Intrepid::Vector<T, N>
  resi_next(dimension);

  Intrepid::Tensor<T, N> const
  I = Intrepid::identity<T, N>(dimension);

  Intrepid::Vector<T, N>
  resi = nls.compute(soln);

  T const
  initial_norm = Intrepid::norm(resi);

  T
  step_length = getInitialStepLength();

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  // Outer solution loop
  while (this->continueSolve() == true) {

    Hessian = computeHessian(nls, soln);

    // Trust region subproblem. Exact algorithm, Nocedal 2nd Ed 4.3
    T
    lambda = 0.0;

    for (Intrepid::Index i{0}; i < getMaximumNumberRestrictIterations(); ++i) {

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

    soln_next = soln + step;

    resi_next = nls.compute(soln_next);

    // Compute reduction factor \rho_k in Nocedal's algorithm 11.5
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
        step_length = std::min(2.0 * step_length, getMaximumStepLength());
      }

    }

    if (reduction > getMinumumReduction()) {
      soln = soln_next;
      resi = nls.compute(soln);
    }

    T const
    norm_resi = Intrepid::norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  return;
}

//
//
//
template<typename NLS, typename T, Intrepid::Index N>
void
ConjugateGradientMethod<NLS, T, N>::
solve(NLS const & nls, Intrepid::Vector<T, N> & soln)
{
  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<T, N>
  gradient = nls.compute(soln);

  Intrepid::Vector<T, N>
  resi = - gradient;

  Intrepid::Tensor<T, N>
  Hessian = computeHessian(nls, soln);

  Intrepid::Vector<T, N>
  precon_resi = Intrepid::solve(Hessian, resi);

  Intrepid::Vector<T, N>
  search_direction = precon_resi;

  Intrepid::Vector<T, N>
  trial_soln(dimension);

  Intrepid::Vector<T, N>
  trial_gradient(dimension);

  T
  projection_new = Intrepid::dot(resi, search_direction);

  T const
  initial_norm = Intrepid::norm(resi);

  Intrepid::Index
  restart_directions_counter = 0;

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  while (this->continueSolve() == true) {

    T
    projection_search = Intrepid::dot(search_direction, search_direction);

    // Newton line search.

    for (Intrepid::Index i{0}; i < getMaximumNumberLineSearchIterations(); ++i) {

      gradient = nls.compute(soln);

      Hessian = computeHessian(nls, soln);

      T const
      projection = Intrepid::dot(gradient, search_direction);

      T const
      contraction =
          Intrepid::dot(search_direction, Intrepid::dot(Hessian, search_direction));

      T const
      step_length = - projection / contraction;

      soln += step_length * search_direction;

      bool const
      secant_converged = step_length * step_length * projection_search <=
      getLineSearchTolerance() * getLineSearchTolerance();

      if (secant_converged == true) break;

    }

    gradient = nls.compute(soln);

    resi = - gradient;

    T const
    projection_old = projection_new;

    T const
    projection_mid = Intrepid::dot(resi, precon_resi);

    Hessian = computeHessian(nls, soln);

    precon_resi = Intrepid::solve(Hessian, resi);

    projection_new = Intrepid::dot(resi, precon_resi);

    T const
    gram_schmidt_factor = (projection_new - projection_mid) / projection_old;

    ++restart_directions_counter;

    bool const
    restart_directions =
        restart_directions_counter == getRestartDirectionsInterval() ||
        gram_schmidt_factor <= 0.0;

    if (restart_directions == true) {

      search_direction = precon_resi;
      restart_directions_counter = 0;

    } else {

      search_direction = precon_resi + gram_schmidt_factor * search_direction;

    }

    T const
    norm_resi = Intrepid::norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  return;
}

//
//
//
template<typename NLS, typename T, Intrepid::Index N>
void
LineSearchRegularizedMethod<NLS, T, N>::
solve(NLS const & nls, Intrepid::Vector<T, N> & soln)
{
  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Tensor<T, N>
  Hessian(dimension);

  Intrepid::Vector<T, N>
  soln_incr(dimension);

  Intrepid::Vector<T, N>
  resi = nls.compute(soln);

  T const
  initial_norm = Intrepid::norm(resi);

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  while (this->continueSolve() == true) {

    Hessian = computeHessian(nls, soln);

    soln_incr = - Intrepid::solve(Hessian, resi);

    soln += soln_incr;

    resi = nls.compute(soln);

    T const
    norm_resi = Intrepid::norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  return;
}

} // namespace LCM
