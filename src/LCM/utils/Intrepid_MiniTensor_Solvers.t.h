//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace Intrepid
{

//
//
//
template<typename NLS, typename T, Index N>
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
template<typename T, typename S, Index N>
void
computeFADInfo(
    Vector<T, N> const & r,
    Tensor<S, N> const & DrDx,
    Vector<T, N> & x)
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
  Matrix<S>
  DrDp(dimension, order);

  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DrDp(i, j) = r(i).dx(j);
    }
  }

  // Solve for all DxDp
  Matrix<S>
  DxDp = solve(DrDx, DrDp);

  // Pack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

}

//
// Residual of nonlinear system
//
template<typename NLS, typename T, Index N>
Vector<T, N>
getValue(NLS const & nls, Vector<T, N> const & x)
{
  Vector<T, N>
  r = nls.evaluate(x);

  return r;
}

//
// Gradient of nonlinear system
//
template<typename NLS, typename T, Index N>
Tensor<T, N>
getGradient(NLS const & nls, Vector<T, N> const & x)
{
  using S = typename Sacado::ValueType<T>::type;
  using AD = typename Sacado::Fad::DFad<S>;

  Index const
  dimension = x.get_dimension();

  Vector<S, N>
  x_val = Sacado::Value<Vector<T, N>>::eval(x);

  Vector<AD, N>
  x_ad(dimension);

  for (Index i{0}; i < dimension; ++i) {
    x_ad(i) = AD(dimension, i, x_val(i));
  }

  Vector<AD, N>
  r_ad = nls.evaluate(x_ad);

  Tensor<T, N>
  Hessian(dimension);

  for (Index i{0}; i < dimension; ++i) {
    for (Index j{0}; j < dimension; ++j) {
      Hessian(i, j) = r_ad(i).dx(j);
    }
  }

  return Hessian;
}

//
//
//
template<typename NLS, typename T, Index N>
void
NewtonMethod<NLS, T, N>::
solve(NLS const & nls, Vector<T, N> & soln)
{
  this->setInitialGuess(soln);

  Index const
  dimension = soln.get_dimension();

  Tensor<T, N>
  Hessian(dimension);

  Vector<T, N>
  step(dimension);

  Vector<T, N>
  resi = nls.evaluate(soln);

  T const
  initial_norm = norm(resi);

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  while (this->continueSolve() == true) {

    Hessian = getGradient(nls, soln);

    step = - Intrepid::solve(Hessian, resi);

    soln += step;

    resi = nls.evaluate(soln);

    T const
    norm_resi = norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  this->setFinalSolution(soln);
  return;
}

//
// Trust Region method.  See Nocedal's algorithm 11.5.
//
template<typename NLS, typename T, Index N>
void
TrustRegionMethod<NLS, T, N>::
solve(NLS const & nls, Vector<T, N> & soln)
{
  this->setInitialGuess(soln);

  Index const
  dimension = soln.get_dimension();

  Tensor<T, N>
  Hessian(dimension);

  Tensor<T, N>
  K(dimension);

  Tensor<T, N>
  L(dimension);

  Vector<T, N>
  step(dimension);

  Vector<T, N>
  q(dimension);

  Vector<T, N>
  soln_next(dimension);

  Vector<T, N>
  resi_next(dimension);

  Tensor<T, N> const
  I = identity<T, N>(dimension);

  Vector<T, N>
  resi = nls.evaluate(soln);

  T const
  initial_norm = norm(resi);

  T
  region_size = getInitialRegionSize();

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  // Outer solution loop
  while (this->continueSolve() == true) {

    Hessian = getGradient(nls, soln);

    // Trust region subproblem. Exact algorithm, Nocedal 2nd Ed 4.3
    T
    lambda = 0.0;

    for (Index i{0}; i < getMaxNumRestrictIterations(); ++i) {

      K = Hessian + lambda * I;

      L = cholesky(K).first;

      step = - Intrepid::solve(K, resi);

      q = Intrepid::solve(L, step);

      T const
      np = norm(step);

      T const
      nps = np * np;

      T const
      nqs = norm_square(q);

      T const
      lambda_incr = nps * (np - region_size) / nqs / region_size;

      lambda += std::max(lambda_incr, 0.0);

    }

    soln_next = soln + step;

    resi_next = nls.evaluate(soln_next);

    // Compute reduction factor \rho_k in Nocedal's algorithm 11.5
    T const
    nr = norm_square(resi);

    T const
    nrp = norm_square(resi_next);

    T const
    nrKp = norm_square(resi + dot(Hessian, step));

    T const
    reduction = (nr - nrp) / (nr - nrKp);

    // Determine whether the trust region should be increased, decreased
    // or left the same.
    T const
    computed_size = norm(step);

    if (reduction < 0.25) {

      region_size = 0.25 * computed_size;

    } else {

      bool const
      at_boundary = std::abs(computed_size / region_size - 1.0) <= 1.0e-8;

      bool const
      increase_region_size = reduction > 0.75 && at_boundary;

      if (increase_region_size == true) {
        region_size = std::min(2.0 * region_size, getMaxRegionSize());
      }

    }

    if (reduction > getMinumumReduction()) {
      soln = soln_next;
      resi = nls.evaluate(soln);
    }

    T const
    norm_resi = norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  this->setFinalSolution(soln);
  return;
}

//
//
//
template<typename NLS, typename T, Index N>
void
ConjugateGradientMethod<NLS, T, N>::
solve(NLS const & nls, Vector<T, N> & soln)
{
  this->setInitialGuess(soln);

  Index const
  dimension = soln.get_dimension();

  Vector<T, N>
  gradient = nls.evaluate(soln);

  Vector<T, N>
  resi = - gradient;

  Tensor<T, N>
  Hessian = getGradient(nls, soln);

  Vector<T, N>
  precon_resi = Intrepid::solve(Hessian, resi);

  Vector<T, N>
  search_direction = precon_resi;

  Vector<T, N>
  trial_soln(dimension);

  Vector<T, N>
  trial_gradient(dimension);

  T
  projection_new = dot(resi, search_direction);

  T const
  initial_norm = norm(resi);

  Index
  restart_directions_counter = 0;

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  while (this->continueSolve() == true) {

    // Newton line search.

    T const
    projection_search = dot(search_direction, search_direction);

    for (Index i{0}; i < getMaxNumLineSearchIterations(); ++i) {

      gradient = nls.evaluate(soln);

      Hessian = getGradient(nls, soln);

      T const
      projection = dot(gradient, search_direction);

      T const
      contraction =
          dot(search_direction, dot(Hessian, search_direction));

      T const
      step_length = - projection / contraction;

      soln += step_length * search_direction;

      bool const
      line_search_converged = step_length * step_length * projection_search <=
      getLineSearchTolerance() * getLineSearchTolerance();

      if (line_search_converged == true) break;

    }

    gradient = nls.evaluate(soln);

    resi = - gradient;

    T const
    projection_old = projection_new;

    T const
    projection_mid = dot(resi, precon_resi);

    Hessian = getGradient(nls, soln);

    precon_resi = Intrepid::solve(Hessian, resi);

    projection_new = dot(resi, precon_resi);

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
    norm_resi = norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  this->setFinalSolution(soln);
  return;
}

//
//
//
template<typename NLS, typename T, Index N>
void
LineSearchRegularizedMethod<NLS, T, N>::
solve(NLS const & nls, Vector<T, N> & soln)
{
  this->setInitialGuess(soln);

  Index const
  dimension = soln.get_dimension();

  Tensor<T, N>
  Hessian(dimension);

  Vector<T, N>
  resi = nls.evaluate(soln);

  Tensor<T, N>
  K(dimension);

  Tensor<T, N>
  L(dimension);

  Vector<T, N>
  step(dimension);

  Vector<T, N>
  q(dimension);

  Tensor<T, N> const
  I = identity<T, N>(dimension);

  T const
  initial_norm = norm(resi);

  T const
  step_length = getInitialStepLength();

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  while (this->continueSolve() == true) {

    Hessian = getGradient(nls, soln);

    bool const
    ill_conditioned = cond(Hessian) <= getHessianConditionTolerance();

    if (ill_conditioned == true) {

      // Trust region subproblem. Exact algorithm, Nocedal 2nd Ed 4.3
      T
      lambda = 0.0;

      for (Index i{0}; i < getMaxNumRestrictIterations(); ++i) {

        K = Hessian + lambda * I;

        L = cholesky(K).first;

        step = - Intrepid::solve(K, resi);

        q = Intrepid::solve(L, step);

        T const
        nps = norm_square(step);

        T const
        nqs = norm_square(q);

        T const
        lambda_incr = nps * (std::sqrt(nps) - step_length) / nqs / step_length;

        lambda += std::max(lambda_incr, 0.0);

      }

    } else {

      // Standard Newton step
      step = - Intrepid::solve(Hessian, resi);

    }

    // Newton line search.

    T const
    projection_step = dot(step, step);

    for (Index i{0}; i < getMaxNumLineSearchIterations(); ++i) {

      resi = nls.evaluate(soln);

      Hessian = getGradient(nls, soln);

      T const
      projection = dot(resi, step);

      T const
      contraction = dot(step, dot(Hessian, step));

      T const
      ls_length = - projection / contraction;

      soln += ls_length * step;

      bool const
      line_search_converged = ls_length * ls_length * projection_step <=
      getLineSearchTolerance() * getLineSearchTolerance();

      if (line_search_converged == true) break;

    }

    resi = nls.evaluate(soln);

    T const
    norm_resi = norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  this->setFinalSolution(soln);
  return;
}

} // namespace Intrepid
