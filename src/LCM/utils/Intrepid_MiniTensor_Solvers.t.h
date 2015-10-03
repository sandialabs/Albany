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
template<typename STEP, typename T, Index N>
void
Minimizer<STEP, T, N>::
printReport(std::ostream & os)
{
  std::string const
  cs = converged == true ? "YES" : "NO";

  //std::string const
  //cs = converged == true ? "\U0001F60A" : "\U0001F623";

  os << "\n\n";
  os << "Method     : " << STEP::NAME << '\n';
  os << "System     : " << function_name << '\n';
  os << "Converged  : " << cs << '\n';
  os << "Max Iters  : " << max_num_iter << '\n';
  os << "Iters Taken: " << num_iter << '\n';

  os << std::scientific << std::setprecision(16);

  os << "Initial |R|: " << std::setw(24) << initial_norm << '\n';
  os << "Abs Tol    : " << std::setw(24) << abs_tol << '\n';
  os << "Abs Error  : " << std::setw(24) << abs_error << '\n';
  os << "Rel Tol    : " << std::setw(24) << rel_tol << '\n';
  os << "Rel Error  : " << std::setw(24) << rel_error << '\n';
  os << "Initial X  : " << initial_guess << '\n';
  os << "Final X    : " << final_soln << '\n';
  os << "f(X)       : " << std::setw(24) << final_value << '\n';
  os << "Df(X)      : " << final_gradient << '\n';
  os << "DDf(X)     : " << final_hessian << '\n';
  os << '\n';

  return;
}

//
//
//
template<typename STEP, typename T, Index N>
void
Minimizer<STEP, T, N>::
updateConvergenceCriterion(T const ae)
{
  abs_error = ae;
  rel_error = initial_norm > 0.0 ? abs_error / initial_norm : 0.0;

  bool const
  converged_absolute = abs_error <= abs_tol;

  bool const
  converged_relative = rel_error <= rel_tol;

  converged = converged_absolute || converged_relative;

  return;
}

//
//
//
template<typename STEP, typename T, Index N>
bool
Minimizer<STEP, T, N>::
continueSolve() const
{
  bool const
  is_max_iter = num_iter >= max_num_iter;

  bool const
  end_solve = is_max_iter == true || converged == true;

  bool const
  continue_solve = end_solve == false;

  return continue_solve;
}

//
//
//
template<typename STEP, typename T, Index N>
template<typename FN>
void
Minimizer<STEP, T, N>::
solve(FN & fn, Vector<T, N> & soln)
{
  function_name = FN::NAME;
  initial_guess = soln;

  Vector<T, N>
  resi = fn.gradient(soln);

  initial_norm = norm(resi);

  updateConvergenceCriterion(initial_norm);

  step_method.initialize(fn, soln, resi);

  while (continueSolve() == true) {

    Vector<T, N> const
    step = step_method.step(fn, soln, resi);

    soln += step;

    resi = fn.gradient(soln);

    T const
    norm_resi = norm(resi);

    updateConvergenceCriterion(norm_resi);
    ++num_iter;
  }

  recordFinals(fn, soln);
  return;
}

//
//
//
template<typename T, Index N>
template<typename FN>
void
NewtonStep<T, N>::
initialize(FN &, Vector<T, N> const &, Vector<T, N> const &)
{
  return;
}

//
// Plain Newton step.
//
template<typename T, Index N>
template<typename FN>
Vector<T, N>
NewtonStep<T, N>::
step(FN & fn, Vector<T, N> const & soln, Vector<T, N> const & resi)
{
  Tensor<T, N> const
  Hessian = fn.hessian(soln);

  Vector<T, N> const
  step = - Intrepid::solve(Hessian, resi);

  return step;
}

//
//
//
template<typename T, Index N>
template<typename FN>
void
TrustRegionStep<T, N>::
initialize(FN &, Vector<T, N> const &, Vector<T, N> const &)
{
  region_size = initial_region_size;

  return;
}

//
// Trust Region method.  See Nocedal's algorithm 11.5.
//
template<typename T, Index N>
template<typename FN>
Vector<T, N>
TrustRegionStep<T, N>::
step(FN & fn, Vector<T, N> const & soln, Vector<T, N> const & resi)
{
  Index const
  dimension = soln.get_dimension();

  Tensor<T, N> const
  I = identity<T, N>(dimension);

  Tensor<T, N> const
  Hessian = fn.hessian(soln);

  Vector<T, N>
  step(dimension);

  // Trust region subproblem. Exact algorithm, Nocedal 2nd Ed 4.3
  T
  lambda = 0.0;

  for (Index i{0}; i < max_num_restrict_iter; ++i) {

    Tensor<T, N> const
    K = Hessian + lambda * I;

    Tensor<T, N> const
    L = cholesky(K).first;

    step = - Intrepid::solve(K, resi);

    Vector<T, N>
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

  Vector<T, N>
  soln_next = soln + step;

  Vector<T, N>
  resi_next = fn.gradient(soln_next);

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
      region_size = std::min(2.0 * region_size, max_region_size);
    }

  }

  if (reduction <= min_reduction) {
    step.fill(ZEROS);
  }

  return step;
}

//
//
//
template<typename T, Index N>
template<typename FN>
void
ConjugateGradientStep<T, N>::
initialize(FN & fn, Vector<T, N> const & soln, Vector<T, N> const & gradient)
{
  Tensor<T, N> const
  Hessian = fn.hessian(soln);

  precon_resi = Intrepid::solve(Hessian, -gradient);

  search_direction = precon_resi;

  projection_new = - dot(gradient, search_direction);

  restart_directions_counter = 0;

  return;
}

//
// Conjugate Gradient Method class.
// For now the Gram-Schmidt method is fixed to Polak-Ribiere
// and preconditioning with the Hessian.
// This is taken from J.R. Shewchuck "painless" conjugate gradient
// manuscript that is all over the place on the net.
//
template<typename T, Index N>
template<typename FN>
Vector<T, N>
ConjugateGradientStep<T, N>::
step(FN & fn, Vector<T, N> const & soln, Vector<T, N> const &)
{
  Index const
  dimension = soln.get_dimension();

  Vector<T, N>
  step(dimension, ZEROS);

  Vector<T, N>
  soln_next(dimension, ZEROS);

  Vector<T, N>
  gradient_next(dimension, ZEROS);

  Tensor<T, N>
  Hessian(dimension, ZEROS);

  T const
  projection_search = dot(search_direction, search_direction);

  // Newton line search.
  for (Index i{0}; i < max_num_line_search_iter; ++i) {

    soln_next = soln + step;

    gradient_next = fn.gradient(soln_next);

    Hessian = fn.hessian(soln_next);

    T const
    projection = dot(gradient_next, search_direction);

    T const
    contraction = dot(search_direction, dot(Hessian, search_direction));

    T const
    step_length = - projection / contraction;

    step += step_length * search_direction;

    bool const
    line_search_converged = step_length * step_length * projection_search <=
    line_search_tol * line_search_tol;

    if (line_search_converged == true) break;

  }

  soln_next = soln + step;

  gradient_next = fn.gradient(soln_next);

  T const
  projection_old = projection_new;

  T const
  projection_mid = - dot(gradient_next, precon_resi);

  Hessian = fn.hessian(soln_next);

  precon_resi = Intrepid::solve(Hessian, -gradient_next);

  projection_new = - dot(gradient_next, precon_resi);

  T const
  gram_schmidt_factor = (projection_new - projection_mid) / projection_old;

  ++restart_directions_counter;

  bool const
  restart_directions =
      restart_directions_counter == restart_directions_interval ||
      gram_schmidt_factor <= 0.0;

  if (restart_directions == true) {

    search_direction = precon_resi;
    restart_directions_counter = 0;

  } else {

    search_direction = precon_resi + gram_schmidt_factor * search_direction;

  }

  return step;
}

//
//
//
template<typename T, Index N>
template<typename FN>
void
LineSearchRegularizedStep<T, N>::
initialize(FN &, Vector<T, N> const &, Vector<T, N> const &)
{
  return;
}

//
// Trust Region method.  See Nocedal's algorithm 11.5.
//
template<typename T, Index N>
template<typename FN>
Vector<T, N>
LineSearchRegularizedStep<T, N>::
step(FN & fn, Vector<T, N> const & soln, Vector<T, N> const & gradient)
{
  Index const
  dimension = soln.get_dimension();

  Tensor<T, N> const
  I = identity<T, N>(dimension);

  Tensor<T, N>
  Hessian = fn.hessian(soln);

  Vector<T, N>
  step(dimension);

  bool const
  ill_conditioned =
      det(Hessian) < hessian_singular_tol || cond(Hessian) > hessian_cond_tol;

  if (ill_conditioned == true) {

    // Trust region subproblem. Exact algorithm, Nocedal 2nd Ed 4.3
    T
    lambda = 0.0;

    for (Index i{0}; i < max_num_restrict_iter; ++i) {

      Tensor<T, N>
      K = Hessian + lambda * I;

      Tensor<T, N>
      L = cholesky(K).first;

      step = - Intrepid::solve(K, gradient);

      Vector<T, N>
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
    step = - Intrepid::solve(Hessian, gradient);

  }

  // Newton line search.

  T const
  projection_step = dot(step, step);

  Vector<T, N>
  ls_step(dimension, ZEROS);

  for (Index i{0}; i < max_num_line_search_iter; ++i) {

    Vector<T, N>
    soln_next = soln + ls_step;

    Vector<T, N> const
    gradient_next = fn.gradient(soln_next);

    Hessian = fn.hessian(soln_next);

    T const
    projection = dot(gradient_next, step);

    T const
    contraction = dot(step, dot(Hessian, step));

    T const
    ls_length = - projection / contraction;

    ls_step += ls_length * step;

    bool const
    line_search_converged = ls_length * ls_length * projection_step <=
    line_search_tol * line_search_tol;

    if (line_search_converged == true) break;

  }

  return ls_step;
}
//
//
//
template<typename NLS, typename T, Index N>
void
NewtonMethod<NLS, T, N>::
solve(NLS & nls, Vector<T, N> & soln)
{
  this->setInitialGuess(soln);

  Index const
  dimension = soln.get_dimension();

  Tensor<T, N>
  Hessian(dimension);

  Vector<T, N>
  step(dimension);

  Vector<T, N>
  resi = nls.gradient(soln);

  T const
  initial_norm = norm(resi);

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  while (this->continueSolve() == true) {

    Hessian = nls.hessian(soln);

    step = - Intrepid::solve(Hessian, resi);

    soln += step;

    resi = nls.gradient(soln);

    T const
    norm_resi = norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  this->recordFinals(nls, soln);
  return;
}

//
// Trust Region method.  See Nocedal's algorithm 11.5.
//
template<typename NLS, typename T, Index N>
void
TrustRegionMethod<NLS, T, N>::
solve(NLS & nls, Vector<T, N> & soln)
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
  resi = nls.gradient(soln);

  T const
  initial_norm = norm(resi);

  T
  region_size = getInitialRegionSize();

  this->initConvergenceCriterion(initial_norm);
  this->updateConvergenceCriterion(initial_norm);

  // Outer solution loop
  while (this->continueSolve() == true) {

    Hessian = nls.hessian(soln);

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

    resi_next = nls.gradient(soln_next);

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
      resi = nls.gradient(soln);
    }

    T const
    norm_resi = norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  this->recordFinals(nls, soln);
  return;
}

//
//
//
template<typename NLS, typename T, Index N>
void
ConjugateGradientMethod<NLS, T, N>::
solve(NLS & nls, Vector<T, N> & soln)
{
  this->setInitialGuess(soln);

  Index const
  dimension = soln.get_dimension();

  Vector<T, N>
  gradient = nls.gradient(soln);

  Vector<T, N>
  resi = - gradient;

  Tensor<T, N>
  Hessian = nls.hessian(soln);

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

      gradient = nls.gradient(soln);

      Hessian = nls.hessian(soln);

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

    gradient = nls.gradient(soln);

    resi = - gradient;

    T const
    projection_old = projection_new;

    T const
    projection_mid = dot(resi, precon_resi);

    Hessian = nls.hessian(soln);

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

  this->recordFinals(nls, soln);
  return;
}

//
//
//
template<typename NLS, typename T, Index N>
void
LineSearchRegularizedMethod<NLS, T, N>::
solve(NLS & nls, Vector<T, N> & soln)
{
  this->setInitialGuess(soln);

  Index const
  dimension = soln.get_dimension();

  Tensor<T, N>
  Hessian(dimension);

  Vector<T, N>
  resi = nls.gradient(soln);

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

    Hessian = nls.hessian(soln);

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

      resi = nls.gradient(soln);

      Hessian = nls.hessian(soln);

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

    resi = nls.gradient(soln);

    T const
    norm_resi = norm(resi);

    this->updateConvergenceCriterion(norm_resi);
    this->increaseIterationCounter();
  }

  this->recordFinals(nls, soln);
  return;
}

} // namespace Intrepid
