//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// Native MiniSolver
//
template<
    typename MIN, typename STEP, typename FN, typename EvalT, minitensor::Index N>
MiniSolver<MIN, STEP, FN, EvalT, N>::
MiniSolver(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    minitensor::Vector<typename EvalT::ScalarT, N> & soln)
{
  MT_ERROR_EXIT("Missing specialization for MiniSolver class.");
  return;
}

template<typename MIN, typename STEP, typename FN, minitensor::Index N>
MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::Residual, N>::
MiniSolver(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    minitensor::Vector<PHAL::AlbanyTraits::Residual::ScalarT, N> & soln)
{
  minimizer.solve(step_method, function, soln);
  return;
}

template<typename MIN, typename STEP, typename FN, minitensor::Index N>
MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::Jacobian, N>::
MiniSolver(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    minitensor::Vector<PHAL::AlbanyTraits::Jacobian::ScalarT, N> & soln)
{
  // Make sure that if Albany is compiled with a static FAD type
  // there won't be confusion with MiniSolver's FAD.
  using AD = minitensor::FAD<RealType, N>;

  using T = PHAL::AlbanyTraits::Jacobian::ScalarT;

  static_assert(
      std::is_same<T, AD>::value == false,
      "Albany and MiniSolver Fad types not allowed to be equal.");

  using ValueT = typename Sacado::ValueType<T>::type;

  minitensor::Vector<ValueT, N>
  soln_val = Sacado::Value<minitensor::Vector<T, N>>::eval(soln);

  minimizer.solve(step_method, function, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  minitensor::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  minitensor::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

template<typename MIN, typename STEP, typename FN, minitensor::Index N>
MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::Tangent, N>::
MiniSolver(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    minitensor::Vector<PHAL::AlbanyTraits::Tangent::ScalarT, N> & soln)
{
  // Make sure that if Albany is compiled with a static FAD type
  // there won't be confusion with MiniSolver's FAD.
  using AD = minitensor::FAD<RealType, N>;

  using T = PHAL::AlbanyTraits::Tangent::ScalarT;

  static_assert(
      std::is_same<T, AD>::value == false,
      "Albany and MiniSolver Fad types not allowed to be equal.");

  using ValueT = typename Sacado::ValueType<T>::type;

  minitensor::Vector<ValueT, N>
  soln_val = Sacado::Value<minitensor::Vector<T, N>>::eval(soln);

  minimizer.solve(step_method, function, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  minitensor::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  minitensor::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

template<typename MIN, typename STEP, typename FN, minitensor::Index N>
MiniSolver<MIN, STEP, FN, PHAL::AlbanyTraits::DistParamDeriv, N>::
MiniSolver(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    minitensor::Vector<PHAL::AlbanyTraits::DistParamDeriv::ScalarT, N> & soln)
{
  return;
}

//
// MiniSolver through ROL.
//
template<typename MIN, typename FN, typename EvalT, minitensor::Index N>
MiniSolverROL<MIN, FN, EvalT, N>::
MiniSolverROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    minitensor::Vector<typename EvalT::ScalarT, N> & soln)
{
  MT_ERROR_EXIT("Missing specialization for MiniSolver class.");
  return;
}

template<typename MIN, typename FN, minitensor::Index N>
MiniSolverROL<MIN, FN, PHAL::AlbanyTraits::Residual, N>::
MiniSolverROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    minitensor::Vector<typename PHAL::AlbanyTraits::Residual::ScalarT, N> & soln)
{
  minimizer.solve(algoname, params, function, soln);
  return;
}

template<typename MIN, typename FN, minitensor::Index N>
MiniSolverROL<MIN, FN, PHAL::AlbanyTraits::Jacobian, N>::
MiniSolverROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    minitensor::Vector<typename PHAL::AlbanyTraits::Jacobian::ScalarT, N> & soln)
{
  // Make sure that if Albany is compiled with a static FAD type
  // there won't be confusion with MiniSolver's FAD.
  using AD = minitensor::FAD<RealType, N>;

  using T = PHAL::AlbanyTraits::Jacobian::ScalarT;

  static_assert(
      std::is_same<T, AD>::value == false,
      "Albany and MiniSolver Fad types not allowed to be equal.");

  using ValueT = typename Sacado::ValueType<T>::type;

  minitensor::Vector<ValueT, N>
  soln_val = Sacado::Value<minitensor::Vector<T, N>>::eval(soln);

  minimizer.solve(algoname, params, function, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  minitensor::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  minitensor::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

template<typename MIN, typename FN, minitensor::Index N>
MiniSolverROL<MIN, FN, PHAL::AlbanyTraits::Tangent, N>::
MiniSolverROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    minitensor::Vector<typename PHAL::AlbanyTraits::Tangent::ScalarT, N> & soln)
{
  // Make sure that if Albany is compiled with a static FAD type
  // there won't be confusion with MiniSolver's FAD.
  using AD = minitensor::FAD<RealType, N>;

  using T = PHAL::AlbanyTraits::Jacobian::ScalarT;

  static_assert(
      std::is_same<T, AD>::value == false,
      "Albany and MiniSolver Fad types not allowed to be equal.");

  using ValueT = typename Sacado::ValueType<T>::type;

  minitensor::Vector<ValueT, N>
  soln_val = Sacado::Value<minitensor::Vector<T, N>>::eval(soln);

  minimizer.solve(algoname, params, function, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  minitensor::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  minitensor::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

template<typename MIN, typename FN, minitensor::Index N>
MiniSolverROL<MIN, FN, PHAL::AlbanyTraits::DistParamDeriv, N>::
MiniSolverROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    minitensor::Vector<typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT, N> & soln)
{
  return;
}

//
// MiniSolver through ROL with bound constraints.
//
template<
typename MIN, typename FN, typename BC, typename EvalT, minitensor::Index N>
MiniSolverBoundsROL<MIN, FN, BC, EvalT, N>::
MiniSolverBoundsROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    BC & bounds,
    minitensor::Vector<typename EvalT::ScalarT, N> & soln)
{
  MT_ERROR_EXIT("Missing specialization for MiniSolverBoundsROL class.");
  return;
}

template<typename MIN, typename FN, typename BC, minitensor::Index N>
MiniSolverBoundsROL<MIN, FN, BC, PHAL::AlbanyTraits::Residual, N>::
MiniSolverBoundsROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    BC & bounds,
    minitensor::Vector<PHAL::AlbanyTraits::Residual::ScalarT, N> & soln)
{
  minimizer.solve(algoname, params, function, bounds, soln);
  return;
}

template<typename MIN, typename FN, typename BC, minitensor::Index N>
MiniSolverBoundsROL<MIN, FN, BC, PHAL::AlbanyTraits::Jacobian, N>::
MiniSolverBoundsROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    BC & bounds,
    minitensor::Vector<PHAL::AlbanyTraits::Jacobian::ScalarT, N> & soln)
{
  // Make sure that if Albany is compiled with a static FAD type
  // there won't be confusion with MiniSolver's FAD.
  using AD = minitensor::FAD<RealType, N>;

  using T = PHAL::AlbanyTraits::Jacobian::ScalarT;

  static_assert(
      std::is_same<T, AD>::value == false,
      "Albany and MiniSolver Fad types not allowed to be equal.");

  using ValueT = typename Sacado::ValueType<T>::type;

  minitensor::Vector<ValueT, N>
  soln_val = Sacado::Value<minitensor::Vector<T, N>>::eval(soln);

  minimizer.solve(algoname, params, function, bounds, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  minitensor::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  minitensor::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

template<typename MIN, typename FN, typename BC, minitensor::Index N>
MiniSolverBoundsROL<MIN, FN, BC, PHAL::AlbanyTraits::Tangent, N>::
MiniSolverBoundsROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    BC & bounds,
    minitensor::Vector<PHAL::AlbanyTraits::Tangent::ScalarT, N> & soln)
{
  // Make sure that if Albany is compiled with a static FAD type
  // there won't be confusion with MiniSolver's FAD.
  using AD = minitensor::FAD<RealType, N>;

  using T = PHAL::AlbanyTraits::Tangent::ScalarT;

  static_assert(
      std::is_same<T, AD>::value == false,
      "Albany and MiniSolver Fad types not allowed to be equal.");

  using ValueT = typename Sacado::ValueType<T>::type;

  minitensor::Vector<ValueT, N>
  soln_val = Sacado::Value<minitensor::Vector<T, N>>::eval(soln);

  minimizer.solve(algoname, params, function, bounds, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  minitensor::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  minitensor::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

template<typename MIN, typename FN, typename BC, minitensor::Index N>
MiniSolverBoundsROL<MIN, FN, BC, PHAL::AlbanyTraits::DistParamDeriv, N>::
MiniSolverBoundsROL(
    MIN & minimizer,
    std::string const & algoname,
    Teuchos::ParameterList & params,
    FN & function,
    BC & bounds,
    minitensor::Vector<PHAL::AlbanyTraits::DistParamDeriv::ScalarT, N> & soln)
{
  return;
}

//
// MiniSolver through ROL with inequality constraints.
//
template<
typename MIN, typename FN, typename EIC, typename EvalT,
minitensor::Index N, minitensor::Index NC>
MiniSolverEqIneqROL<MIN, FN, EIC, EvalT, N, NC>::
MiniSolverEqIneqROL(
      MIN & minimizer,
      std::string const & algoname,
      Teuchos::ParameterList & params,
      FN & function,
      EIC & eqineq,
      minitensor::Vector<typename EvalT::ScalarT, N> & soln,
      minitensor::Vector<typename EvalT::ScalarT, NC> & cv)
{
  MT_ERROR_EXIT("Missing specialization for MiniSolverEqIneqROL class.");
}

template<
typename MIN, typename FN, typename EIC,
minitensor::Index N, minitensor::Index NC>
MiniSolverEqIneqROL<MIN, FN, EIC, PHAL::AlbanyTraits::Residual, N, NC>::
MiniSolverEqIneqROL(
      MIN & minimizer,
      std::string const & algoname,
      Teuchos::ParameterList & params,
      FN & function,
      EIC & eqineq,
      minitensor::Vector<typename PHAL::AlbanyTraits::Residual::ScalarT, N> & soln,
      minitensor::Vector<typename PHAL::AlbanyTraits::Residual::ScalarT, NC> & cv)
{
  minimizer.solve(algoname, params, function, eqineq, soln, cv);
  return;
}

template<
typename MIN, typename FN, typename EIC,
minitensor::Index N, minitensor::Index NC>
MiniSolverEqIneqROL<MIN, FN, EIC, PHAL::AlbanyTraits::Jacobian, N, NC>::
MiniSolverEqIneqROL(
      MIN & minimizer,
      std::string const & algoname,
      Teuchos::ParameterList & params,
      FN & function,
      EIC & eqineq,
      minitensor::Vector<typename PHAL::AlbanyTraits::Jacobian::ScalarT, N> & soln,
      minitensor::Vector<typename PHAL::AlbanyTraits::Jacobian::ScalarT, NC> & cv)
{
  // Make sure that if Albany is compiled with a static FAD type
  // there won't be confusion with MiniSolver's FAD.
  using AD = minitensor::FAD<RealType, N>;

  using T = PHAL::AlbanyTraits::Jacobian::ScalarT;

  static_assert(
      std::is_same<T, AD>::value == false,
      "Albany and MiniSolver Fad types not allowed to be equal.");

  using ValueT = typename Sacado::ValueType<T>::type;

  minitensor::Vector<ValueT, N>
  soln_val = Sacado::Value<minitensor::Vector<T, N>>::eval(soln);

  minitensor::Vector<ValueT, NC>
  cv_val = Sacado::Value<minitensor::Vector<T, NC>>::eval(cv);

  minimizer.solve(algoname, params, function, eqineq, soln_val, cv_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  minitensor::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  minitensor::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

//
//
//
template<typename T, typename S, minitensor::Index N>
void
computeFADInfo(
    minitensor::Vector<T, N> const & r,
    minitensor::Tensor<S, N> const & DrDx,
    minitensor::Vector<T, N> & x)
{
  // Check whether dealing with AD type.
  if (Sacado::IsADType<T>::value == false) return;

  // Deal with derivative information
  auto const
  dimension = r.get_dimension();

  if (0 == dimension) return;

  auto const
  order = r[0].size();

  // No FAD info. Nothing to do.
  if (order == 0) return;

  // Extract sensitivities of r wrt p
  minitensor::Matrix<S, N, minitensor::DYNAMIC>
  DrDp(dimension, order);

  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DrDp(i, j) = r(i).dx(j);
    }
  }

  // Solve for all DxDp
  minitensor::Matrix<S, N, minitensor::DYNAMIC>
  DxDp = minitensor::solve(DrDx, DrDp);

  // Pack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }
}

} // namespace LCM
