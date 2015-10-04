//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// miniMinimizer
//
template<typename OPT, typename FN, Intrepid::Index N>
void
miniMinimize(OPT & optimizer, FN & function, Intrepid::Vector<RealType, N> & x)
{
  optimizer.solve(function, x);

  return;
}

template<typename OPT, typename FN, typename T, Intrepid::Index N>
void
miniMinimize(OPT & optimizer, FN & function, Intrepid::Vector<T, N> & soln)
{
  // Extract values and use them to minimize the function.
  using ValueT = typename Sacado::ValueType<T>::type;

  Intrepid::Vector<ValueT, N>
  soln_val = Sacado::Value<Intrepid::Vector<T, N>>::eval(soln);

  optimizer.solve(function, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  Intrepid::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  // Now compute gradient with solution that has Albany sensitivities.
  Intrepid::Vector<T, N>
  resi = function.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

//
// Specializations for MiniNonlinear solver
//

//
// Residual
//
template<typename NLS, Intrepid::Index N>
void
MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, N>::
solve(NLS & nls, Intrepid::Vector<ScalarT, N> & soln)
{
  this->nonlinear_method_.solve(nls, soln);
  return;
}

//
// Jacobian
//
template<typename NLS, Intrepid::Index N>
void
MiniNonlinearSolver<PHAL::AlbanyTraits::Jacobian, NLS, N>::
solve(NLS & nls, Intrepid::Vector<ScalarT, N> & soln)
{
  // Extract values and use them to solve the NLS.
  Intrepid::Vector<ValueT, N>
  soln_val = Sacado::Value<Intrepid::Vector<ScalarT, N>>::eval(soln);

  this->nonlinear_method_.solve(nls, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  Intrepid::Tensor<ValueT, N>
  DrDx = nls.hessian(soln_val);

  // Now evaluate nls with soln that has Albany sensitivities.
  Intrepid::Vector<ScalarT, N>
  resi = nls.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

//
// Tangent
//
template<typename NLS, Intrepid::Index N>
void
MiniNonlinearSolver<PHAL::AlbanyTraits::Tangent, NLS, N>::
solve(NLS & nls, Intrepid::Vector<ScalarT, N> & soln)
{
  // Extract values and use them to solve the NLS.
  Intrepid::Vector<ValueT, N>
  soln_val = Sacado::Value<Intrepid::Vector<ScalarT, N>>::eval(soln);

  this->nonlinear_method_.solve(nls, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  Intrepid::Tensor<ValueT, N>
  DrDx = nls.hessian(soln_val);

  // Now evaluate nls with soln that has Albany sensitivities.
  Intrepid::Vector<ScalarT, N>
  resi = nls.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

//
// DistParamDeriv
//
template<typename NLS, Intrepid::Index N>
void
MiniNonlinearSolver<PHAL::AlbanyTraits::DistParamDeriv, NLS, N>::
solve(NLS & nls, Intrepid::Vector<ScalarT, N> & soln)
{
  // Extract values and use them to solve the NLS.
  Intrepid::Vector<ValueT, N>
  soln_val = Sacado::Value<Intrepid::Vector<ScalarT, N>>::eval(soln);

  this->nonlinear_method_.solve(nls, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  Intrepid::Tensor<ValueT, N>
  DrDx = nls.hessian(soln_val);

  // Now evaluate nls with soln that has Albany sensitivities.
  Intrepid::Vector<ScalarT, N>
  resi = nls.gradient(soln);

  // Solve for solution sensitivities.
  computeFADInfo(resi, DrDx, soln);

  return;
}

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
