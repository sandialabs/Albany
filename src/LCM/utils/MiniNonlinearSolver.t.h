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
template<typename MIN, typename STEP, typename FN, Intrepid::Index N>
void
miniMinimize(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    Intrepid::Vector<PHAL::AlbanyTraits::Residual::ScalarT, N> & soln)
{
  minimizer.solve(step_method, function, soln);

  return;
}

template<typename MIN, typename STEP, typename FN, typename T, Intrepid::Index N>
void
miniMinimize(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    Intrepid::Vector<T, N> & soln)
{
  minimizer.solve(step_method, function, soln);

  using ValueT = typename Sacado::ValueType<T>::type;

  Intrepid::Vector<ValueT, N>
  soln_val = Sacado::Value<Intrepid::Vector<T, N>>::eval(soln);

//  auto const
//  dimension = soln.get_dimension();

  // Put values back in solution vector
//  for (auto i = 0; i < dimension; ++i) {
//    soln(i).val() = soln_val(i);
//    soln(i) = T(dimension, i, soln_val(i));
//  }

  std::cout << "soln: " << soln << '\n';
  std::cout << "_val: " << soln_val << '\n';

  // Check if there is FAD info.
  //auto const
  //order = soln[0].size();

  //if (order == 0) return;

  // Get the Hessian evaluated at the solution.
  Intrepid::Tensor<ValueT, N>
  DrDx = function.hessian(soln_val);

  std::cout << "DrDx: " << DrDx << '\n';

  // Now compute gradient with solution that has Albany sensitivities.
  Intrepid::Vector<T, N>
  resi = function.gradient(soln);

  std::cout << "resi: " << resi << '\n';

  // Solve for solution sensitivities.
  //computeFADInfo(resi, DrDx, soln);

  return;
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

  std::cout << "FAD Info dimension : " << dimension << '\n';
  std::cout << "FAD Info order     : " << order << '\n';
  std::cout << "Soln before        : " << x << '\n';

  // Extract sensitivities of r wrt p
  Intrepid::Matrix<S, N>
  DrDp(dimension, order);

  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DrDp(i, j) = r(i).dx(j);
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<S, N>
  DxDp = Intrepid::solve(DrDx, DrDp);

  // Pack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

  std::cout << "Soln after         : " << x << '\n';
}

} // namespace LCM
