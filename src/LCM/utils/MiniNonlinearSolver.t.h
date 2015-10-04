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
    Intrepid::Vector<RealType, N> & soln)
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
  // Extract values and use them to minimize the function.
  using ValueT = typename Sacado::ValueType<T>::type;

  Intrepid::Vector<ValueT, N>
  soln_val = Sacado::Value<Intrepid::Vector<T, N>>::eval(soln);

  minimizer.solve(step_method, function, soln_val);

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

} // namespace LCM
