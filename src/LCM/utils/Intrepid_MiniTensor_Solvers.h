//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_MiniTensor_Solvers_h)
#define Intrepid_MiniTensor_Solvers_h

#include <utility>

#include <Intrepid_MiniTensor.h>

namespace Intrepid
{

///
/// Function base class that defines the interface to Mini Solvers.
///
template<typename Function_Derived, typename S>
class Function_Base
{
public:
  Function_Base()
  {
    //constexpr bool
    //is_fad = Sacado::IsADType<S>::value == true;

    //static_assert(is_fad == false, "AD types not allowed for type S");
  }

  ///
  /// By default use merit function 0.5 dot(gradient, gradient)
  /// as the target to optimize if only the gradient is provided.
  /// \param in to pass in values needed to compute f(x), df(x), ddf(x)
  ///
  template<typename T, Index N, Index IN = 0, Index OUT = 0>
  T
  value(
      Function_Derived & f,
      Vector<T, N> const & x,
      Vector<T, IN> const & in = Vector<T, IN>(),
      Vector<T, OUT> && out = Vector<T, OUT>());

  ///
  /// By default compute gradient with AD from value().
  ///
  template<typename T, Index N, Index IN = 0, Index OUT = 0>
  Vector<T, N>
  gradient(
      Function_Derived & f,
      Vector<T, N> const & x,
      Vector<T, IN> const & in = Vector<T, IN>(),
      Vector<T, OUT> && out = Vector<T, OUT>());

  ///
  /// By default compute Hessian with AD from gradient().
  ///
  template<typename T, Index N, Index IN = 0, Index OUT = 0>
  Tensor<T, N>
  hessian(
      Function_Derived & f,
      Vector<T, N> const & x,
      Vector<T, IN> const & in = Vector<T, IN>(),
      Vector<T, OUT> && out = Vector<T, OUT>());

};

///
/// Minimizer Struct
///
template<typename T, Index N>
struct Minimizer
{
public:
  Minimizer()
  {
    constexpr bool
    is_fad = Sacado::IsADType<T>::value == true;

    static_assert(is_fad == false, "AD types not allowed for type T");
  }

  template<typename STEP, typename FN>
  void
  solve(STEP & step_method, FN & fn, Vector<T, N> & x);

  void
  printReport(std::ostream & os);

private:
  void
  updateConvergenceCriterion(T const abs_error);

  bool
  continueSolve() const;

  template<typename FN>
  void
  recordFinals(FN & fn, Vector<T, N> const & x)
  {
    final_soln = x;
    final_value = fn.value(x);
    final_gradient = fn.gradient(x);
    final_hessian = fn.hessian(x);
  }

public:
  Index
  max_num_iter{256};

  T
  rel_tol{1.0e-12};

  T
  rel_error{1.0};

  T
  abs_tol{1.0e-12};

  T
  abs_error{1.0};

  bool
  converged{false};

private:
  T
  initial_norm{1.0};

  Index
  num_iter{0};

  Vector<T, N>
  initial_guess;

  Vector<T, N>
  final_soln;

  T
  final_value{0.0};

  Vector<T, N>
  final_gradient;

  Tensor<T, N>
  final_hessian;

  char const *
  step_method_name{nullptr};

  char const *
  function_name{nullptr};
};

///
/// Newton line search
///
template<typename T, Index N>
struct NewtonLineSearch
{
  template<typename FN>
  Vector<T, N>
  step(FN & fn, Vector<T, N> const & direction, Vector<T, N> const & soln);

  Index
  max_num_iter{16};

  T
  tolerance{1.0e-6};
};

///
/// Trust region subproblem. Exact algorithm, Nocedal 2nd Ed 4.3
///
template<typename T, Index N>
struct TrustRegionExact
{
  Vector<T, N>
  step(Tensor<T, N> const & Hessian, Vector<T, N> const & gradient);

  Index
  max_num_iter{4};

  T
  region_size{1.0};

  T
  initial_lambda{0.0};
};

///
/// Step Base
///
template<typename T>
struct Step_Base
{
  Step_Base()
  {
    constexpr bool
    is_fad = Sacado::IsADType<T>::value == true;

    static_assert(is_fad == false, "AD types not allowed for type T");
  }
};

///
/// Plain Newton Step
///
template<typename T, Index N>
struct NewtonStep : public Step_Base<T>
{
  static constexpr
  char const * const
  NAME = "Newton";

  template<typename FN>
  void
  initialize(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);

  template<typename FN>
  Vector<T, N>
  step(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);
};

///
/// Trust Region Step
///
template<typename T, Index N>
struct TrustRegionStep : public Step_Base<T>
{
  static constexpr
  char const * const
  NAME = "Trust Region";

  template<typename FN>
  void
  initialize(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);

  template<typename FN>
  Vector<T, N>
  step(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);

  T
  max_region_size{10.0};

  T
  initial_region_size{10.0};

  T
  min_reduction{0.0};

private:
  T
  region_size{0.0};
};

///
/// Conjugate Gradient Step
///
template<typename T, Index N>
struct ConjugateGradientStep : public Step_Base<T>
{
  static constexpr
  char const * const
  NAME = "Preconditioned Conjugate Gradient";

  template<typename FN>
  void
  initialize(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);

  template<typename FN>
  Vector<T, N>
  step(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);

  Index
  restart_directions_interval{32};

private:
  Vector<T, N>
  search_direction;

  Vector<T, N>
  precon_resi;

  T
  projection_new{0.0};

  Index
  restart_directions_counter{0};
};

///
/// Line Search Regularized Step
///
template<typename T, Index N>
struct LineSearchRegularizedStep : public Step_Base<T>
{
  static constexpr
  char const * const
  NAME = "Line Search Regularized";

  template<typename FN>
  void
  initialize(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);

  template<typename FN>
  Vector<T, N>
  step(FN & fn, Vector<T, N> const & x, Vector<T, N> const & r);

  T
  step_length{1.0};

  T
  hessian_cond_tol{1.0e+08};

  T
  hessian_singular_tol{1.0e-12};
};

} // namespace Intrepid

#include "Intrepid_MiniTensor_Solvers.t.h"

#endif // Intrepid_MiniTensor_Solvers_h
