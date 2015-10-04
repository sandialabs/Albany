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
/// Types of nonlinear method for Intrepid nonlinear mini solvers.
///
enum class NonlinearMethod
{
  NEWTON,
  TRUST_REGION,
  CONJUGATE_GRADIENT,
  LINE_SEARCH_REGULARIZED};

///
/// Deal with derivative information for all the mini solvers.
/// Call this when a converged solution is obtained on a system that is
/// typed on a FAD type.
/// Assuming that T is a FAD type and S is a simple type.
///
template<typename T, typename S, Index N>
void
computeFADInfo(
    Vector<T, N> const & r,
    Tensor<S, N> const & DrDx,
    Vector<T, N> & x);

///
/// Function base class that defines the interface to Mini Solvers.
///
template<typename Function_Derived>
class Function_Base
{
public:

  ///
  /// By default use merit function 0.5 dot(gradient, gradient)
  /// as the target to optimize if only the gradient is provided.
  ///
  template<typename T, Index N>
  T
  value(Function_Derived & f, Vector<T, N> const & x)
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == Function_Derived::DIMENSION);

    Vector<T, N> const
    r = f.gradient(x);

    return 0.5 * dot(r, r);
  }

  ///
  /// By default compute gradient with AD from value().
  ///
  template<typename T, Index N>
  Vector<T, N>
  gradient(Function_Derived & f, Vector<T, N> const & x)
  {
    using AD = typename Sacado::Fad::DFad<T>;

    Index const
    dimension = x.get_dimension();

    assert(dimension == Function_Derived::DIMENSION);

    Vector<AD, N>
    x_ad(dimension);

    for (Index i{0}; i < dimension; ++i) {
      x_ad(i) = AD(dimension, i, x(i));
    }

    AD
    f_ad = f.value(x_ad);

    Vector<T, N>
    gradient(dimension);

    for (Index i{0}; i < dimension; ++i) {
      gradient(i) = f_ad.dx(i);
    }

    return gradient;
  }

  ///
  /// By default compute Hessian with AD from gradient().
  ///
  template<typename T, Index N>
  Tensor<T, N>
  hessian(Function_Derived & f, Vector<T, N> const & x)
  {
    using AD = typename Sacado::Fad::DFad<T>;

    Index const
    dimension = x.get_dimension();

    assert(dimension == Function_Derived::DIMENSION);

    Vector<AD, N>
    x_ad(dimension);

    for (Index i{0}; i < dimension; ++i) {
      x_ad(i) = AD(dimension, i, x(i));
    }

    Vector<AD, N>
    r_ad = f.gradient(x_ad);

    Tensor<T, N>
    Hessian(dimension);

    for (Index i{0}; i < dimension; ++i) {
      for (Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = r_ad(i).dx(j);
      }
    }

    return Hessian;
  }

};

///
/// Plain Newton Step
///
template<typename T, Index N>
struct NewtonStep
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
struct TrustRegionStep
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

  Index
  max_num_restrict_iter{4};

  T
  region_size{0.0};

  T
  max_region_size{10.0};

  T
  initial_region_size{10.0};

  T
  min_reduction{0.0};
};

///
/// Conjugate Gradient Step
///
template<typename T, Index N>
struct ConjugateGradientStep
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
  max_num_line_search_iter{16};

  Index
  restart_directions_interval{32};

  T
  line_search_tol{1.0e-6};

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
struct LineSearchRegularizedStep
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

  Index
  max_num_restrict_iter{4};

  T
  step_length{1.0};

  T
  hessian_cond_tol{1.0e+08};

  T
  hessian_singular_tol{1.0e-12};

  Index
  max_num_line_search_iter{16};

  T
  line_search_tol{1.0e-6};
};

///
/// Minimizer Struct
///
template<typename STEP, typename T, Index N>
struct Minimizer
{
public:

  Minimizer(STEP & s) : step_method(s)
  {
    STATIC_ASSERT(Sacado::IsADType<T>::value == false, NO_FAD_ALLOWED);
  }

  template<typename FN>
  void
  solve(FN & fn, Vector<T, N> & x);

  void
  printReport(std::ostream & os);

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

  Index
  max_num_iter{128};

  Index
  num_iter{0};

  T
  rel_tol{1.0e-10};

  T
  rel_error{1.0};

  T
  abs_tol{1.0e-10};

  T
  abs_error{1.0};

  T
  initial_norm{1.0};

  bool
  converged{false};

  Vector<T, N>
  initial_guess;

  Vector<T, N>
  final_soln;

  T
  final_value;

  Vector<T, N>
  final_gradient;

  Tensor<T, N>
  final_hessian;

  STEP &
  step_method;

  char const *
  function_name{nullptr};
};

} // namespace Intrepid

#include "Intrepid_MiniTensor_Solvers.t.h"

#endif // Intrepid_MiniTensor_Solvers_h
