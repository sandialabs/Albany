//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniUtils_h)
#define LCM_MiniUtils_h

#include <utility>

#include <Intrepid_MiniTensor.h>

namespace LCM
{

///
/// Types of nonlinear method for LCM nonlinear mini solvers.
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
template<typename T, typename S, Intrepid::Index N>
void
computeFADInfo(
    Intrepid::Vector<T, N> const & r,
    Intrepid::Tensor<S, N> const & DrDx,
    Intrepid::Vector<T, N> & x);

///
/// Nonlinear system (NLS) interface for mini nonlinear solver
///
template <typename S>
class NonlinearSystem_Base
{
public:
  NonlinearSystem_Base()
  {
    STATIC_ASSERT(Sacado::IsADType<S>::value == false, NO_FAD_ALLOWED);
  }

};

///
/// Utility function to compute residual of a nonlinear system.
/// The returned residual has the underlying value type of the
/// type of the evaluation vector.
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
Intrepid::Vector<typename Sacado::ValueType<T>::type, N>
computeResidual(NLS const & nls, Intrepid::Vector<T, N> const & x);

///
/// Utility function to compute Hessian of a nonlinear system.
/// The returned Hessian has the underlying value type of the
/// type of the evaluation vector.
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
Intrepid::Tensor<typename Sacado::ValueType<T>::type, N>
computeHessian(NLS const & nls, Intrepid::Vector<T, N> const & x);

///
/// Nonlinear Method Base Class
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class NonlinearMethod_Base
{
public:

  NonlinearMethod_Base()
  {
    STATIC_ASSERT(Sacado::IsADType<T>::value == false, NO_FAD_ALLOWED);
  }

  virtual
  ~NonlinearMethod_Base() {}

  virtual
  void
  solve(NLS const & nls, Intrepid::Vector<T, N> & x) = 0;

  void
  setMaxNumIterations(Intrepid::Index const mni)
  {max_num_iter_ = mni;}

  Intrepid::Index
  getMaxNumIterations()
  {return max_num_iter_;}

  Intrepid::Index
  getNumberIterations()
  {return num_iter_;}

  void
  setRelativeTolerance(T const rt)
  {rel_tol_ = rt;}

  T
  getRelativeTolerance() const
  {return rel_tol_;}

  T
  getRelativeError() const
  {return rel_error_;}

  void
  setAbsoluteTolerance(T const at)
  {abs_tol_ = at;}

  T
  getAbsoluteTolerance() const
  {return abs_tol_;}

  T
  getAbsoluteError() const
  {return abs_error_;}

  bool
  isConverged() const
  {return converged_;}

protected:
  void
  initConvergenceCriterion(T const in)
  {initial_norm_ = in;}

  void
  updateConvergenceCriterion(T const abs_error)
  {
    abs_error_ = abs_error;
    rel_error_ = abs_error_ / initial_norm_;

    bool const
    converged_absolute = abs_error_ <= abs_tol_;

    bool const
    converged_relative = rel_error_ <= rel_tol_;

    converged_ = converged_absolute || converged_relative;
  }

  bool
  continueSolve() const
  {
    bool const
    is_max_iter = num_iter_ >= max_num_iter_;

    bool const
    end_solve = is_max_iter == true || converged_ == true;

    bool const
    continue_solve = end_solve == false;

    return continue_solve;
  }

  void
  increaseIterationCounter()
  {
    ++num_iter_;
  }

protected:
  Intrepid::Index
  max_num_iter_{128};

  Intrepid::Index
  num_iter_{0};

  T
  rel_tol_{1.0e-10};

  T
  rel_error_{1.0};

  T
  abs_tol_{1.0e-10};

  T
  abs_error_{1.0};

  T
  initial_norm_{1.0};

  bool
  converged_{false};
};

///
/// Nonlinear method factory
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
std::unique_ptr<NonlinearMethod_Base<NLS, T, N>>
nonlinearMethodFactory(NonlinearMethod const method_type);

///
/// Newton Method class
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class NewtonMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~NewtonMethod() {}

  virtual
  void
  solve(NLS const & nls, Intrepid::Vector<T, N> & x) override;
};

///
/// Trust Region method class.  See Nocedal's algorithm 11.5.
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class TrustRegionMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~TrustRegionMethod() {}

  virtual
  void
  solve(NLS const & nls, Intrepid::Vector<T, N> & x) override;

  void
  setMaxNumRestrictIterations(Intrepid::Index const mntri)
  {max_num_restrict_iter_ = mntri;}

  Intrepid::Index
  getMaxNumRestrictIterations()
  {return max_num_restrict_iter_;}

  void
  setMaxStepLength(T const msl)
  {max_step_length_ = msl;}

  T
  getMaxStepLength() const
  {return max_step_length_;}

  void
  setInitialStepLength(T const isl)
  {initial_step_length_ = isl;}

  T
  getInitialStepLength() const
  {return initial_step_length_;}

  void
  setMinimumReduction(T const mr)
  {min_reduction_ = mr;}

  T
  getMinumumReduction() const
  {return min_reduction_;}

private:
  Intrepid::Index
  max_num_restrict_iter_{4};

  T
  max_step_length_{1.0};

  T
  initial_step_length_{1.0};

  T
  min_reduction_{0.25};
};

///
/// Conjugate Gradient Method class.
/// For now the Gram-Schmidt method is fixed to Polak-Ribiere
/// and preconditioning with the Hessian.
/// This is taken from J.R. Shewchuck "painless" conjugate gradient
/// manuscript that is all over the place on the net.
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class ConjugateGradientMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~ConjugateGradientMethod() {}

  virtual
  void
  solve(NLS const & nls, Intrepid::Vector<T, N> & x) override;

  void
  setMaxNumLineSearchIterations(T const n)
  {max_num_line_search_iter_ = n;}

  Intrepid::Index
  getMaxNumLineSearchIterations()
  {return max_num_line_search_iter_;}

  void
  setLineSearchTolerance(T const st)
  {line_search_tol_ = st;}

  T
  getLineSearchTolerance() const
  {return line_search_tol_;}

  void
  setRestartDirectionsInterval(Intrepid::Index const rdi)
  {restart_directions_interval_ = rdi;}

  Intrepid::Index
  getRestartDirectionsInterval() const
  {return restart_directions_interval_;}

private:
  Intrepid::Index
  max_num_line_search_iter_{16};

  Intrepid::Index
  restart_directions_interval_{32};

  T
  line_search_tol_{1.0e-6};
};

///
/// LineSearchRegularized Method class. See Nocedal's 2nd Ed Algorithm 11.4
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class LineSearchRegularizedMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~LineSearchRegularizedMethod() {}

  virtual
  void
  solve(NLS const & nls, Intrepid::Vector<T, N> & x) override;

  void
  setMaxNumRestrictIterations(Intrepid::Index const mntri)
  {max_num_restrict_iter_ = mntri;}

  Intrepid::Index
  getMaxNumRestrictIterations()
  {return max_num_restrict_iter_;}

  void
  setMaxStepLength(T const msl)
  {max_step_length_ = msl;}

  T
  getMaxStepLength() const
  {return max_step_length_;}

  void
  setInitialStepLength(T const isl)
  {initial_step_length_ = isl;}

  T
  getInitialStepLength() const
  {return initial_step_length_;}

  void
  setHessianConditionTolerance(T const tol)
  {hessian_cond_tol_ = tol;}

  T
  getHessianConditionTolerance() const
  {return hessian_cond_tol_;}

  void
  setMaxNumLineSearchIterations(T const n)
  {max_num_line_search_iter_ = n;}

  Intrepid::Index
  getMaxNumLineSearchIterations()
  {return max_num_line_search_iter_;}

  void
  setLineSearchTolerance(T const st)
  {line_search_tol_ = st;}

  T
  getLineSearchTolerance() const
  {return line_search_tol_;}

private:
  Intrepid::Index
  max_num_restrict_iter_{4};

  T
  max_step_length_{1.0};

  T
  initial_step_length_{1.0};

  T
  hessian_cond_tol_{1.0e+08};

  Intrepid::Index
  max_num_line_search_iter_{16};

  T
  line_search_tol_{1.0e-6};
};

} // namespace LCM

#include "MiniUtils.t.h"

#endif // LCM_MiniUtils_h
