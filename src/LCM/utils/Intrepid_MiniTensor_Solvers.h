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
/// Utility function to compute value of a nonlinear system.
///
template<typename NLS, typename T, Index N = DYNAMIC>
Vector<T, N>
getValue(NLS const & nls, Vector<T, N> const & x);

///
/// Utility function to compute gradient of a nonlinear system.
///
template<typename NLS, typename T, Index N = DYNAMIC>
Tensor<T, N>
getGradient(NLS const & nls, Vector<T, N> const & x);

///
/// Nonlinear Method Base Class
///
template<typename NLS, typename T, Index N = DYNAMIC>
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
  char const * const
  name() const = 0;

  virtual
  void
  solve(NLS const & nls, Vector<T, N> & x) = 0;

  void
  setMaxNumIterations(Index const mni)
  {max_num_iter_ = mni;}

  Index
  getMaxNumIterations()
  {return max_num_iter_;}

  Index
  getNumIterations()
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

  T
  getInitialResidualNorm() const
  {return initial_norm_;}

  Vector<T, N>
  getInitialGuess() const
  {return initial_guess_;}

  Vector<T, N>
  getFinalSolution() const
  {return final_soln_;}

  void printReport(std::ostream & os)
  {
    std::string const
    cs = isConverged() == true ? "YES" : "NO";

    //std::string const
    //cs = isConverged() == true ? "\U0001F60A" : "\U0001F623";

    os << "\n\n";
    os << "Method     : " << name() << '\n';
    os << "System     : " << NLS::NAME << '\n';
    os << "Converged  : " << cs << '\n';
    os << "Max Iters  : " << getMaxNumIterations() << '\n';
    os << "Iters Taken: " << getNumIterations() << '\n';

    os << std::scientific << std::setprecision(16);

    os << "Initial |R|: " << getInitialResidualNorm() << '\n';
    os << "Abs Tol    : " << getAbsoluteTolerance() << '\n';
    os << "Abs Error  : " << getAbsoluteError() << '\n';
    os << "Rel Tol    : " << getRelativeTolerance() << '\n';
    os << "Rel Error  : " << getRelativeError() << '\n';
    os << "Initial X  : " << getInitialGuess() << '\n';
    os << "Final X    : " << getFinalSolution() << '\n';
    os << '\n';
  }

protected:
  void
  initConvergenceCriterion(T const in)
  {initial_norm_ = in;}

  void
  updateConvergenceCriterion(T const abs_error)
  {
    abs_error_ = abs_error;
    rel_error_ = initial_norm_ > 0.0 ? abs_error_ / initial_norm_ : 0.0;

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

  void
  setInitialGuess(Vector<T, N> const & x)
  {
    initial_guess_ = x;
  }

  void
  setFinalSolution(Vector<T, N> const & x)
  {
    final_soln_ = x;
  }

protected:
  Index
  max_num_iter_{128};

  Index
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

  Vector<T, N>
  initial_guess_;

  Vector<T, N>
  final_soln_;
};

///
/// Nonlinear method factory
///
template<typename NLS, typename T, Index N = DYNAMIC>
std::unique_ptr<NonlinearMethod_Base<NLS, T, N>>
nonlinearMethodFactory(NonlinearMethod const method_type);

///
/// Newton Method class
///
template<typename NLS, typename T, Index N = DYNAMIC>
class NewtonMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~NewtonMethod() {}

  virtual
  char const * const
  name() const override
  {return "Newton";}

  virtual
  void
  solve(NLS const & nls, Vector<T, N> & x) override;
};

///
/// Trust Region method class.  See Nocedal's algorithm 11.5.
///
template<typename NLS, typename T, Index N = DYNAMIC>
class TrustRegionMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~TrustRegionMethod() {}

  virtual
  char const * const
  name() const override
  {return "Trust Region";}

  virtual
  void
  solve(NLS const & nls, Vector<T, N> & x) override;

  void
  setMaxNumRestrictIterations(Index const n)
  {max_num_restrict_iter_ = n;}

  Index
  getMaxNumRestrictIterations()
  {return max_num_restrict_iter_;}

  void
  setMaxRegionSize(T const l)
  {max_region_size_ = l;}

  T
  getMaxRegionSize() const
  {return max_region_size_;}

  void
  setInitialRegionSize(T const l)
  {initial_region_size_ = l;}

  T
  getInitialRegionSize() const
  {return initial_region_size_;}

  void
  setMinimumReduction(T const r)
  {min_reduction_ = r;}

  T
  getMinumumReduction() const
  {return min_reduction_;}

private:
  Index
  max_num_restrict_iter_{4};

  T
  max_region_size_{10.0};

  T
  initial_region_size_{10.0};

  T
  min_reduction_{0.0};
};

///
/// Conjugate Gradient Method class.
/// For now the Gram-Schmidt method is fixed to Polak-Ribiere
/// and preconditioning with the Hessian.
/// This is taken from J.R. Shewchuck "painless" conjugate gradient
/// manuscript that is all over the place on the net.
///
template<typename NLS, typename T, Index N = DYNAMIC>
class ConjugateGradientMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~ConjugateGradientMethod() {}

  virtual
  char const * const
  name() const override
  {return "Preconditioned Conjugate Gradient";}

  virtual
  void
  solve(NLS const & nls, Vector<T, N> & x) override;

  void
  setMaxNumLineSearchIterations(T const n)
  {max_num_line_search_iter_ = n;}

  Index
  getMaxNumLineSearchIterations()
  {return max_num_line_search_iter_;}

  void
  setLineSearchTolerance(T const st)
  {line_search_tol_ = st;}

  T
  getLineSearchTolerance() const
  {return line_search_tol_;}

  void
  setRestartDirectionsInterval(Index const rdi)
  {restart_directions_interval_ = rdi;}

  Index
  getRestartDirectionsInterval() const
  {return restart_directions_interval_;}

private:
  Index
  max_num_line_search_iter_{16};

  Index
  restart_directions_interval_{32};

  T
  line_search_tol_{1.0e-6};
};

///
/// LineSearchRegularized Method class. See Nocedal's 2nd Ed Algorithm 11.4
///
template<typename NLS, typename T, Index N = DYNAMIC>
class LineSearchRegularizedMethod : public NonlinearMethod_Base<NLS, T, N>
{
public:

  virtual
  ~LineSearchRegularizedMethod() {}

  virtual
  char const * const
  name() const override
  {return "Line Search Regularized Newton-like";}

  virtual
  void
  solve(NLS const & nls, Vector<T, N> & x) override;

  void
  setMaxNumRestrictIterations(Index const mntri)
  {max_num_restrict_iter_ = mntri;}

  Index
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

  Index
  getMaxNumLineSearchIterations()
  {return max_num_line_search_iter_;}

  void
  setLineSearchTolerance(T const st)
  {line_search_tol_ = st;}

  T
  getLineSearchTolerance() const
  {return line_search_tol_;}

private:
  Index
  max_num_restrict_iter_{4};

  T
  max_step_length_{1.0};

  T
  initial_step_length_{1.0};

  T
  hessian_cond_tol_{1.0e+08};

  Index
  max_num_line_search_iter_{16};

  T
  line_search_tol_{1.0e-6};
};

} // namespace Intrepid

#include "Intrepid_MiniTensor_Solvers.t.h"

#endif // Intrepid_MiniTensor_Solvers_h
