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
  REGULARIZED_LINE_SEARCH};

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
template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class NonlinearSystem_Base
{
public:
  virtual
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) = 0;

  virtual
  ~NonlinearSystem_Base() {}
};

///
/// Nonlinear Method Base Class
///
template<typename NLS, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class NonlinearMethod_Base
{
public:

  NonlinearMethod_Base()
  {
    STATIC_ASSERT(Sacado::IsADType<T>::value == false, no_fad_allowed);
  }

  virtual
  ~NonlinearMethod_Base() {}

  virtual
  void
  solve(NLS & nls, Intrepid::Vector<T, N> & x) = 0;

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
  solve(NLS & nls, Intrepid::Vector<T, N> & x) override;

  void
  setMaximumNumberIterations(Intrepid::Index const mni)
  {max_num_iter_ = mni;}

  Intrepid::Index
  getMaximumNumberIterations()
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

private:
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

  bool
  converged_{false};
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
  solve(NLS & nls, Intrepid::Vector<T, N> & x) override;

  void
  setMaximumNumberIterations(Intrepid::Index const mni)
  {max_num_iter_ = mni;}

  Intrepid::Index
  getMaximumNumberIterations()
  {return max_num_iter_;}

  void
  setMaximumNumberRestrictIterations(T const mntri)
  {max_num_restrict_iter_ = mntri;}

  Intrepid::Index
  getMaximumNumberRestrictIterations()
  {return max_num_restrict_iter_;}

  Intrepid::Index
  getNumberIterations()
  {return num_iter_;}

  void
  setMaximumStepLength(T const msl)
  {max_step_length_ = msl;}

  T
  getMaximumStepLength() const
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

private:
  Intrepid::Index
  max_num_iter_{128};

  Intrepid::Index
  max_num_restrict_iter_{4};

  Intrepid::Index
  num_iter_{0};

  T
  max_step_length_{1.0};

  T
  initial_step_length_{1.0};

  T
  min_reduction_{0.25};

  T
  rel_tol_{1.0e-10};

  T
  rel_error_{1.0};

  T
  abs_tol_{1.0e-10};

  T
  abs_error_{1.0};

  bool
  converged_{false};
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
  solve(NLS & nls, Intrepid::Vector<T, N> & x) override;

  void
  setMaximumNumberIterations(Intrepid::Index const mni)
  {max_num_iter_ = mni;}

  Intrepid::Index
  getMaximumNumberIterations()
  {return max_num_iter_;}

  T
  getInitialSecantStepLength() const
  {return initial_secant_step_length_;}

  void
  setInitialSecantStepLength(T const issl)
  {initial_secant_step_length_ = issl;}

  void
  setMaximumNumberSecantIterations(T const mntri)
  {max_num_secant_iter_ = mntri;}

  Intrepid::Index
  getMaximumNumberSecantIterations()
  {return max_num_secant_iter_;}

  void
  setSecantTolerance(T const st)
  {secant_tol_ = st;}

  T
  getSecantTolerance() const
  {return secant_tol_;}

  Intrepid::Index
  getNumberIterations()
  {return num_iter_;}

  void
  setRestartDirectionsInterval(Intrepid::Index const rdi)
  {restart_directions_interval_ = rdi;}

  Intrepid::Index
  getRestartDirectionsInterval() const
  {return restart_directions_interval_;}

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

private:
  Intrepid::Index
  max_num_iter_{128};

  Intrepid::Index
  max_num_secant_iter_{16};

  Intrepid::Index
  num_iter_{0};

  Intrepid::Index
  restart_directions_interval_{32};

  T
  initial_secant_step_length_{0.9};

  T
  secant_tol_{1.0e-6};

  T
  rel_tol_{1.0e-10};

  T
  rel_error_{1.0};

  T
  abs_tol_{1.0e-10};

  T
  abs_error_{1.0};

  bool
  converged_{false};
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
  solve(NLS & nls, Intrepid::Vector<T, N> & x) override;

  void
  setMaximumNumberIterations(Intrepid::Index const mni)
  {max_num_iter_ = mni;}

  Intrepid::Index
  getMaximumNumberIterations()
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

private:
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

  bool
  converged_{false};
};

} // namespace LCM

#include "MiniUtils.t.h"

#endif // LCM_MiniUtils_h
