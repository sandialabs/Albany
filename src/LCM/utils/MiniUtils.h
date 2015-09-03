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
enum class NonlinearMethod {NEWTON, TRUST_REGION, CONJUGATE_GRADIENT};

///
/// Residual interface for mini nonlinear solver
/// To use the solver framework, derive from this class and perform
/// residual computations in the compute method.
///
template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class Residual_Base
{
public:
  virtual
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) = 0;

  virtual
  ~Residual_Base() {}
};

///
/// Mini Solver Base Class
///
template<typename Residual, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class MiniSolver
{
public:

  MiniSolver()
  {
    STATIC_ASSERT(Sacado::IsADType<T>::value == false, no_fad_allowed);
  }

  virtual
  ~MiniSolver()
  {}

  virtual
  void
  solve(Residual & residual, Intrepid::Vector<T, N> & x) = 0;

};

///
/// Nonlinear mini solvers
///
template<typename Residual, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
std::unique_ptr<MiniSolver<Residual, T, N>>
nonlinearMethodFactory(NonlinearMethod const method_type);

///
/// Newton Mini Solver class
///
template<typename Residual, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
class NewtonMiniSolver : public MiniSolver<Residual, T, N>
{
public:

  virtual
  ~NewtonMiniSolver()
  {}

  virtual
  void
  solve(Residual & residual, Intrepid::Vector<T, N> & x) override;

  void setMaximumNumberIterations(T && mni)
  {max_num_iter_ = std::forward<T>(mni);}

  Intrepid::Index
  getMaximumNumberIterations()
  {return max_num_iter_;}

  Intrepid::Index
  getNumberIterations()
  {return num_iter_;}

  void setRelativeTolerance(T && rt)
  {rel_tol_ = std::forward<T>(rt);}

  T
  getRelativeTolerance() const
  {return rel_tol_;}

  T
  getRelativeError() const
  {return rel_error_;}

  void setAbsoluteTolerance(T && at)
  {abs_tol_ = std::forward<T>(at);}

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

//
// Deal with derivative information for all the mini solvers.
// Call this when a converged solution is obtained on a system that is
// typed on a FAD type.
// Assuming that T is a FAD type and S is a simple type.
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

  // Extract sensitivities of r wrt p
  Intrepid::Matrix<S>
  DrDp(dimension, order);

  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DrDp(i, j) = r(i).dx(j);
    }
  }

  // Solve for all DxDp
  Intrepid::Matrix<S>
  DxDp = Intrepid::solve(DrDx, DrDp);

  // Pack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }

}

} // namespace LCM

#include "MiniUtils.t.h"

#endif // LCM_MiniUtils_h
