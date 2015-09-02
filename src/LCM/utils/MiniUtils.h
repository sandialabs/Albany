//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniUtils_h)
#define LCM_MiniUtils_h

#include <Intrepid_MiniTensor.h>

namespace LCM
{

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

#endif // LCM_MiniUtils_h
