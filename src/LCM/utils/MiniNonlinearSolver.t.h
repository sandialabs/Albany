//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_NonlinearSolver_t_h)
#define Intrepid_NonlinearSolver_t_h

namespace Intrepid
{

template <typename Residual, typename T, Index N>
Vector<T, N>
NonlinearSolver<Residual, T, N>::solve(
    Residual & residual,
    Vector<T, N> const & initial_guess)
{
  Index const
  dimension = initial_guess.get_dimension();

  Vector<T, N>
  X = initial_guess;

  Vector<FAD, N>
  XD(dimension);

  bool
  converged = false;

  T const
  tolerance = 1.0e-10;

  while (converged == false) {

    for (Index i{0}; i < dimension; ++i) {
      XD(i) = FAD(dimension, i, X(i));
    }

    Vector<FAD, N> const
    RD = residual.compute(XD);

    Vector<T, N> const
    R = Sacado::Value<Vector<T, N>>::eval(RD);

    Tensor<T, N>
    dRdX(dimension);

    for (Index i{0}; i < dimension; ++i) {
      for (Index j{0}; j < dimension; ++j) {
        dRdX(i, j) = RD(i).dx(j);
      }
    }

    Vector<T, N> const
    dX = - solve(dRdX, R);

    X += dX;

  }

  return initial_guess;
}

} // namespace Intrepid

#endif // Intrepid_NonlinearSolver_t_h
