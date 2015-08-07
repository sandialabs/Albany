//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_NonlinearSolver_h)
#define Intrepid_NonlinearSolver_h

#include <Intrepid_MiniTensor.h>

namespace Intrepid
{

template <typename Residual, typename T, Index N>
class NonlinearSolver
{
public:
  using FAD = Sacado::Fad::DFad<T>;

  Vector<T, N>
  solve(Residual & residual, Vector<T, N> const & initial_guess);
};

} // namespace Intrepid

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
