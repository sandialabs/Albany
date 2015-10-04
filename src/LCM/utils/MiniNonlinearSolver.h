//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MiniNonlinearSolver_h)
#define LCM_MiniNonlinearSolver_h

#include "PHAL_AlbanyTraits.hpp"
#include "Intrepid_MiniTensor_Solvers.h"

namespace LCM{

///
/// miniMinimize function that wraps the MiniTensor Nonlinear Solvers
/// and deals with Albany traits and AD sensitivities.
///
template<typename MIN, typename STEP, typename FN, Intrepid::Index N>
void
miniMinimize(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    Intrepid::Vector<RealType, N> & soln);

template<typename MIN, typename STEP, typename FN, typename T, Intrepid::Index N>
void
miniMinimize(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    Intrepid::Vector<T, N> & soln);

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // LCM_MiniNonlinearSolver_h
