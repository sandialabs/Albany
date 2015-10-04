//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Intrepid_NonlinearSolver_h)
#define Intrepid_NonlinearSolver_h

#include "PHAL_AlbanyTraits.hpp"
#include "Intrepid_MiniTensor_Solvers.h"

namespace LCM{

///
/// miniMinimize function that wraps the MiniTensor Nonlinear Solvers
/// and deals with Albany traits and AD sensitivities.
///
template<typename OPT, typename FN, Intrepid::Index N>
void
miniMinimize(OPT & optimizer, FN & function, Intrepid::Vector<RealType, N> & x);

template<typename OPT, typename FN, typename T, Intrepid::Index N>
void
miniMinimize(OPT & optimizer, FN & function, Intrepid::Vector<T, N> & x);

} //namesapce LCM

#include "MiniNonlinearSolver.t.h"

#endif // Intrepid_NonlinearSolver_h
