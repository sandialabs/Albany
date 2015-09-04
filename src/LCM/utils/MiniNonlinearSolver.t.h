//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// Specializations for MiniNonlinear solver
//

//
// Residual
//
template<typename NLS, Intrepid::Index N>
void
MiniNonlinearSolver<PHAL::AlbanyTraits::Residual, NLS, N>::
solve(NLS & nls, Intrepid::Vector<ScalarT, N> & soln)
{
  this->nonlinear_method_.solve(nls, soln);
  return;
}

//
// Jacobian
//

//
// Tangent
//

//
// DistParamDeriv
//

#ifdef ALBANY_SG
//
// SGResidual
//

//
// SGJacobian
//

//
// SGTangent
//
#endif // ALBANY_SG

#ifdef ALBANY_ENSEMBLE
//
// MPResidual
//

//
// MPJacobian
//

//
// MPTangent
//

#endif // ALBANY_ENSEMBLE

} // namespace LCM
