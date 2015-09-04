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
template<typename NLS, Intrepid::Index N>
void
MiniNonlinearSolver<PHAL::AlbanyTraits::Jacobian, NLS, N>::
solve(NLS & nls, Intrepid::Vector<ScalarT, N> & soln)
{
  // Extract values and use them to solve the NLS.
  Intrepid::Vector<ValueT, N>
  soln_val = Sacado::Value<Intrepid::Vector<ScalarT, N>>::eval(soln);

  this->nonlinear_method_.solve(nls, soln_val);

  auto const
  dimension = soln.get_dimension();

  // Put values back in solution vector
  for (auto i = 0; i < dimension; ++i) {
    soln(i).val() = soln_val(i);
  }

  // Get the Hessian evaluated at the solution.
  using AD = typename Sacado::Fad::DFad<ValueT>;

  Intrepid::Vector<AD, N>
  soln_ad(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    soln_ad(i) = AD(dimension, i, soln_val(i));
  }

  Intrepid::Vector<AD, N>
  resi_ad = nls.compute(soln_ad);

  Intrepid::Tensor<ValueT, N>
  DrDx(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    for (Intrepid::Index j{0}; j < dimension; ++j) {
      DrDx(i, j) = resi_ad(i).dx(j);
    }
  }

  // Now evaluate nls with soln that has Albany sensitivities.
  Intrepid::Vector<ScalarT, N>
  resi = nls.compute(soln);

  computeFADInfo(resi, DrDx, soln);

  return;
}

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
