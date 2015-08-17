//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// Specializations
//

//
// Residual
//
template<typename Residual, Intrepid::Index N>
void
NewtonSolver<PHAL::AlbanyTraits::Residual, Residual, N>::
solve(
    Residual & residual,
    Intrepid::Vector<FadT, N> & x)
{
  Intrepid::Index const
  dimension = x.get_dimension();

  Intrepid::Vector<FadT, N>
  r = residual.compute(x);

  Intrepid::Vector<ValueT, N>
  x_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(x);

  Intrepid::Vector<ValueT, N>
  r_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(r);

  ValueT const
  initial_norm = Intrepid::norm(r_val);

  this->number_iterations = 0;

  if (initial_norm <= this->absolute_tolerance) return;

  bool
  converged = false;

  while (converged == false) {

    r = residual.compute(x);

    r_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(r);

    this->absolute_error = Intrepid::norm(r_val);

    this->relative_error = this->absolute_error / initial_norm;

    bool const
    converged_relative = this->relative_error <= this->relative_tolerance;

    bool const
    converged_absolute = this->absolute_error <= this->absolute_tolerance;

    converged = converged_relative || converged_absolute;

    bool const
    is_max_iter = this->number_iterations >= this->maximum_number_iterations;

    bool const
    end_solve = converged || is_max_iter;

    if (end_solve == true) break;

    Intrepid::Tensor<ValueT, N>
    DrDx(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        DrDx(i, j) = r(i).dx(j);
      }
    }

    Intrepid::Vector<ValueT, N> const
    x_incr = - Intrepid::solve(DrDx, r_val);

    x += x_incr;

    ++this->number_iterations;
  }

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
