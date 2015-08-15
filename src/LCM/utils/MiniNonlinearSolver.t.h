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
    Residual const & residual,
    Intrepid::Vector<ScalarT, N> & x)
{
  Intrepid::Index const
  dimension = x.get_dimension();

  Intrepid::Vector<ValueT, N>
  x_val = Sacado::Value<Intrepid::Vector<ScalarT, N>>::eval(x);

  Intrepid::Vector<ValueT, N>
  r_val = residual.compute(x_val);

  ValueT const
  initial_norm = Intrepid::norm(r_val);

  if (initial_norm <= this->absolute_tolerance) return;

  Intrepid::Index
  num_iter = 0;

  bool
  converged = false;

  Intrepid::Vector<FadT, N>
  x_fad(dimension);

  while (converged == false) {

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      x_fad(i) = FadT(dimension, i, x_val(i));
    }

    Intrepid::Vector<FadT, N> const
    r_fad = residual.compute(x_fad);

    r_val = Sacado::Value<Intrepid::Vector<ScalarT, N>>::eval(r_fad);

    ValueT const
    residual_norm = Intrepid::norm(r_val);

    ValueT const
    relative_error = residual_norm / initial_norm;

    bool const
    converged_relative = relative_error <= this->relative_tolerance;

    bool const
    converged_absolute = residual_norm <= this->absolute_tolerance;

    converged = converged_relative || converged_absolute;

    bool const
    reached_max_iter = num_iter >= this->maximum_number_iterations;

    bool const
    end_solve = converged || reached_max_iter;

    if (end_solve == true) break;

    Intrepid::Tensor<ValueT, N>
    DrDx(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        DrDx(i, j) = r_fad(i).dx(j);
      }
    }

    Intrepid::Vector<ValueT, N> const
    x_incr = - solve(DrDx, r_val);

    x_val += x_incr;
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
