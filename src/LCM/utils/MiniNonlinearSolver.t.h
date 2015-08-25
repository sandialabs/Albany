//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// Specializations for Newton solver
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
  r_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(r);

  ValueT const
  initial_norm = Intrepid::norm(r_val);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    r = residual.compute(x);

    r_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(r);

    this->abs_error_ = Intrepid::norm(r_val);

    this->rel_error_ = this->abs_error_ / initial_norm;

    bool const
    converged_relative = this->rel_error_ <= this->rel_tol_;

    bool const
    converged_absolute = this->abs_error_ <= this->abs_tol_;

    this->converged_ = converged_relative || converged_absolute;

    bool const
    is_max_iter = this->num_iter_ >= this->max_num_iter_;

    bool const
    end_solve = this->converged_ || is_max_iter;

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

    ++this->num_iter_;
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

//
// Specializations for trust-region solver
//

//
// Residual
//
template<typename Residual, Intrepid::Index N>
void
TrustRegionSolver<PHAL::AlbanyTraits::Residual, Residual, N>::
solve(
    Residual & residual,
    Intrepid::Vector<FadT, N> & x)
{
  Intrepid::Index const
  dimension = x.get_dimension();

  Intrepid::Vector<FadT, N>
  r = residual.compute(x);

  Intrepid::Vector<ValueT, N>
  r_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(r);

  ValueT const
  initial_norm = Intrepid::norm(r_val);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    r = residual.compute(x);

    r_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(r);

    this->abs_error_ = Intrepid::norm(r_val);

    this->rel_error_ = this->abs_error_ / initial_norm;

    bool const
    converged_relative = this->rel_error_ <= this->rel_tol_;

    bool const
    converged_absolute = this->abs_error_ <= this->abs_tol_;

    this->converged_ = converged_relative || converged_absolute;

    bool const
    is_max_iter = this->num_iter_ >= this->max_num_iter_;

    bool const
    end_solve = this->converged_ || is_max_iter;

    if (end_solve == true) break;

    Intrepid::Tensor<ValueT, N>
    DrDx(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        DrDx(i, j) = r(i).dx(j);
      }
    }

    Intrepid::Vector<ValueT, N> const
    p = Intrepid::solve(DrDx, r_val);

    Intrepid::Vector<ValueT, N>
    x_val = Sacado::Value<Intrepid::Vector<FadT, N>>::eval(x);

    ValueT const
    f_x = this->abs_error_;

    Intrepid::Vector<ValueT, N> const
    xp_val = x_val + p;

    Intrepid::Vector<ValueT, N> const
    rp_val = residual.compute(xp_val);

    ValueT const
    f_xp = Intrepid::norm(rp_val);

    ValueT const
    m_0 = f_x;

    ValueT const
    m_p = f_x;

    ++this->num_iter_;
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
