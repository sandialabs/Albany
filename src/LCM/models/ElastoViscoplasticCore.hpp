//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ElastoViscoplasticCore_hpp)
#define ElastoViscoplasticCore_hpp

#include <MiniNonlinearSolver.h>

namespace EV
{

static constexpr Intrepid2::Index MAX_DIM = 3;

//
//! Nonlinear Solver (NLS) class for the ElastoViscoplastic model
//
template<typename EvalT>
class ElastoViscoplasticNLS:
    public Intrepid2::Function_Base<ElastoViscoplasticNLS<EvalT>, typename EvalT::ScalarT>
{
  using S = typename EvalT::ScalarT;

public:

  //! Constructor.
  ElastoViscoplasticNLS(
      RealType dt,
      RealType kw,
      RealType en,
      RealType sn,
      RealType fn,
      RealType eHn,
      RealType eHn_coeff,
      RealType sHn,
      RealType fHen,
      RealType fHen_coeff,
      RealType alpha1,
      RealType alpha2,
      RealType Ra,
      RealType q1,
      RealType q2,
      RealType q3,
      RealType fcrit,
      RealType ffail,
      S mu,
      S bulk,
      S Y,
      S H,
      S Rd,
      S rate_coeff,
      S rate_exp,
      S p,
      Intrepid2::Tensor<S, MAX_DIM> const & s);

  static constexpr char const * const NAME =
      "ElastoViscoplastic Nonlinear System";

  //! Default implementation of value.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const;


  //! Default implementation of hessian.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

private:

  RealType num_dim_;
  RealType dt_;
  RealType kw_;
  RealType en_;
  RealType sn_;
  RealType fn_;
  RealType eHn_;
  RealType eHn_coeff_;
  RealType sHn_;
  RealType fHen_;
  RealType fHen_coeff_;
  RealType alpha1_;
  RealType alpha2_;
  RealType Ra_;
  RealType q1_;
  RealType q2_;
  RealType q3_;
  RealType fcrit_;
  RealType ffail_;
  S mu_;
  S bulk_;
  S Y_;
  S H_;
  S Rd_;
  S rate_coeff_;
  S rate_exp_;
  S p_;
  Intrepid2::Tensor<S, MAX_DIM> const & s_;

};

} // namespace EV

#include "ElastoViscoplasticCore_Def.hpp"

#endif
