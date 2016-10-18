//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_MiniSolvers_h)
#define LCM_MiniSolvers_h

#include "Intrepid2_MiniTensor_Solvers.h"
#include "MiniNonlinearSolver.h"
#include "PHAL_AlbanyTraits.hpp"

//
// Define some nonlinear systems (NLS) to test nonlinear solution methods.
//
namespace LCM {

//
//
//
template<typename S, Intrepid2::Index M = 2>
class Banana : public Intrepid2::Function_Base<Banana<S, M>, S, M>
{
public:

  Banana()
  {
  }

  static constexpr
  char const * const
  NAME{"Rosenbrock's Banana"};

  using Base = Intrepid2::Function_Base<Banana<S, M>, S, M>;

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Base::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == Base::DIMENSION);

    Intrepid2::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * (x(0) - 1.0) + 400.0 * x(0) * (x(0) * x(0) - x(1));
    r(1) = 200.0 * (x(1) - x(0) * x(0));

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Base::hessian(*this, x);
  }

};

//
//
//
template<typename EvalT, Intrepid2::Index M = 2>
class Banana_Traits : public
Intrepid2::Function_Base<Banana_Traits<EvalT, M>, typename EvalT::ScalarT, M>
{
  using S = typename EvalT::ScalarT;

public:

  Banana_Traits(S a = 1.0, S b = 100.0) : a_(a), b_(b)
  {
  }

  static constexpr
  char const * const
  NAME{"Banana_Traits' Function 2D"};

  using Base =
    Intrepid2::Function_Base<Banana_Traits<EvalT, M>, typename EvalT::ScalarT, M>;

  // Explicit value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const
    a = peel<EvalT, T, N>()(a_);

    T const
    b = peel<EvalT, T, N>()(b_);

    T const
    c = (a_ - x(0));

    T const
    d = (x(1) - x(0) * x(0));

    return c * c + b * d * d;
  }

  // Default AD gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x)
  {
    return Base::gradient(*this, x);
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Base::hessian(*this, x);
  }

private:
  S
  a_{1.0};

  S
  b_{100.0};
};

} // namespace LCM

#endif // LCM_MiniSolvers_h
