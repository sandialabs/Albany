//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_MiniSolvers_h)
#define LCM_MiniSolvers_h

#include "Intrepid_MiniTensor_Solvers.h"

//
// Define some nonlinear systems (NLS) to test nonlinear solution methods.
//
namespace LCM {

template <typename S>
class SquareRootNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  SquareRootNLS(S const c) : c_(c) {}

  static constexpr
  Intrepid::Index
  DIMENSION = 1;

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = x(0) * x(0) - c_;

    return r;
  }

private:
  S const
  c_{0.0};
};

template <typename S>
class QuadraticNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  QuadraticNLS(S const a, S const b, S const c) :  a_(a), b_(b), c_(c) {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * c_ * (x(0) - a_);
    r(1) = 2.0 * c_ * (x(1) - b_);

    return r;
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};

  S const
  c_{0.0};
};

template <typename S>
class GaussianNLS  : public Intrepid::NonlinearSystem_Base<S>
{
public:

  GaussianNLS(S const a, S const b, S const c) : a_(a), b_(b), c_(c) {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    T const
    xa = (x(0) - a_) * c_;

    T const
    xb = (x(1) - b_) * c_;

    T const
    e = std::exp(- xa * xa - xb * xb);

    r(0) = 2.0 * xa * e * c_ * c_;
    r(1) = 2.0 * xb * e * c_ * c_;

    return r;
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};

  S const
  c_{0.0};
};

template <typename S>
class BananaNLS  : public Intrepid::NonlinearSystem_Base<S>
{
public:

  BananaNLS(S const a, S const b) : a_(a), b_(b) {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = 2 * (x(0) - a_) + 4 * b_ * x(0) * (x(0) * x(0) - x(1));
    r(1) = 2 * b_ * (x(1) - x(0) * x(0));

    return r;
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};
};

//
// Define some nonlinear functions (NLF) to test nonlinear optimization methods.
//
template <typename S>
class CubicFn
{
public:

  CubicFn(S const c) : c_(c) {}

  static constexpr
  Intrepid::Index
  DIMENSION = 1;

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  T
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    T
    f = x(0) * x(0) * x(0) / 3.0 - c_ * x(0);

    return f;
  }

private:
  S const
  c_{0.0};
};

} // namespace LCM

#endif // LCM_MiniSolvers_h
