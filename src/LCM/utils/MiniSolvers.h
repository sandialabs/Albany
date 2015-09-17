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

  static constexpr
  char const * const
  NAME = "Square Root";

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

  static constexpr
  char const * const
  NAME = "Quadratic";

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
class GaussianNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  GaussianNLS(S const a, S const b, S const c) : a_(a), b_(b), c_(c) {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  static constexpr
  char const * const
  NAME = "Inverted Gaussian";

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
class BananaNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  BananaNLS() {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  static constexpr
  char const * const
  NAME = "Rosenbrock's Banana";

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * (x(0) - 1.0) + 400.0 * x(0) * (x(0) * x(0) - x(1));
    r(1) = 200.0 * (x(1) - x(0) * x(0));

    return r;
  }

};

template <typename S>
class MatyasNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  MatyasNLS() {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  static constexpr
  char const * const
  NAME = "Matyas";

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = (13.0 * x(0) - 12.0 * x(1)) / 25.0;
    r(1) = (13.0 * x(1) - 12.0 * x(0)) / 25.0;

    return r;
  }

};

template <typename S>
class McCormickNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  McCormickNLS() {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  static constexpr
  char const * const
  NAME = "McCormick";

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = std::cos(x(0) + x(1)) + 2.0 * x(0) - 2.0 * x(1) - 1.5;
    r(1) = std::cos(x(0) + x(1)) - 2.0 * x(0) + 2.0 * x(1) + 2.5;

    return r;
  }

};

template <typename S>
class StyblinskiTangNLS : public Intrepid::NonlinearSystem_Base<S>
{
public:

  StyblinskiTangNLS() {}

  static constexpr
  Intrepid::Index
  DIMENSION = 2;

  static constexpr
  char const * const
  NAME = "Styblinski-Tang";

  template <typename T, Intrepid::Index N = Intrepid::DYNAMIC>
  Intrepid::Vector<T, N>
  compute(Intrepid::Vector<T, N> const & x) const
  {
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * x(0) * x(0) * x(0) - 16.0 * x(0) + 2.5;
    r(1) = 2.0 * x(1) * x(1) * x(1) - 16.0 * x(1) + 2.5;

    return r;
  }

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

  static constexpr
  char const * const
  NAME = "Cubic";

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
