//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_MiniSolvers_h)
#define LCM_MiniSolvers_h

#include <boost/any.hpp>

#include "Intrepid2_MiniTensor_Solvers.h"

//
// Define some nonlinear systems (NLS) to test nonlinear solution methods.
//
namespace LCM {

//
//
//
template<typename S>
class SquareRootNLS : public Intrepid2::Function_Base<SquareRootNLS<S>, S>
{
public:
  SquareRootNLS(S const c) : c_(c)
  {
  }

  static constexpr
  Intrepid2::Index
  DIMENSION{1};

  static constexpr
  char const * const
  NAME{"Square Root"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<SquareRootNLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid2::Vector<T, N>
    r(dimension);

    r(0) = x(0) * x(0) - c_;

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<SquareRootNLS<S>, S>::hessian(*this, x);
  }

private:
  S const
  c_{0.0};
};

//
//
//
template<typename S>
class QuadraticNLS : public Intrepid2::Function_Base<QuadraticNLS<S>, S>
{
public:
  QuadraticNLS(S const a, S const b, S const c) :  a_(a), b_(b), c_(c)
  {
  }

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Quadratic"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<QuadraticNLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid2::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * c_ * (x(0) - a_);
    r(1) = 2.0 * c_ * (x(1) - b_);

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<QuadraticNLS<S>, S>::hessian(*this, x);
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};

  S const
  c_{0.0};
};

//
//
//
template<typename S>
class GaussianNLS : public Intrepid2::Function_Base<GaussianNLS<S>, S>
{
public:
  GaussianNLS(S const a, S const b, S const c) : a_(a), b_(b), c_(c)
  {
  }

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Inverted Gaussian"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<GaussianNLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid2::Vector<T, N>
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

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<GaussianNLS<S>, S>::hessian(*this, x);
  }

private:
  S const
  a_{0.0};

  S const
  b_{0.0};

  S const
  c_{0.0};
};

//
//
//
template<typename S>
class BananaNLS : public Intrepid2::Function_Base<BananaNLS<S>, S>
{
public:

  BananaNLS()
  {
  }

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Rosenbrock's Banana"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<BananaNLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

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
    return Intrepid2::Function_Base<BananaNLS<S>, S>::hessian(*this, x);
  }

};

//
//
//
template<typename S>
class MatyasNLS : public Intrepid2::Function_Base<MatyasNLS<S>, S>
{
public:

  MatyasNLS() {}

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Matyas"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<MatyasNLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid2::Vector<T, N>
    r(dimension);

    r(0) = (13.0 * x(0) - 12.0 * x(1)) / 25.0;
    r(1) = (13.0 * x(1) - 12.0 * x(0)) / 25.0;

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<MatyasNLS<S>, S>::hessian(*this, x);
  }

};

//
//
//
template<typename S>
class McCormickNLS : public Intrepid2::Function_Base<McCormickNLS<S>, S>
{
public:

  McCormickNLS() {}

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"McCormick"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<McCormickNLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid2::Vector<T, N>
    r(dimension);

    r(0) = std::cos(x(0) + x(1)) + 2.0 * x(0) - 2.0 * x(1) - 1.5;
    r(1) = std::cos(x(0) + x(1)) - 2.0 * x(0) + 2.0 * x(1) + 2.5;

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<McCormickNLS<S>, S>::hessian(*this, x);
  }

};

//
//
//
template<typename S>
class StyblinskiTangNLS : public Intrepid2::Function_Base<StyblinskiTangNLS<S>, S>
{
public:

  StyblinskiTangNLS() {}

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Styblinski-Tang"};

  // Default value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<StyblinskiTangNLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    Intrepid2::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * x(0) * x(0) * x(0) - 16.0 * x(0) + 2.5;
    r(1) = 2.0 * x(1) * x(1) * x(1) - 16.0 * x(1) + 2.5;

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<StyblinskiTangNLS<S>, S>::hessian(*this, x);
  }

};

//
// Define some nonlinear functions (NLF) to test nonlinear optimization methods.
//

//
// Paraboloid of revolution
//
template<typename S>
class Paraboloid : public Intrepid2::Function_Base<Paraboloid<S>, S>
{
public:

  Paraboloid() {}

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Paraboloid"};

  // Explicit value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & x)
  {
    Intrepid2::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    T const
    f = (x(0) * x(0) + x(1) * x(1));

    return f;
  }

  // Default AD gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<Paraboloid<S>, S>::gradient(*this, x);
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<Paraboloid<S>, S>::hessian(*this, x);
  }

};

//
// Beale's function
//
template<typename S>
class Beale : public Intrepid2::Function_Base<Beale<S>, S>
{
public:

  Beale() {}

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Beale"};

  // Explicit value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & X)
  {
    Intrepid2::Index const
    dimension = X.get_dimension();

    assert(dimension == DIMENSION);

    T const &
    x = X(0);

    T const &
    y = X(1);

    T const
    a = 1.5 - x + x * y;

    T const
    b = 2.25 - x + x * y * y;

    T const
    c = 2.625 - x + x * y * y * y;

    T const
    f = a * a + b * b + c * c;

    return f;
  }

  // Default AD gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<Beale<S>, S>::gradient(*this, x);
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<Beale<S>, S>::hessian(*this, x);
  }

};

//
// Booth's function
//
template<typename S>
class Booth : public Intrepid2::Function_Base<Booth<S>, S>
{
public:

  Booth() {}

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Booth"};

  // Explicit value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & X)
  {
    Intrepid2::Index const
    dimension = X.get_dimension();

    assert(dimension == DIMENSION);

    T const &
    x = X(0);

    T const &
    y = X(1);

    T const
    a = x + 2 * y - 7;

    T const
    b = 2 * x + y - 5;

    T const
    f = a * a + b * b;

    return f;
  }

  // Default AD gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<Booth<S>, S>::gradient(*this, x);
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<Booth<S>, S>::hessian(*this, x);
  }

};

//
// Goldstein-Price function
//
template<typename S>
class GoldsteinPrice : public Intrepid2::Function_Base<GoldsteinPrice<S>, S>
{
public:

  GoldsteinPrice() {}

  static constexpr
  Intrepid2::Index
  DIMENSION{2};

  static constexpr
  char const * const
  NAME{"Goldstein-Price"};

  // Explicit value.
  template<typename T, Intrepid2::Index N>
  T
  value(Intrepid2::Vector<T, N> const & X)
  {
    Intrepid2::Index const
    dimension = X.get_dimension();

    assert(dimension == DIMENSION);

    T const &
    x = X(0);

    T const &
    y = X(1);

    T const
    a = x + y + 1;

    T const
    b = 19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y;

    T const
    c = 2 * x - 3 * y;

    T const
    d = 18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y;

    T const
    e = 1 + a * a * b;

    T const
    f = 30 + c * c * d;

    T const
    fn = e * f;

    return fn;
  }

  // Default AD gradient.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<GoldsteinPrice<S>, S>::gradient(*this, x);
  }

  // Default AD hessian.
  template<typename T, Intrepid2::Index N>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x)
  {
    return Intrepid2::Function_Base<GoldsteinPrice<S>, S>::hessian(*this, x);
  }

};

} // namespace LCM

#endif // LCM_MiniSolvers_h
