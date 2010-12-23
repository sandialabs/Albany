//
//
//
#if !defined(LCM_Tensor_i_cc)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "Teuchos_TestForException.hpp"

namespace LCM {

  //
  // Vector construction/destruction
  //
  template<typename Scalar>
  inline
  Vector<Scalar>::Vector()
  {
    e[0] = std::numeric_limits<Scalar>::quiet_NaN();
    e[1] = std::numeric_limits<Scalar>::quiet_NaN();
    e[2] = std::numeric_limits<Scalar>::quiet_NaN();

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Vector<Scalar>::Vector(const Scalar s)
  {
    e[0] = s;
    e[1] = s;
    e[2] = s;

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Vector<Scalar>::Vector(const Scalar s0, const Scalar s1, const Scalar s2)
  {
    e[0] = s0;
    e[1] = s1;
    e[2] = s2;

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Vector<Scalar>::Vector(const Vector<Scalar> & V)
  {
    e[0] = V.e[0];
    e[1] = V.e[1];
    e[2] = V.e[2];

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Vector<Scalar>::~Vector()
  {
    return;
  }

  //
  // Vector utilities
  //
  template<typename Scalar>
  inline void
  Vector<Scalar>::clear()
  {
    e[0] = 0.0;
    e[1] = 0.0;
    e[2] = 0.0;

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline const Scalar &
  Vector<Scalar>::operator()(const Index i) const
  {
    assert(i < MaxDim);
    return e[i];
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar &
  Vector<Scalar>::operator()(const Index i)
  {
    assert(i < MaxDim);
    return e[i];
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar> &
  Vector<Scalar>::operator=(const Vector<Scalar> & v)
  {
    if (*this != v) {
      e[0] = v.e[0];
      e[1] = v.e[1];
      e[2] = v.e[2];
    }
    return *this;
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  operator+(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    Vector<Scalar> s;

    s(0) = u(0) + v(0);
    s(1) = u(1) + v(1);
    s(2) = u(2) + v(2);

    return s;
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  operator-(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    Vector<Scalar> s;

    s(0) = u(0) - v(0);
    s(1) = u(1) - v(1);
    s(2) = u(2) - v(2);

    return s;
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar> &
  Vector<Scalar>::operator+=(const Vector<Scalar> & v)
  {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];

    return *this;
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar> &
  Vector<Scalar>::operator-=(const Vector<Scalar> & v)
  {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];

    return *this;
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  operator*(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    return dot(u, v);
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  operator*(const Scalar s, const Vector<Scalar> & u)
  {
    return Vector<Scalar>(s*u(0), s*u(1), s*u(2));
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  operator*(const Vector<Scalar> & u, const Scalar s)
  {
    return s * u;
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  dot(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    return u(0)*v(0) + u(1)*v(1) + u(2)*v(2);
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  cross(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    Vector<Scalar> w;

    w(0) = u(1)*v(2) - u(2)*v(1);
    w(1) = u(2)*v(0) - u(0)*v(2);
    w(2) = u(0)*v(1) - u(1)*v(0);

    return w;
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  norm(const Vector<Scalar> & u)
  {
    return sqrt(u(0)*u(0) + u(1)*u(1) + u(2)*u(2));
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  norm_1(const Vector<Scalar> & u)
  {
    return std::fabs(u(0)) + std::fabs(u(1)) + std::fabs(u(2));
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  norm_infinity(const Vector<Scalar> & u)
  {
    return std::max(std::max(std::fabs(u(0)),std::fabs(u(1))),std::fabs(u(2)));
  }

  //
  // Second order tensor construction/destruction
  //
  template<typename Scalar>
  inline
  Tensor<Scalar>::Tensor()
  {
    e[0][0] = std::numeric_limits<Scalar>::quiet_NaN();
    e[0][1] = std::numeric_limits<Scalar>::quiet_NaN();
    e[0][2] = std::numeric_limits<Scalar>::quiet_NaN();

    e[1][0] = std::numeric_limits<Scalar>::quiet_NaN();
    e[1][1] = std::numeric_limits<Scalar>::quiet_NaN();
    e[1][2] = std::numeric_limits<Scalar>::quiet_NaN();

    e[2][0] = std::numeric_limits<Scalar>::quiet_NaN();
    e[2][1] = std::numeric_limits<Scalar>::quiet_NaN();
    e[2][2] = std::numeric_limits<Scalar>::quiet_NaN();

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Tensor<Scalar>::Tensor(const Scalar s)
  {
    e[0][0] = s;
    e[0][1] = s;
    e[0][2] = s;

    e[1][0] = s;
    e[1][1] = s;
    e[1][2] = s;

    e[2][0] = s;
    e[2][1] = s;
    e[2][2] = s;

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Tensor<Scalar>::Tensor(
      const Scalar s00, const Scalar s01, const Scalar s02,
      const Scalar s10, const Scalar s11, const Scalar s12,
      const Scalar s20, const Scalar s21, const Scalar s22)
  {
    e[0][0] = s00;
    e[0][1] = s01;
    e[0][2] = s02;

    e[1][0] = s10;
    e[1][1] = s11;
    e[1][2] = s12;

    e[2][0] = s20;
    e[2][1] = s21;
    e[2][2] = s22;

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Tensor<Scalar>::Tensor(const Tensor<Scalar> & A)
  {
    e[0][0] = A.e[0][0];
    e[0][1] = A.e[0][1];
    e[0][2] = A.e[0][2];

    e[1][0] = A.e[1][0];
    e[1][1] = A.e[1][1];
    e[1][2] = A.e[1][2];

    e[2][0] = A.e[2][0];
    e[2][1] = A.e[2][1];
    e[2][2] = A.e[2][2];

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline
  Tensor<Scalar>::~Tensor()
  {
    return;
  }

  //
  // Tensor utilities
  //
  template<typename Scalar>
  inline void
  Tensor<Scalar>::clear()
  {
    e[0][0] = 0.0;
    e[0][1] = 0.0;
    e[0][2] = 0.0;

    e[1][0] = 0.0;
    e[1][1] = 0.0;
    e[1][2] = 0.0;

    e[2][0] = 0.0;
    e[2][1] = 0.0;
    e[2][2] = 0.0;

    return;
  }

  //
  //
  //
  template<typename Scalar>
  inline const Scalar &
  Tensor<Scalar>::operator()(const Index i, const Index j) const
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    return e[i][j];
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar &
  Tensor<Scalar>::operator()(const Index i, const Index j)
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    return e[i][j];
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar> &
  Tensor<Scalar>::operator=(const Tensor<Scalar> & A)
  {
    if (*this != A) {
      e[0][0] = A.e[0][0];
      e[0][1] = A.e[0][1];
      e[0][2] = A.e[0][2];

      e[1][0] = A.e[1][0];
      e[1][1] = A.e[1][1];
      e[1][2] = A.e[1][2];

      e[2][0] = A.e[2][0];
      e[2][1] = A.e[2][1];
      e[2][2] = A.e[2][2];
    }
    return *this;
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar> &
  Tensor<Scalar>::operator+=(const Tensor<Scalar> & A)
  {
    e[0][0] += A.e[0][0];
    e[0][1] += A.e[0][1];
    e[0][2] += A.e[0][2];

    e[1][0] += A.e[1][0];
    e[1][1] += A.e[1][1];
    e[1][2] += A.e[1][2];

    e[2][0] += A.e[2][0];
    e[2][1] += A.e[2][1];
    e[2][2] += A.e[2][2];

    return *this;
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar> &
  Tensor<Scalar>::operator-=(const Tensor<Scalar> & A)
  {
    e[0][0] -= A.e[0][0];
    e[0][1] -= A.e[0][1];
    e[0][2] -= A.e[0][2];

    e[1][0] -= A.e[1][0];
    e[1][1] -= A.e[1][1];
    e[1][2] -= A.e[1][2];

    e[2][0] -= A.e[2][0];
    e[2][1] -= A.e[2][1];
    e[2][2] -= A.e[2][2];

    return *this;
  }


  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  operator+(const Tensor<Scalar> & A, const Tensor<Scalar> & B)
  {
    Tensor<Scalar> S;

    S(0,0) = A(0,0) + B(0,0);
    S(0,1) = A(0,1) + B(0,1);
    S(0,2) = A(0,2) + B(0,2);

    S(1,0) = A(1,0) + B(1,0);
    S(1,1) = A(1,1) + B(1,1);
    S(1,2) = A(1,2) + B(1,2);

    S(2,0) = A(2,0) + B(2,0);
    S(2,1) = A(2,1) + B(2,1);
    S(2,2) = A(2,2) + B(2,2);

    return S;
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  operator-(const Tensor<Scalar> & A, const Tensor<Scalar> & B)
  {
    Tensor<Scalar> S;

    S(0,0) = A(0,0) - B(0,0);
    S(0,1) = A(0,1) - B(0,1);
    S(0,2) = A(0,2) - B(0,2);

    S(1,0) = A(1,0) - B(1,0);
    S(1,1) = A(1,1) - B(1,1);
    S(1,2) = A(1,2) - B(1,2);

    S(2,0) = A(2,0) - B(2,0);
    S(2,1) = A(2,1) - B(2,1);
    S(2,2) = A(2,2) - B(2,2);

    return S;
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  operator*(const Tensor<Scalar> & A, const Tensor<Scalar> & B)
  {
    return dot(A, B);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  operator*(const Scalar s, const Tensor<Scalar> & A)
  {
    return Tensor<Scalar>(
        s*A(0,0), s*A(0,1), s*A(0,2),
        s*A(1,0), s*A(1,1), s*A(1,2),
        s*A(2,0), s*A(2,1), s*A(2,2));
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  operator*(const Tensor<Scalar> & A, const Scalar s)
  {
    return s * A;
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  operator*(const Tensor<Scalar> & A, const Vector<Scalar> & u)
  {
    return dot(A,u);
  }


  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  operator*(const Vector<Scalar> & u, const Tensor<Scalar> & A)
  {
    return dot(u,A);
  }


  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  dot(const Tensor<Scalar> & A, const Tensor<Scalar> & B)
  {
    Tensor<Scalar> C;

    C(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0) + A(0,2)*B(2,0);
    C(0,1) = A(0,0)*B(0,1) + A(0,1)*B(1,1) + A(0,2)*B(2,1);
    C(0,2) = A(0,0)*B(0,2) + A(0,1)*B(1,2) + A(0,2)*B(2,2);

    C(1,0) = A(1,0)*B(0,0) + A(1,1)*B(1,0) + A(1,2)*B(2,0);
    C(1,1) = A(1,0)*B(0,1) + A(1,1)*B(1,1) + A(1,2)*B(2,1);
    C(1,2) = A(1,0)*B(0,2) + A(1,1)*B(1,2) + A(1,2)*B(2,2);

    C(2,0) = A(2,0)*B(0,0) + A(2,1)*B(1,0) + A(2,2)*B(2,0);
    C(2,1) = A(2,0)*B(0,1) + A(2,1)*B(1,1) + A(2,2)*B(2,1);
    C(2,2) = A(2,0)*B(0,2) + A(2,1)*B(1,2) + A(2,2)*B(2,2);

    return C;
  }


  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  dot(const Tensor<Scalar> & A, const Vector<Scalar> & u)
  {
    Vector<Scalar> v;

    v(0) = A(0,0)*u(0) + A(0,1)*u(1) + A(0,2)*u(2);
    v(1) = A(1,0)*u(0) + A(1,1)*u(1) + A(1,2)*u(2);
    v(2) = A(2,0)*u(0) + A(2,1)*u(1) + A(2,2)*u(2);

    return v;
  }


  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  dot(const Vector<Scalar> & u, const Tensor<Scalar> & A)
  {
    Vector<Scalar> v;

    v(0) = A(0,0)*u(0) + A(1,0)*u(1) + A(2,0)*u(2);
    v(1) = A(0,1)*u(0) + A(1,1)*u(1) + A(2,1)*u(2);
    v(2) = A(0,2)*u(0) + A(1,2)*u(1) + A(2,2)*u(2);

    return v;
  }


  //
  //
  //
  template<typename Scalar>
  inline Scalar
  dotdot(const Tensor<Scalar> & A, const Tensor<Scalar> & B)
  {
    Scalar s = 0.0;

    s+= A(0,0)*B(0,0) + A(0,1)*B(0,1) + A(0,2)*B(0,2);
    s+= A(1,0)*B(1,0) + A(1,1)*B(1,1) + A(1,2)*B(1,2);
    s+= A(2,0)*B(2,0) + A(2,1)*B(2,1) + A(2,2)*B(2,2);

    return s;
  }


  //
  // Frobenius norm
  //
  template<typename Scalar>
  inline Scalar
  norm(const Tensor<Scalar> & A)
  {
    Scalar s = 0.0;

    s+= A(0,0)*A(0,0) + A(0,1)*A(0,1) + A(0,2)*A(0,2);
    s+= A(1,0)*A(1,0) + A(1,1)*A(1,1) + A(1,2)*A(1,2);
    s+= A(2,0)*A(2,0) + A(2,1)*A(2,1) + A(2,2)*A(2,2);

    return sqrt(s);
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  norm_1(const Tensor<Scalar> & A)
  {
    Scalar s0 = std::fabs(A(0,0)) + std::fabs(A(1,0)) + std::fabs(A(2,0));
    Scalar s1 = std::fabs(A(0,1)) + std::fabs(A(1,1)) + std::fabs(A(2,1));
    Scalar s2 = std::fabs(A(0,2)) + std::fabs(A(1,2)) + std::fabs(A(2,2));

    return std::max(std::max(s0,s1),s2);
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  norm_infinity(const Tensor<Scalar> & A)
  {
    Scalar s0 = std::fabs(A(0,0)) + std::fabs(A(0,1)) + std::fabs(A(0,2));
    Scalar s1 = std::fabs(A(1,0)) + std::fabs(A(1,1)) + std::fabs(A(1,2));
    Scalar s2 = std::fabs(A(2,0)) + std::fabs(A(2,1)) + std::fabs(A(2,2));

    return std::max(std::max(s0,s1),s2);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  dyad(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    Tensor<Scalar> A;

    A(0,0) = u(0) * v(0);
    A(0,1) = u(0) * v(1);
    A(0,2) = u(0) * v(2);

    A(1,0) = u(1) * v(0);
    A(1,1) = u(1) * v(1);
    A(1,2) = u(1) * v(2);

    A(2,0) = u(2) * v(0);
    A(2,1) = u(2) * v(1);
    A(2,2) = u(2) * v(2);

    return A;
  }

  //
  // Just for Jay
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  bun(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    return dyad(u,v);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  tensor(const Vector<Scalar> & u, const Vector<Scalar> & v)
  {
    return dyad(u,v);
  }

  //
  //
  //
  template<typename Scalar>
  inline const Tensor<Scalar>
  eye()
  {
    return Tensor<Scalar>(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
  }

  //
  //
  //
  template<typename Scalar>
  inline const Tensor<Scalar>
  identity()
  {
    return Tensor<Scalar>(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  transpose(const Tensor<Scalar> & A)
  {
    return Tensor<Scalar>(
        A(0,0),A(1,0),A(2,0),
        A(0,1),A(1,1),A(2,1),
        A(0,2),A(1,2),A(2,2));
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  symm(const Tensor<Scalar> & A)
  {
    const Scalar s00 = A(0,0);
    const Scalar s11 = A(1,1);
    const Scalar s22 = A(2,2);

    const Scalar s01 = 0.5*(A(0,1)+A(1,0));
    const Scalar s02 = 0.5*(A(0,2)+A(2,0));
    const Scalar s12 = 0.5*(A(1,2)+A(2,1));

    return Tensor<Scalar>(
        s00, s01, s02,
        s01, s11, s12,
        s02, s12, s22);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  skew(const Tensor<Scalar> & A)
  {
    const Scalar s01 = 0.5*(A(0,1)-A(1,0));
    const Scalar s02 = 0.5*(A(0,2)-A(2,0));
    const Scalar s12 = 0.5*(A(1,2)-A(2,1));

    return Tensor<Scalar>(
         0.0,  s01,  s02,
        -s01,  0.0,  s12,
        -s02, -s12,  0.0);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  skew(const Vector<Scalar> & u)
  {
    return Tensor<Scalar>(
         0.0, -u(2),  u(1),
        u(2),   0.0, -u(0),
       -u(1),  u(0),   0.0);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  inverse(const Tensor<Scalar> & A)
  {
    const Scalar d = det(A);
    TEST_FOR_EXCEPT_MSG(d == 0.0, "Attempted to invert a singular Tensor.");
    Tensor<Scalar> B(
        -A(1,2)*A(2,1) + A(1,1)*A(2,2),
         A(0,2)*A(2,1) - A(0,1)*A(2,2),
        -A(0,2)*A(1,1) + A(0,1)*A(1,2),
         A(1,2)*A(2,0) - A(1,0)*A(2,2),
        -A(0,2)*A(2,0) + A(0,0)*A(2,2),
         A(0,2)*A(1,0) - A(0,0)*A(1,2),
        -A(1,1)*A(2,0) + A(1,0)*A(2,1),
         A(0,1)*A(2,0) - A(0,0)*A(2,1),
        -A(0,1)*A(1,0) + A(0,0)*A(1,1)
    );
    return (1.0 / d) * B;
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  det(const Tensor<Scalar> & A)
  {
    return
        -A(0,2)*A(1,1)*A(2,0) + A(0,1)*A(1,2)*A(2,0) +
         A(0,2)*A(1,0)*A(2,1) - A(0,0)*A(1,2)*A(2,1) -
         A(0,1)*A(1,0)*A(2,2) + A(0,0)*A(1,1)*A(2,2);
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  trace(const Tensor<Scalar> & A)
  {
    return A(0,0) + A(1,1) + A(2,2);
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  I1(const Tensor<Scalar> & A)
  {
    return trace(A);
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  I2(const Tensor<Scalar> & A)
  {
    const Scalar trA = trace(A);

    return 0.5 * (trA*trA - A(0,0)*A(0,0) - A(1,1)*A(1,1) - A(2,2)*A(2,2)) -
        A(0,1)*A(1,0) - A(0,2)*A(2,0) - A(1,2)*A(2,1);
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar
  I3(const Tensor<Scalar> & A)
  {
    return det(A);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  exp(const Tensor<Scalar> & A)
  {
    const Index maxNumIter = 128;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();

    Index k = 0;
    const Tensor<Scalar> term = identity<Scalar>();

    // Relative error taken wrt to the first term, which is I and norm = 1
    Scalar relError = 1.0;

    Tensor<Scalar> B = term;

    while (relError > tol && k < maxNumIter) {
      term = (1.0 / (k + 1.0)) * term * A;
      B = B + term;
      relError = norm_1(term);
      ++k;
    }

    return B;
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  log(const Tensor<Scalar> & A)
  {
    const Index maxNumIter = 128;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();

    Index k = 1;
    const Scalar normA = norm_1(A);
    const Tensor<Scalar> Am1 = A - identity<Scalar>();
    Tensor<Scalar> term = Am1;
    Scalar normTerm = norm_1(term);
    Scalar relError = normTerm / normA;

    Tensor<Scalar> B = term;

    while (relError > tol && k < maxNumIter) {
      term = - (k / (k + 1.0)) * term * Am1;
      B = B + term;
      normTerm = norm_1(term);
      relError = normTerm / normA;
      ++k;
    }

    return B;
  }

  //
  //
  //
  template<typename Scalar>
  inline const Scalar &
  Tensor3<Scalar>::operator()(const Index i, const Index j, const Index k) const
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    assert(k < MaxDim);
    return e[i][j][k];
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar &
  Tensor3<Scalar>::operator()(const Index i, const Index j, const Index k)
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    assert(k < MaxDim);
    return e[i][j][k];
  }

  //
  //
  //
  template<typename Scalar>
  inline const Scalar &
  Tensor4<Scalar>::operator()(
      const Index i, const Index j, const Index k, const Index l) const
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    assert(k < MaxDim);
    assert(l < MaxDim);
    return e[i][j][k][l];
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar &
  Tensor4<Scalar>::operator()(
      const Index i, const Index j, const Index k, const Index l)
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    assert(k < MaxDim);
    assert(l < MaxDim);
    return e[i][j][k][l];
  }

} // namespace LCM

#endif // LCM_Tensor_i_cc
