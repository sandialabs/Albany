//
//
//
#if !defined(LCM_Tensor_i_cc)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

//#include "Teuchos_TestForException.hpp"

namespace LCM {

  //
  // Vector construction/destruction
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::Vector()
  {
    e[0] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[1] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[2] = std::numeric_limits<ScalarT>::quiet_NaN();

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::Vector(const ScalarT s)
  {
    e[0] = s;
    e[1] = s;
    e[2] = s;

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::Vector(const ScalarT s0, const ScalarT s1, const ScalarT s2)
  {
    e[0] = s0;
    e[1] = s1;
    e[2] = s2;

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::Vector(Vector<ScalarT> const & V)
  {
    e[0] = V.e[0];
    e[1] = V.e[1];
    e[2] = V.e[2];

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::~Vector()
  {
    return;
  }

  //
  // Vector utilities
  //
  template<typename ScalarT>
  inline void
  Vector<ScalarT>::clear()
  {
    e[0] = 0.0;
    e[1] = 0.0;
    e[2] = 0.0;

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  inline const ScalarT &
  Vector<ScalarT>::operator()(const Index i) const
  {
    assert(i < MaxDim);
    return e[i];
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT &
  Vector<ScalarT>::operator()(const Index i)
  {
    assert(i < MaxDim);
    return e[i];
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT> &
  Vector<ScalarT>::operator=(Vector<ScalarT> const & v)
  {
    if (this != &v) {
      e[0] = v.e[0];
      e[1] = v.e[1];
      e[2] = v.e[2];
    }
    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator+(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    Vector<ScalarT> s;

    s(0) = u(0) + v(0);
    s(1) = u(1) + v(1);
    s(2) = u(2) + v(2);

    return s;
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator-(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    Vector<ScalarT> s;

    s(0) = u(0) - v(0);
    s(1) = u(1) - v(1);
    s(2) = u(2) - v(2);

    return s;
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT> &
  Vector<ScalarT>::operator+=(Vector<ScalarT> const & v)
  {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT> &
  Vector<ScalarT>::operator-=(Vector<ScalarT> const & v)
  {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  operator*(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return dot(u, v);
  }

  //
  //
  //
  template<typename ScalarT>
  inline bool
  operator==(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return u(0)==v(0) && u(1)==v(1) && u(2)==v(2);
  }

  //
  //
  //
  template<typename ScalarT>
  inline bool
  operator!=(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return !(u==v);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(const ScalarT s, Vector<ScalarT> const & u)
  {
    return Vector<ScalarT>(s*u(0), s*u(1), s*u(2));
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, const ScalarT s)
  {
    return s * u;
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  dot(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return u(0)*v(0) + u(1)*v(1) + u(2)*v(2);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  cross(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    Vector<ScalarT> w;

    w(0) = u(1)*v(2) - u(2)*v(1);
    w(1) = u(2)*v(0) - u(0)*v(2);
    w(2) = u(0)*v(1) - u(1)*v(0);

    return w;
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  norm(Vector<ScalarT> const & u)
  {
    return sqrt(u(0)*u(0) + u(1)*u(1) + u(2)*u(2));
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  norm_1(Vector<ScalarT> const & u)
  {
    return std::fabs(u(0)) + std::fabs(u(1)) + std::fabs(u(2));
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  norm_infinity(Vector<ScalarT> const & u)
  {
    return std::max(std::max(std::fabs(u(0)),std::fabs(u(1))),std::fabs(u(2)));
  }

  //
  // Second order tensor construction/destruction
  //
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::Tensor()
  {
    e[0][0] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[0][1] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[0][2] = std::numeric_limits<ScalarT>::quiet_NaN();

    e[1][0] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[1][1] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[1][2] = std::numeric_limits<ScalarT>::quiet_NaN();

    e[2][0] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[2][1] = std::numeric_limits<ScalarT>::quiet_NaN();
    e[2][2] = std::numeric_limits<ScalarT>::quiet_NaN();

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::Tensor(const ScalarT s)
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
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::Tensor(
      const ScalarT s00, const ScalarT s01, const ScalarT s02,
      const ScalarT s10, const ScalarT s11, const ScalarT s12,
      const ScalarT s20, const ScalarT s21, const ScalarT s22)
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
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::Tensor(Tensor<ScalarT> const & A)
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
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::~Tensor()
  {
    return;
  }

  //
  // Tensor utilities
  //
  template<typename ScalarT>
  inline void
  Tensor<ScalarT>::clear()
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
  template<typename ScalarT>
  inline const ScalarT &
  Tensor<ScalarT>::operator()(const Index i, const Index j) const
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    return e[i][j];
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT &
  Tensor<ScalarT>::operator()(const Index i, const Index j)
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    return e[i][j];
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT> &
  Tensor<ScalarT>::operator=(Tensor<ScalarT> const & A)
  {
    if (this != &A) {
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
  template<typename ScalarT>
  inline Tensor<ScalarT> &
  Tensor<ScalarT>::operator+=(Tensor<ScalarT> const & A)
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
  template<typename ScalarT>
  inline Tensor<ScalarT> &
  Tensor<ScalarT>::operator-=(Tensor<ScalarT> const & A)
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
  template<typename ScalarT>
  inline Tensor<ScalarT>
  operator+(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    Tensor<ScalarT> S;

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
  template<typename ScalarT>
  inline Tensor<ScalarT>
  operator-(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    Tensor<ScalarT> S;

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
  template<typename ScalarT>
  inline Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    return dot(A, B);
  }

  //
  //
  //
  template<typename ScalarT>
  inline bool
  operator==(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    return
      A(0,0)==B(0,0) && A(0,1)==B(0,1) && A(0,2)==B(0,2) &&
      A(1,0)==B(1,0) && A(1,1)==B(1,1) && A(1,2)==B(1,2) &&
      A(2,0)==B(2,0) && A(2,1)==B(2,1) && A(2,2)==B(2,2);
  }

  //
  //
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    return !(A==B);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  operator*(const ScalarT s, Tensor<ScalarT> const & A)
  {
    return Tensor<ScalarT>(
        s*A(0,0), s*A(0,1), s*A(0,2),
        s*A(1,0), s*A(1,1), s*A(1,2),
        s*A(2,0), s*A(2,1), s*A(2,2));
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, const ScalarT s)
  {
    return s * A;
  }

  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(Tensor<ScalarT> const & A, Vector<ScalarT> const & u)
  {
    return dot(A,u);
  }


  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, Tensor<ScalarT> const & A)
  {
    return dot(u,A);
  }


  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  dot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    Tensor<ScalarT> C;

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
  template<typename ScalarT>
  inline Vector<ScalarT>
  dot(Tensor<ScalarT> const & A, Vector<ScalarT> const & u)
  {
    Vector<ScalarT> v;

    v(0) = A(0,0)*u(0) + A(0,1)*u(1) + A(0,2)*u(2);
    v(1) = A(1,0)*u(0) + A(1,1)*u(1) + A(1,2)*u(2);
    v(2) = A(2,0)*u(0) + A(2,1)*u(1) + A(2,2)*u(2);

    return v;
  }


  //
  //
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  dot(Vector<ScalarT> const & u, Tensor<ScalarT> const & A)
  {
    Vector<ScalarT> v;

    v(0) = A(0,0)*u(0) + A(1,0)*u(1) + A(2,0)*u(2);
    v(1) = A(0,1)*u(0) + A(1,1)*u(1) + A(2,1)*u(2);
    v(2) = A(0,2)*u(0) + A(1,2)*u(1) + A(2,2)*u(2);

    return v;
  }


  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  dotdot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    ScalarT s = 0.0;

    s+= A(0,0)*B(0,0) + A(0,1)*B(0,1) + A(0,2)*B(0,2);
    s+= A(1,0)*B(1,0) + A(1,1)*B(1,1) + A(1,2)*B(1,2);
    s+= A(2,0)*B(2,0) + A(2,1)*B(2,1) + A(2,2)*B(2,2);

    return s;
  }


  //
  // Frobenius norm
  //
  template<typename ScalarT>
  inline ScalarT
  norm(Tensor<ScalarT> const & A)
  {
    ScalarT s = 0.0;

    s+= A(0,0)*A(0,0) + A(0,1)*A(0,1) + A(0,2)*A(0,2);
    s+= A(1,0)*A(1,0) + A(1,1)*A(1,1) + A(1,2)*A(1,2);
    s+= A(2,0)*A(2,0) + A(2,1)*A(2,1) + A(2,2)*A(2,2);

    return sqrt(s);
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  norm_1(Tensor<ScalarT> const & A)
  {
    ScalarT s0 = std::fabs(A(0,0)) + std::fabs(A(1,0)) + std::fabs(A(2,0));
    ScalarT s1 = std::fabs(A(0,1)) + std::fabs(A(1,1)) + std::fabs(A(2,1));
    ScalarT s2 = std::fabs(A(0,2)) + std::fabs(A(1,2)) + std::fabs(A(2,2));

    return std::max(std::max(s0,s1),s2);
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  norm_infinity(Tensor<ScalarT> const & A)
  {
    ScalarT s0 = std::fabs(A(0,0)) + std::fabs(A(0,1)) + std::fabs(A(0,2));
    ScalarT s1 = std::fabs(A(1,0)) + std::fabs(A(1,1)) + std::fabs(A(1,2));
    ScalarT s2 = std::fabs(A(2,0)) + std::fabs(A(2,1)) + std::fabs(A(2,2));

    return std::max(std::max(s0,s1),s2);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  dyad(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    Tensor<ScalarT> A;

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
  template<typename ScalarT>
  inline Tensor<ScalarT>
  bun(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return dyad(u,v);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  tensor(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return dyad(u,v);
  }

  //
  //
  //
  template<typename ScalarT>
  inline const Tensor<ScalarT>
  identity()
  {
    return Tensor<ScalarT>(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
  }

  //
  //
  //
  template<typename ScalarT>
  inline const Tensor<ScalarT>
  eye()
  {
    return identity<ScalarT>();
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  transpose(Tensor<ScalarT> const & A)
  {
    return Tensor<ScalarT>(
        A(0,0),A(1,0),A(2,0),
        A(0,1),A(1,1),A(2,1),
        A(0,2),A(1,2),A(2,2));
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  symm(Tensor<ScalarT> const & A)
  {
    const ScalarT s00 = A(0,0);
    const ScalarT s11 = A(1,1);
    const ScalarT s22 = A(2,2);

    const ScalarT s01 = 0.5*(A(0,1)+A(1,0));
    const ScalarT s02 = 0.5*(A(0,2)+A(2,0));
    const ScalarT s12 = 0.5*(A(1,2)+A(2,1));

    return Tensor<ScalarT>(
        s00, s01, s02,
        s01, s11, s12,
        s02, s12, s22);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  skew(Tensor<ScalarT> const & A)
  {
    const ScalarT s01 = 0.5*(A(0,1)-A(1,0));
    const ScalarT s02 = 0.5*(A(0,2)-A(2,0));
    const ScalarT s12 = 0.5*(A(1,2)-A(2,1));

    return Tensor<ScalarT>(
         0.0,  s01,  s02,
        -s01,  0.0,  s12,
        -s02, -s12,  0.0);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  skew(Vector<ScalarT> const & u)
  {
    return Tensor<ScalarT>(
         0.0, -u(2),  u(1),
        u(2),   0.0, -u(0),
       -u(1),  u(0),   0.0);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  inverse(Tensor<ScalarT> const & A)
  {
    const ScalarT d = det(A);
    //TEST_FOR_EXCEPT_MSG(d == 0.0, "Attempted to invert a singular Tensor.");
    Tensor<ScalarT> B(
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
  template<typename ScalarT>
  inline ScalarT
  det(Tensor<ScalarT> const & A)
  {
    return
        -A(0,2)*A(1,1)*A(2,0) + A(0,1)*A(1,2)*A(2,0) +
         A(0,2)*A(1,0)*A(2,1) - A(0,0)*A(1,2)*A(2,1) -
         A(0,1)*A(1,0)*A(2,2) + A(0,0)*A(1,1)*A(2,2);
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  trace(Tensor<ScalarT> const & A)
  {
    return A(0,0) + A(1,1) + A(2,2);
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  I1(Tensor<ScalarT> const & A)
  {
    return trace(A);
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  I2(Tensor<ScalarT> const & A)
  {
    const ScalarT trA = trace(A);

    return 0.5 * (trA*trA - A(0,0)*A(0,0) - A(1,1)*A(1,1) - A(2,2)*A(2,2)) -
        A(0,1)*A(1,0) - A(0,2)*A(2,0) - A(1,2)*A(2,1);
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT
  I3(Tensor<ScalarT> const & A)
  {
    return det(A);
  }

  //
  //
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  exp(Tensor<ScalarT> const & A)
  {
    const Index maxNumIter = 128;
    const ScalarT tol = std::numeric_limits<ScalarT>::epsilon();

    Index k = 0;
    const Tensor<ScalarT> term = identity<ScalarT>();

    // Relative error taken wrt to the first term, which is I and norm = 1
    ScalarT relError = 1.0;

    Tensor<ScalarT> B = term;

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
  template<typename ScalarT>
  inline Tensor<ScalarT>
  log(Tensor<ScalarT> const & A)
  {
    const Index maxNumIter = 128;
    const ScalarT tol = std::numeric_limits<ScalarT>::epsilon();

    Index k = 1;
    const ScalarT normA = norm_1(A);
    const Tensor<ScalarT> Am1 = A - identity<ScalarT>();
    Tensor<ScalarT> term = Am1;
    ScalarT normTerm = norm_1(term);
    ScalarT relError = normTerm / normA;

    Tensor<ScalarT> B = term;

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
  template<typename ScalarT>
  inline const ScalarT &
  Tensor3<ScalarT>::operator()(const Index i, const Index j, const Index k) const
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    assert(k < MaxDim);
    return e[i][j][k];
  }

  //
  //
  //
  template<typename ScalarT>
  inline ScalarT &
  Tensor3<ScalarT>::operator()(const Index i, const Index j, const Index k)
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    assert(k < MaxDim);
    return e[i][j][k];
  }

  //
  //
  //
  template<typename ScalarT>
  inline const ScalarT &
  Tensor4<ScalarT>::operator()(
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
  template<typename ScalarT>
  inline ScalarT &
  Tensor4<ScalarT>::operator()(
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
