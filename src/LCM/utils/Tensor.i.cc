//
//
//
#if !defined(LCM_Tensor_i_cc)

#include <cassert>
#include <cmath>
#include <limits>

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
  Vector<Scalar>::Vector(const Scalar & s)
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
  Vector<Scalar>::Vector(const Scalar & s0,
     const Scalar & s1,
     const Scalar & s2)
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
  Vector<Scalar>::Vector(const Scalar s[MaxDim])
  {
    e[0] = s[0];
    e[1] = s[1];
    e[2] = s[2];

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
  inline const Scalar &
  Vector<Scalar>::operator[](const Index i) const
  {
    assert(i < MaxDim);
    return e[i];
  }

  //
  //
  //
  template<typename Scalar>
  inline Scalar &
  Vector<Scalar>::operator[](const Index i)
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
  Vector<Scalar>::operator+(const Vector<Scalar> & v) const
  {
    Vector<Scalar> s;

    s.e[0] = e[0] + v.e[0];
    s.e[1] = e[1] + v.e[1];
    s.e[2] = e[2] + v.e[2];

    return s;
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  Vector<Scalar>::operator-(const Vector<Scalar> & v) const
  {
    Vector<Scalar> s;

    s.e[0] = e[0] - v.e[0];
    s.e[1] = e[1] - v.e[1];
    s.e[2] = e[2] - v.e[2];

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
  Vector<Scalar>::operator*(const Vector<Scalar> & v) const
  {
    return dot(*this, v);
  }

  //
  //
  //
  template<typename Scalar>
  inline Vector<Scalar>
  Vector<Scalar>::operator*(const Scalar & s) const
  {
    Vector<Scalar> v;

    v.e[0] = s * e[0];
    v.e[1] = s * e[1];
    v.e[2] = s * e[2];

    return v;
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
  norm(const Vector<Scalar> & v)
  {
    return sqrt(dot(v, v));
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
  Tensor<Scalar>::Tensor(const Scalar & s)
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
      const Scalar & s00, const Scalar & s01, const Scalar & s02,
      const Scalar & s10, const Scalar & s11, const Scalar & s12,
      const Scalar & s20, const Scalar & s21, const Scalar & s22)
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
  Tensor<Scalar>::Tensor(const Scalar s[MaxDim][MaxDim])
  {
    e[0][0] = s[0][0];
    e[0][1] = s[0][1];
    e[0][2] = s[0][2];

    e[1][0] = s[1][0];
    e[1][1] = s[1][1];
    e[1][2] = s[1][2];

    e[2][0] = s[2][0];
    e[2][1] = s[2][1];
    e[2][2] = s[2][2];

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
  inline Tensor<Scalar>
  Tensor<Scalar>::operator+(const Tensor<Scalar> & A) const
  {
    Tensor<Scalar> S;

    S.e[0][0] = e[0][0] + A.e[0][0];
    S.e[0][1] = e[0][1] + A.e[0][1];
    S.e[0][2] = e[0][2] + A.e[0][2];

    S.e[1][0] = e[1][0] + A.e[1][0];
    S.e[1][1] = e[1][1] + A.e[1][1];
    S.e[1][2] = e[1][2] + A.e[1][2];

    S.e[2][0] = e[2][0] + A.e[2][0];
    S.e[2][1] = e[2][1] + A.e[2][1];
    S.e[2][2] = e[2][2] + A.e[2][2];

    return S;
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  Tensor<Scalar>::operator-(const Tensor<Scalar> & A) const
  {
    Tensor<Scalar> S;

    S.e[0][0] = e[0][0] - A.e[0][0];
    S.e[0][1] = e[0][1] - A.e[0][1];
    S.e[0][2] = e[0][2] - A.e[0][2];

    S.e[1][0] = e[1][0] - A.e[1][0];
    S.e[1][1] = e[1][1] - A.e[1][1];
    S.e[1][2] = e[1][2] - A.e[1][2];

    S.e[2][0] = e[2][0] - A.e[2][0];
    S.e[2][1] = e[2][1] - A.e[2][1];
    S.e[2][2] = e[2][2] - A.e[2][2];

    return S;
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
  Tensor<Scalar>::operator*(const Tensor<Scalar> & A) const
  {
    return dot(*this, A);
  }

  //
  //
  //
  template<typename Scalar>
  inline Tensor<Scalar>
  Tensor<Scalar>::operator*(const Scalar & s) const
  {
    Tensor<Scalar> A;

    A.e[0][0] = s * e[0][0];
    A.e[0][1] = s * e[0][1];
    A.e[0][2] = s * e[0][2];

    A.e[1][0] = s * e[1][0];
    A.e[1][1] = s * e[1][1];
    A.e[1][2] = s * e[1][2];

    A.e[2][0] = s * e[2][0];
    A.e[2][1] = s * e[2][1];
    A.e[2][2] = s * e[2][2];

    return A;
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

    v(0) = A(0,0) * u(0) + A(0,1) * u(1) + A(0,2) * u(2);
    v(1) = A(1,0) * u(0) + A(1,1) * u(1) + A(1,2) * u(2);
    v(2) = A(2,0) * u(0) + A(2,1) * u(1) + A(2,2) * u(2);

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

    v(0) = A(0,0) * u(0) + A(1,0) * u(1) + A(2,0) * u(2);
    v(1) = A(0,1) * u(0) + A(1,1) * u(1) + A(2,1) * u(2);
    v(2) = A(0,2) * u(0) + A(1,2) * u(1) + A(2,2) * u(2);

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
  //
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
  eye()
  {
    return Tensor<Scalar>(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
  }


} // namespace LCM

#endif // LCM_Tensor_i_cc
