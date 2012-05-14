///
/// \file Tensor.i.cc
/// First cut of LCM small tensor utilities. Inline functions.
/// \author Alejandro Mota
/// \author Jake Ostien
///
#if !defined(LCM_Tensor_i_cc)
#define LCM_Tensor_i_cc

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <Sacado_MathFunctions.hpp>

namespace LCM {

  //
  // Default constructor that initializes to NaNs
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
  // Create vector from a scalar
  // \param s all components are set equal to this value
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
  // Create vector specifying components
  // \param s0
  // \param s1
  // \param s2 are the vector components in the canonical basis
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
  // Create vector from array - const version
  // \param dataPtr
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::Vector(const ScalarT * dataPtr)
  {
    assert(dataPtr != NULL);
    e[0] = dataPtr[0];
    e[1] = dataPtr[1];
    e[2] = dataPtr[2];

    return;
  }

  //
  // Create vector from array
  // \param dataPtr
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::Vector(ScalarT * dataPtr)
  {
    assert(dataPtr != NULL);
    e[0] = dataPtr[0];
    e[1] = dataPtr[1];
    e[2] = dataPtr[2];

    return;
  }

  //
  // Copy constructor
  // \param v the values of its componets are copied to the new vector
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::Vector(Vector<ScalarT> const & v)
  {
    e[0] = v.e[0];
    e[1] = v.e[1];
    e[2] = v.e[2];

    return;
  }

  //
  // Simple destructor
  //
  template<typename ScalarT>
  inline
  Vector<ScalarT>::~Vector()
  {
    return;
  }

  //
  // Indexing for constant vector
  // \param i the index
  //
  template<typename ScalarT>
  inline const ScalarT &
  Vector<ScalarT>::operator()(const Index i) const
  {
    assert(i < MaxDim);
    return e[i];
  }

  //
  // Vector indexing
  // \param i the index
  //
  template<typename ScalarT>
  inline ScalarT &
  Vector<ScalarT>::operator()(const Index i)
  {
    assert(i < MaxDim);
    return e[i];
  }

  //
  // Copy assignment
  // \param v the values of its componets are copied to this vector
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
  // Vector increment
  // \param v added to currrent vector
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
  // Vector decrement
  // \param v substracted from current vector
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
  // Fill with zeros
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
  // Vector addition
  // \param u
  // \param v the operands
  // \return \f$ u + v \f$
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
  // Vector substraction
  // \param u
  // \param v the operands
  // \return \f$ u - v \f$
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
  // Vector minus
  // \param u
  // \return \f$ -u \f$
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator-(Vector<ScalarT> const & u)
  {
    return Vector<ScalarT>(-u(0), -u(1), -u(2));
  }

  //
  // Vector dot product
  // \param u
  // \param v the operands
  // \return \f$ u \cdot v \f$
  //
  template<typename ScalarT>
  inline ScalarT
  operator*(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return dot(u, v);
  }

  //
  // Vector equality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \equiv v \f$
  //
  template<typename ScalarT>
  inline bool
  operator==(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return u(0)==v(0) && u(1)==v(1) && u(2)==v(2);
  }

  //
  // Vector inequality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \neq v \f$
  //
  template<typename ScalarT>
  inline bool
  operator!=(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return !(u==v);
  }

  //
  // Scalar vector product
  // \param s scalar factor
  // \param u vector factor
  // \return \f$ s u \f$
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(const ScalarT s, Vector<ScalarT> const & u)
  {
    return Vector<ScalarT>(s*u(0), s*u(1), s*u(2));
  }

  //
  // Vector scalar product
  // \param u vector factor
  // \param s scalar factor
  // \return \f$ s u \f$
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, const ScalarT s)
  {
    return s * u;
  }

  //
  // Vector scalar division
  // \param u vector
  // \param s scalar that divides each component of vector
  // \return \f$ u / s \f$
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator/(Vector<ScalarT> const & u, const ScalarT s)
  {
    return Vector<ScalarT>(u(0)/s, u(1)/s, u(2)/s);
  }

  //
  // Vector dot product
  // \param u
  // \param v operands
  // \return \f$ u \cdot v \f$
  //
  template<typename ScalarT>
  inline ScalarT
  dot(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return u(0)*v(0) + u(1)*v(1) + u(2)*v(2);
  }

  //
  // Cross product
  // \param u
  // \param v operands
  // \return \f$ u \times v \f$
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
  // Vector 2-norm
  // \return \f$ \sqrt{u \cdot u} \f$
  //
  template<typename ScalarT>
  inline ScalarT
  norm(Vector<ScalarT> const & u)
  {
    return sqrt(u(0)*u(0) + u(1)*u(1) + u(2)*u(2));
  }

  //
  // Vector 1-norm
  // \return \f$ |u_0|+|u_1|+|u_2| \f$
  //
  template<typename ScalarT>
  inline ScalarT
  norm_1(Vector<ScalarT> const & u)
  {
    return std::fabs(u(0)) + std::fabs(u(1)) + std::fabs(u(2));
  }

  //
  // Vector infinity-norm
  // \return \f$ \max(|u_0|,|u_1|,|u_2|) \f$
  //
  template<typename ScalarT>
  inline ScalarT
  norm_infinity(Vector<ScalarT> const & u)
  {
    return std::max(std::max(std::fabs(u(0)),std::fabs(u(1))),std::fabs(u(2)));
  }

  //
  // Default constructor that initializes to NaNs
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
  // Create tensor from a scalar
  // \param s all components are set equal to this value
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
  // Create tensor specifying components
  // The parameters are the components in the canonical basis
  // \param s00 ...
  // \param s22
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
  // Create tensor from array - const version
  // \param dataPtr
  //
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::Tensor(const ScalarT * dataPtr)
  {
    assert(dataPtr != NULL);
    e[0][0] = dataPtr[0];
    e[0][1] = dataPtr[1];
    e[0][2] = dataPtr[2];

    e[1][0] = dataPtr[3];
    e[1][1] = dataPtr[4];
    e[1][2] = dataPtr[5];

    e[2][0] = dataPtr[6];
    e[2][1] = dataPtr[7];
    e[2][2] = dataPtr[8];

    return;
  }

  //
  // Create vector from array
  // \param dataPtr
  //
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::Tensor(ScalarT * dataPtr)
  {
    assert(dataPtr != NULL);
    e[0][0] = dataPtr[0];
    e[0][1] = dataPtr[1];
    e[0][2] = dataPtr[2];

    e[1][0] = dataPtr[3];
    e[1][1] = dataPtr[4];
    e[1][2] = dataPtr[5];

    e[2][0] = dataPtr[6];
    e[2][1] = dataPtr[7];
    e[2][2] = dataPtr[8];

    return;
  }

  //
  // Copy constructor
  // \param A the values of its componets are copied to the new tensor
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
  // Simple destructor
  //
  template<typename ScalarT>
  inline
  Tensor<ScalarT>::~Tensor()
  {
    return;
  }

  //
  // Indexing for constant tensor
  // \param i index
  // \param j index
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
  // Tensor indexing
  // \param i index
  // \param j index
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
  // Copy assignment
  // \param A the values of its componets are copied to this tensor
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
  // Tensor increment
  // \param A added to current tensor
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
  // Tensor decrement
  // \param A substracted from current tensor
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
  // Fill with zeros
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
  // Tensor addition
  // \return \f$ A + B \f$
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
  // Tensor substraction
  // \return \f$ A - B \f$
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
  // Tensor minus
  // \return \f$ -A \f$
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  operator-(Tensor<ScalarT> const & A)
  {
    return Tensor<ScalarT>(
        -A(0,0),-A(0,1),-A(0,2),
        -A(1,0),-A(1,1),-A(1,2),
        -A(2,0),-A(2,1),-A(2,2));
  }

  //
  // Tensor equality
  // Tested by components
  // \return \f$ A \equiv B \f$
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
  // Tensor inequality
  // Tested by components
  // \return \f$ A \neq B \f$
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    return !(A == B);
  }

  //
  // Scalar tensor product
  // \param s scalar
  // \param A tensor
  // \return \f$ s A \f$
  //
  template<typename ScalarT, typename T>
  inline Tensor<ScalarT>
  operator*(const T s, Tensor<ScalarT> const & A)
  {
    return Tensor<ScalarT>(
        s*A(0,0), s*A(0,1), s*A(0,2),
        s*A(1,0), s*A(1,1), s*A(1,2),
        s*A(2,0), s*A(2,1), s*A(2,2));
  }

  //
  // Tensor scalar product
  // \param A tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename ScalarT, typename T>
  inline Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, const T s)
  {
    return s * A;
  }

  //
  // Tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(Tensor<ScalarT> const & A, Vector<ScalarT> const & u)
  {
    return dot(A,u);
  }

  //
  // Vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename ScalarT>
  inline Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, Tensor<ScalarT> const & A)
  {
    return dot(u,A);
  }

  //
  // Tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
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
  // Vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
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
  // Tensor dot product C = A B
  // \return \f$ A \cdot B \f$
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    return dot(A, B);
  }

  //
  // Tensor tensor product C = A B
  // \param A tensor
  // \param B tensor
  // \return a tensor \f$ A \cdot B \f$
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
  // Tensor tensor double dot product (contraction)
  // \param A tensor
  // \param B tensor
  // \return a scalar \f$ A : B \f$
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
  // Tensor Frobenius norm
  // \return \f$ \sqrt{A:A} \f$
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
  // Tensor 1-norm
  // \return \f$ \max_{j \in {0,1,2}}\Sigma_{i=0}^2 |A_{ij}| \f$
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
  // Tensor infinity-norm
  // \return \f$ \max_{i \in {0,1,2}}\Sigma_{j=0}^2 |A_{ij}| \f$
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
  // Dyad
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
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
  // Bun operator, just for Jay
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  bun(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return dyad(u,v);
  }

  //
  // Tensor product
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  tensor(Vector<ScalarT> const & u, Vector<ScalarT> const & v)
  {
    return dyad(u,v);
  }

  //
  // Zero 2nd-order tensor
  // All components are zero
  //
  template<typename ScalarT>
  inline const Tensor<ScalarT>
  zero()
  {
    return Tensor<ScalarT>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
  }

  //
  // 2nd-order identity tensor
  //
  template<typename ScalarT>
  inline const Tensor<ScalarT>
  identity()
  {
    return Tensor<ScalarT>(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
  }

  //
  // 2nd-order identity tensor, à la Matlab
  //
  template<typename ScalarT>
  inline const Tensor<ScalarT>
  eye()
  {
    return identity<ScalarT>();
  }

  //
  // 2nd-order tensor transpose
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
  // Symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A + A^T) \f$
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
  // Skew symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A - A^T) \f$
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
  // Skew symmetric 2nd-order tensor from vector
  // \param u vector
  // \return \f$ {{0, -u_2, u_1}, {u_2, 0, -u_0}, {-u_1, u+0, 0}} \f$
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
  // Volumetric part of 2nd-order tensor
  // \return \f$ \frac{1}{3} \mathrm{tr}\:(A) I \f$
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  vol(Tensor<ScalarT> const & A)
  {
    const ScalarT tr = (1.0/3.0)*trace(A);

    return tr * eye<ScalarT>();
  }

  //
  // Deviatoric part of 2nd-order tensor
  // \return \f$ A - vol(A) \f$
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  dev(Tensor<ScalarT> const & A)
  {
    return A - vol(A);
  }

  //
  // 2nd-order tensor inverse
  // \param A nonsingular tensor
  // \return \f$ A^{-1} \f$
  //
  template<typename ScalarT>
  inline Tensor<ScalarT>
  inverse(Tensor<ScalarT> const & A)
  {
    const ScalarT d = det(A);
    assert(d != 0.0);
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
    return ScalarT(1.0 / d) * B;
  }

  //
  // Determinant
  // \param A tensor
  // \return \f$ \det A \f$
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
  // Trace
  // \param A tensor
  // \return \f$ A:I \f$
  //
  template<typename ScalarT>
  inline ScalarT
  trace(Tensor<ScalarT> const & A)
  {
    return A(0,0) + A(1,1) + A(2,2);
  }

  //
  // First invariant, trace
  // \param A tensor
  // \return \f$ I_A = A:I \f$
  //
  template<typename ScalarT>
  inline ScalarT
  I1(Tensor<ScalarT> const & A)
  {
    return trace(A);
  }

  //
  // Second invariant
  // \param A tensor
  // \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
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
  // Third invariant
  // \param A tensor
  // \return \f$ III_A = \det A \f$
  //
  template<typename ScalarT>
  inline ScalarT
  I3(Tensor<ScalarT> const & A)
  {
    return det(A);
  }

  //
  // Indexing for constant 3rd order tensor
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename ScalarT>
  inline const ScalarT &
  Tensor3<ScalarT>::operator()(
      const Index i,
      const Index j,
      const Index k) const
  {
    assert(i < MaxDim);
    assert(j < MaxDim);
    assert(k < MaxDim);
    return e[i][j][k];
  }

  //
  // 3rd-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
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
  // Indexing for constant 4th order tensor
  // \param i index
  // \param j index
  // \param k index
  // \param l index
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
  // 4th-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  // \param l index
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
