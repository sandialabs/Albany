//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Tensor_h)
#define LCM_Tensor_h

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <iostream>
#include <limits>
#include <vector>

#include <boost/tuple/tuple.hpp>

#include "Sacado.hpp"

namespace LCM {

  ///
  /// Indexing type
  ///
  typedef unsigned int Index;

  ///
  /// Sign function
  ///
  template<typename T>
  int
  sgn(T const & s);

  ///
  /// NaN function. Necessary to choose the proper underlying NaN
  /// for non-floating-point types.
  /// Assumption: non-floating-point types have a typedef that
  /// determines the underlying floating-point type.
  ///
  template<typename T>
  typename Sacado::ScalarType<T>::type
  not_a_number();

  ///
  /// Machine epsilon function. Necessary to choose the proper underlying
  /// machine epsilon for non-floating-point types.
  /// Assumption: non-floating-point types have a typedef that
  /// determines the underlying floating-point type.
  ///
  template<typename T>
  typename Sacado::ScalarType<T>::type
  machine_epsilon();

  ///
  /// Vector in R^N.
  ///
  template<typename T>
  class Vector
  {
  public:

    ///
    /// Default constructor
    ///
    Vector();

    ///
    /// Constructor that initializes to NaNs
    /// \param N dimension
    ///
    Vector(const Index N);

    ///
    /// Create vector from a scalar
    /// \param N dimension
    /// \param s all components are set equal to this value
    ///
    // Vector(const Index N, T const & s);

    ///
    /// Create vector specifying components
    /// \param N dimension
    /// \param s0, s1 are the vector components in the R^2 canonical basis
    ///
    Vector(T const & s0, T const & s1);

    ///
    /// Create vector specifying components
    /// \param N dimension
    /// \param s0, s1, s2 are the vector components in the R^3 canonical basis
    ///
    Vector(T const & s0, T const & s1, T const & s2);

    ///
    /// Create vector from array - const version
    /// \param N dimension
    /// \param data_ptr pointer into the array
    ///
    Vector(const Index N, T const * data_ptr);

    ///
    /// Create vector from array
    /// \param N dimension
    /// \param data_ptr pointer into the array
    ///
    Vector(const Index N, T * data_ptr);

    ///
    /// Copy constructor
    /// \param v the values of its componets are copied to the new vector
    ///
    Vector(Vector<T> const & v);

    ///
    /// Simple destructor
    ///
    ~Vector();

    ///
    /// Indexing for constant vector
    /// \param i the index
    ///
    const T &
    operator()(const Index i) const;

    ///
    /// Vector indexing
    /// \param i the index
    ///
    T &
    operator()(const Index i);

    ///
    /// \return dimension
    ///
    Index
    get_dimension() const;

    ///
    /// \param N dimension of vector
    ///
    void
    set_dimension(const Index N);

    ///
    /// Copy assignment
    /// \param v the values of its componets are copied to this vector
    ///
    Vector<T> &
    operator=(Vector<T> const & v);

    ///
    /// Vector increment
    /// \param v added to currrent vector
    ///
    Vector<T> &
    operator+=(Vector<T> const & v);

    ///
    /// Vector decrement
    /// \param v substracted from current vector
    ///
    Vector<T> &
    operator-=(Vector<T> const & v);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    Vector(const Index N, T const & s);

    ///
    /// Vector components
    ///
    std::vector<T>
    e;

  };

  ///
  /// Second order tensor in R^N.
  ///
  template<typename T>
  class Tensor
  {
  public:

    ///
    /// Default constructor
    ///
    Tensor();

    ///
    /// Constructor that initializes to NaNs
    /// \param N dimension
    ///
    Tensor(const Index N);

    ///
    /// Create tensor from a scalar
    /// \param N dimension
    /// \param s all components are set equal to this value
    ///
    Tensor(const Index N, T const & s);

    ///
    /// Create tensor specifying components
    /// \param N dimension
    /// \param  s00, s01, ... components in the R^2 canonical basis
    ///
    Tensor(T const & s00, T const & s01, T const & s10, T const & s11);

    ///
    /// Create tensor specifying components
    /// \param N dimension
    /// \param  s00, s01, ... components in the R^3 canonical basis
    ///
    Tensor(
        T const & s00, T const & s01, T const & s02,
        T const & s10, T const & s11, T const & s12,
        T const & s20, T const & s21, T const & s22);

    ///
    /// Create tensor from array - const version
    /// \param data_ptr pointer into the array
    ///
    Tensor(const Index N, T const * data_ptr);

    ///
    /// Create tensor from array
    /// \param data_ptr pointer into the array
    ///
    Tensor(const Index N, T * data_ptr);

    ///
    /// Copy constructor
    /// \param A the values of its componets are copied to the new tensor
    ///
    Tensor(Tensor<T> const & A);

    ///
    /// Simple destructor
    ///
    ~Tensor();

    ///
    /// Indexing for constant tensor
    /// \param i index
    /// \param j index
    ///
    const T &
    operator()(const Index i, const Index j) const;

    ///
    /// Tensor indexing
    /// \param i index
    /// \param j index
    ///
    T &
    operator()(const Index i, const Index j);

    ///
    /// \return dimension
    ///
    Index
    get_dimension() const;

    ///
    /// \param N dimension of 2nd-order tensor
    ///
    void
    set_dimension(const Index N);

    ///
    /// Copy assignment
    /// \param A the values of its componets are copied to this tensor
    ///
    Tensor<T> &
    operator=(Tensor<T> const & A);

    ///
    /// Tensor increment
    /// \param A added to current tensor
    ///
    Tensor<T> &
    operator+=(Tensor<T> const & A);

    ///
    /// Tensor decrement
    /// \param A substracted from current tensor
    ///
    Tensor<T> &
    operator-=(Tensor<T> const & A);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    std::vector<std::vector<T> >
    e;

  };

  ///
  /// Third order tensor in R^N.
  ///
  template<typename T>
  class Tensor3
  {
  public:

    ///
    /// Default constructor
    ///
    Tensor3();

    ///
    /// 3rd-order tensor constructor with NaNs
    ///
    Tensor3(const Index N);

    ///
    /// 3rd-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor3(const Index N, T const & s);

    ///
    /// Copy constructor
    /// 3rd-order tensor constructor from 3rd-order tensor
    /// \param A from which components are copied
    ///
    Tensor3(Tensor3<T> const & A);

    ///
    /// 3rd-order tensor simple destructor
    ///
    ~Tensor3();

    ///
    /// Indexing for constant 3rd-order tensor
    /// \param i index
    /// \param j index
    /// \param k index
    ///
    const T &
    operator()(const Index i, const Index j, const Index k) const;

    ///
    /// 3rd-order tensor indexing
    /// \param i index
    /// \param j index
    /// \param k index
    ///
    T &
    operator()(const Index i, const Index j, const Index k);

    ///
    /// \return dimension
    ///
    Index
    get_dimension() const;

    ///
    /// \param N dimension of 3rd-order tensor
    ///
    void
    set_dimension(const Index N);

    ///
    /// 3rd-order tensor copy assignment
    ///
    Tensor3<T> &
    operator=(Tensor3<T> const & A);

    ///
    /// 3rd-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor3<T> &
    operator+=(Tensor3<T> const & A);

    ///
    /// 3rd-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor3<T> &
    operator-=(Tensor3<T> const & A);

    ///
    /// Fill 3rd-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    std::vector<std::vector<std::vector<T> > >
    e;

  };

  ///
  /// Fourth order tensor in R^N.
  ///
  template<typename T>
  class Tensor4
  {
  public:

    ///
    /// Default constructor
    ///
    Tensor4();

    ///
    /// 4th-order tensor constructor with NaNs
    ///
    Tensor4(const Index N);

    ///
    /// 4th-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor4(const Index N, T const & s);

    ///
    /// Copy constructor
    /// 4th-order tensor constructor with 4th-order tensor
    /// \param A from which components are copied
    ///
    Tensor4(Tensor4<T> const & A);

    ///
    /// 4th-order tensor simple destructor
    ///
    ~Tensor4();

    ///
    /// Indexing for constant 4th-order tensor
    /// \param i index
    /// \param j index
    /// \param k index
    /// \param l index
    ///
    const T &
    operator()(
        const Index i,
        const Index j,
        const Index k,
        const Index l) const;

    ///
    /// 4th-order tensor indexing
    /// \param i index
    /// \param j index
    /// \param k index
    /// \param l index
    ///
    T &
    operator()(
        const Index i,
        const Index j,
        const Index k,
        const Index l);

    ///
    /// \return dimension
    ///
    Index
    get_dimension() const;

    ///
    /// \param N dimension of 4th-order tensor
    ///
    void
    set_dimension(const Index N);

    ///
    /// 4th-order tensor copy assignment
    ///
    Tensor4<T> &
    operator=(Tensor4<T> const & A);

    ///
    /// 4th-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor4<T> &
    operator+=(Tensor4<T> const & A);

    ///
    /// 4th-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor4<T> &
    operator-=(Tensor4<T> const & A);

    ///
    /// Fill 4th-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    std::vector<std::vector<std::vector<std::vector<T> > > >
    e;

  };

  //
  // Prototypes for utilities
  //

  ///
  /// Vector addition
  /// \param u
  /// \param v the operands
  /// \return \f$ u + v \f$
  ///
  template<typename T>
  Vector<T>
  operator+(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Vector substraction
  /// \param u
  /// \param v the operands
  /// \return \f$ u - v \f$
  ///
  template<typename T>
  Vector<T>
  operator-(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Vector minus
  /// \param u
  /// \return \f$ -u \f$
  ///
  template<typename T>
  Vector<T>
  operator-(Vector<T> const & u);

  ///
  /// Vector dot product
  /// \param u
  /// \param v the operands
  /// \return \f$ u \cdot v \f$
  ///
  template<typename T>
  T
  operator*(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Vector equality tested by components
  /// \param u
  /// \param v the operands
  /// \return \f$ u \equiv v \f$
  ///
  template<typename T>
  bool
  operator==(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Vector inequality tested by components
  /// \param u
  /// \param v the operands
  /// \return \f$ u \neq v \f$
  ///
  template<typename T>
  bool
  operator!=(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Scalar vector product
  /// \param s scalar factor
  /// \param u vector factor
  /// \return \f$ s u \f$
  ///
  template<typename T, typename S>
  Vector<T>
  operator*(S const & s, Vector<T> const & u);

  ///
  /// Vector scalar product
  /// \param u vector factor
  /// \param s scalar factor
  /// \return \f$ s u \f$
  ///
  template<typename T, typename S>
  Vector<T>
  operator*(Vector<T> const & u, S const & s);

  ///
  /// Vector scalar division
  /// \param u vector
  /// \param s scalar that divides each component of vector
  /// \return \f$ u / s \f$
  ///
  template<typename T, typename S>
  Vector<T>
  operator/(Vector<T> const & u, S const & s);

  ///
  /// Vector dot product
  /// \param u
  /// \param v operands
  /// \return \f$ u \cdot v \f$
  ///
  template<typename T>
  T
  dot(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Cross product only valid for R^3.
  /// R^N with N != 3 will produce an error.
  /// \param u
  /// \param v operands
  /// \return \f$ u \times v \f$
  ///
  template<typename T>
  Vector<T>
  cross(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Vector 2-norm
  /// \return \f$ \sqrt{u \cdot u} \f$
  ///
  template<typename T>
  T
  norm(Vector<T> const & u);

  ///
  /// Vector 1-norm
  /// \return \f$ |u_0|+|u_1|+|u_2| \f$
  ///
  template<typename T>
  T
  norm_1(Vector<T> const & u);

  ///
  /// Vector infinity-norm
  /// \return \f$ \max(|u_0|,|u_1|,|u_2|) \f$
  ///
  template<typename T>
  T
  norm_infinity(Vector<T> const & u);

  ///
  /// Tensor addition
  /// \return \f$ A + B \f$
  ///
  template<typename T>
  Tensor<T>
  operator+(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Tensor substraction
  /// \return \f$ A - B \f$
  ///
  template<typename T>
  Tensor<T>
  operator-(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T>
  Tensor<T>
  operator-(Tensor<T> const & A);

  ///
  /// Tensor equality
  /// Tested by components
  /// \return \f$ A \equiv B \f$
  ///
  template<typename T>
  bool
  operator==(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Tensor inequality
  /// Tested by components
  /// \return \f$ A \neq B \f$
  ///
  template<typename T>
  bool
  operator!=(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Scalar tensor product
  /// \param s scalar
  /// \param A tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor<T>
  operator*(S const & s, Tensor<T> const & A);

  ///
  /// Tensor scalar product
  /// \param A tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor<T>
  operator*(Tensor<T> const & A, S const & s);

  ///
  /// Tensor scalar division
  /// \param A tensor
  /// \param s scalar
  /// \return \f$ A / s \f$
  ///
  template<typename T, typename S>
  Tensor<T>
  operator/(Tensor<T> const & A, S const & s);

  ///
  /// Scalar 3rd-order tensor product
  /// \param s scalar
  /// \param A 3rd-order tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor3<T>
  operator*(S const & s, Tensor3<T> const & A);

  ///
  /// 3rd-order tensor scalar product
  /// \param A 3rd-order tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor3<T>
  operator*(Tensor3<T> const & A, S const & s);

  ///
  /// 3rd-order tensor vector product
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T>
  Tensor<T>
  dot(Tensor3<T> const & A, Vector<T> const & u);

  ///
  /// vector 3rd-order tensor product
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ u A \f$
  ///
  template<typename T>
  Tensor<T>
  dot(Vector<T> const & u, Tensor3<T> const & A);

  ///
  /// 3rd-order tensor vector product2 (contract 2nd index)
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T>
  Tensor<T>
  dot2(Tensor3<T> const & A, Vector<T> const & u);

  ///
  /// vector 3rd-order tensor product2 (contract 2nd index)
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ u A \f$
  ///
  template<typename T>
  Tensor<T>
  dot2(Vector<T> const & u, Tensor3<T> const & A);

  ///
  /// Scalar 4th-order tensor product
  /// \param s scalar
  /// \param A 4th-order tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor4<T>
  operator*(S const & s, Tensor4<T> const & A);

  ///
  /// 4th-order tensor scalar product
  /// \param A 4th-order tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor4<T>
  operator*(Tensor4<T> const & A, S const & s);

  ///
  /// Tensor vector product v = A u
  /// \param A tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T>
  Vector<T>
  dot(Tensor<T> const & A, Vector<T> const & u);

  ///
  /// Vector tensor product v = u A
  /// \param A tensor
  /// \param u vector
  /// \return \f$ u A = A^T u \f$
  ///
  template<typename T>
  Vector<T>
  dot(Vector<T> const & u, Tensor<T> const & A);

  ///
  /// Tensor vector product v = A u
  /// \param A tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T>
  Vector<T>
  operator*(Tensor<T> const & A, Vector<T> const & u);

  ///
  /// Vector tensor product v = u A
  /// \param A tensor
  /// \param u vector
  /// \return \f$ u A = A^T u \f$
  ///
  template<typename T>
  Vector<T>
  operator*(Vector<T> const & u, Tensor<T> const & A);

  ///
  /// Tensor dot product C = A B
  /// \return \f$ A \cdot B \f$
  ///
  template<typename T>
  Tensor<T>
  operator*(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Tensor tensor product C = A B
  /// \param A tensor
  /// \param B tensor
  /// \return a tensor \f$ A \cdot B \f$
  ///
  template<typename T>
  Tensor<T>
  dot(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Tensor tensor double dot product (contraction)
  /// \param A tensor
  /// \param B tensor
  /// \return a scalar \f$ A : B \f$
  ///
  template<typename T>
  T
  dotdot(Tensor<T> const & A, Tensor<T> const & B);

  /// Tensor4 Tensor4 double dot product
  /// \param A Tensor4
  /// \param B Tensor4
  /// \return a Tensor4 \f$ C_{ijkl} = A_{ijmn} : B){mnkl} \f$
  template<typename T>
  Tensor4<T>
  dotdot(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// Tensor Frobenius norm
  /// \return \f$ \sqrt{A:A} \f$
  ///
  template<typename T>
  T
  norm(Tensor<T> const & A);

  ///
  /// Tensor 1-norm
  /// \return \f$ \max_{j \in {0,1,2}}\Sigma_{i=0}^2 |A_{ij}| \f$
  ///
  template<typename T>
  T
  norm_1(Tensor<T> const & A);

  ///
  /// Tensor infinity-norm
  /// \return \f$ \max_{i \in {0,1,2}}\Sigma_{j=0}^2 |A_{ij}| \f$
  ///
  template<typename T>
  T
  norm_infinity(Tensor<T> const & A);

  ///
  /// Dyad
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename T>
  Tensor<T>
  dyad(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Bun operator, just for Jay
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename T>
  Tensor<T>
  bun(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Tensor product
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename T>
  Tensor<T>
  tensor(Vector<T> const & u, Vector<T> const & v);

  ///
  /// Diagonal tensor from vector
  /// \param v vector
  /// \return A = diag(v)
  ///
  template<typename T>
  Tensor<T>
  diag(Vector<T> const & v);

  ///
  /// Diagonal of tensor in a vector
  /// \param A tensor
  /// \return v = diag(A)
  ///
  template<typename T>
  Vector<T>
  diag(Tensor<T> const & A);

  ///
  /// Zero 2nd-order tensor
  /// All components are zero
  ///
  template<typename T>
  const Tensor<T>
  zero(const Index N);

  ///
  /// 2nd-order identity tensor
  ///
  template<typename T>
  const Tensor<T>
  identity(const Index N);

  ///
  /// 2nd-order identity tensor, à la Matlab
  ///
  template<typename T>
  const Tensor<T>
  eye(const Index N);

  ///
  /// 2nd-order tensor transpose
  ///
  template<typename T>
  Tensor<T>
  transpose(Tensor<T> const & A);

  ///
  /// 4th-order tensor transpose
  ///
  template<typename T>
  Tensor4<T>
  transpose(Tensor4<T> const & A);

  ///
  /// Symmetric part of 2nd-order tensor
  /// \return \f$ \frac{1}{2}(A + A^T) \f$
  ///
  template<typename T>
  Tensor<T>
  symm(Tensor<T> const & A);

  ///
  /// Skew symmetric part of 2nd-order tensor
  /// \return \f$ \frac{1}{2}(A - A^T) \f$
  ///
  template<typename T>
  Tensor<T>
  skew(Tensor<T> const & A);

  ///
  /// Skew symmetric 2nd-order tensor from vector valid for R^3 only.
  /// R^N with N != 3 will produce an error
  /// \param u vector
  /// \return \f$ {{0, -u_2, u_1}, {u_2, 0, -u_0}, {-u_1, u+0, 0}} \f$
  ///
  template<typename T>
  Tensor<T>
  skew(Vector<T> const & u);

  ///
  /// Volumetric part of 2nd-order tensor  
  /// \param A tensor
  /// \return \f$ \frac{1}{3} \mathrm{tr}\:A I \f$
  ///
  template<typename T>
  Tensor<T>
  vol(Tensor<T> const & A);

  ///
  /// Deviatoric part of 2nd-order tensor
  /// \param A tensor
  /// \return \f$ A - vol(A) \f$
  ///
  template<typename T>
  Tensor<T>
  dev(Tensor<T> const & A);

  ///
  /// 2nd-order tensor inverse
  /// \param A nonsingular tensor
  /// \return \f$ A^{-1} \f$
  ///
  template<typename T>
  Tensor<T>
  inverse(Tensor<T> const & A);

  ///
  /// Subtensor
  /// \param A tensor
  /// \param i index
  /// \param j index
  /// \return Subtensor with i-row and j-col deleted.
  ///
  template<typename T>
  Tensor<T>
  subtensor(Tensor<T> const & A, Index i, Index j);

  ///
  /// Swap row. Echange rows i and j in place
  /// \param A tensor
  /// \param i index
  /// \param j index
  ///
  template<typename T>
  void
  swap_row(Tensor<T> & A, Index i, Index j);

  ///
  /// Swap column. Echange columns i and j in place
  /// \param A tensor
  /// \param i index
  /// \param j index
  ///
  template<typename T>
  void
  swap_col(Tensor<T> & A, Index i, Index j);

  ///
  /// Determinant
  /// \param A tensor
  /// \return \f$ \det A \f$
  ///
  template<typename T>
  T
  det(Tensor<T> const & A);

  ///
  /// Trace
  /// \param A tensor
  /// \return \f$ A:I \f$
  ///
  template<typename T>
  T
  trace(Tensor<T> const & A);

  ///
  /// First invariant, trace
  /// \param A tensor
  /// \return \f$ I_A = A:I \f$
  ///
  template<typename T>
  T
  I1(Tensor<T> const & A);

  ///
  /// Second invariant
  /// \param A tensor
  /// \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
  ///
  template<typename T>
  T
  I2(Tensor<T> const & A);

  ///
  /// Third invariant
  /// \param A tensor
  /// \return \f$ III_A = \det A \f$
  ///
  template<typename T>
  T
  I3(Tensor<T> const & A);

  ///
  /// Exponential map by Taylor series, radius of convergence is infinity
  /// \param A tensor
  /// \return \f$ \exp A \f$
  ///
  template<typename T>
  Tensor<T>
  exp(Tensor<T> const & A);

  ///
  /// Logarithmic map by Taylor series, converges for \f$ |A-I| < 1 \f$
  /// \param A tensor
  /// \return \f$ \log A \f$
  ///
  template<typename T>
  Tensor<T>
  log(Tensor<T> const & A);

  ///
  /// Logarithmic map of a rotation
  /// \param R with \f$ R \in SO(3) \f$
  /// \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  ///
  template<typename T>
  Tensor<T>
  log_rotation(Tensor<T> const & R);

  ///
  /// Logarithmic map of a 180 degree rotation
  /// \param R with \f$ R \in SO(3) \f$
  /// \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  ///
  template<typename T>
  Tensor<T>
  log_rotation_pi(Tensor<T> const & R);

  /// Gaussian Elimination with partial pivot
  /// \param A
  /// \return \f$ xvec \f$
  ///
  template<typename T>
  Tensor<T>
  gaussian_elimination(Tensor<T> const & A);

  /// Apply Givens-Jacobi rotation on the left in place.
  /// \param c and s for a rotation G in form [c, s; -s, c]
  /// \param A
  ///
  template<typename T>
  void
  givens_left(T const & c, T const & s, Index i, Index k, Tensor<T> & A);

  /// Apply Givens-Jacobi rotation on the right in place.
  /// \param A
  /// \param c and s for a rotation G in form [c, s; -s, c]
  ///
  template<typename T>
  void
  givens_right(T const & c, T const & s, Index i, Index k, Tensor<T> & A);

  ///
  /// Exponential map of a skew-symmetric tensor
  /// \param r \f$ r \in so(3) \f$
  /// \return \f$ R = \exp R \f$ with \f$ R \in SO(3) \f$
  ///
  template<typename T>
  Tensor<T>
  exp_skew_symmetric(Tensor<T> const & r);

  ///
  /// Off-diagonal norm. Useful for SVD and other algorithms
  /// that rely on Jacobi-type procedures.
  /// \param A
  /// \return \f$ \sqrt(\sum_i \sum_{j, j\neq i} a_{ij}^2) \f$
  ///
  template<typename T>
  T
  norm_off_diagonal(Tensor<T> const & A);

  ///
  /// Arg max abs. Useful for inverse and other algorithms
  /// that rely on Jacobi-type procedures.
  /// \param A
  /// \return \f$ (p,q) = arg max_{i,j} |a_{ij}| \f$
  ///
  template<typename T>
  std::pair<Index, Index>
  arg_max_abs(Tensor<T> const & A);

  ///
  /// Arg max off-diagonal. Useful for SVD and other algorithms
  /// that rely on Jacobi-type procedures.
  /// \param A
  /// \return \f$ (p,q) = arg max_{i \neq j} |a_{ij}| \f$
  ///
  template<typename T>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T> const & A);

  ///
  /// Sort and index. Useful for ordering singular values
  /// and eigenvalues and corresponding vectors in the
  /// respective decompositions.
  /// \param u vector to sort
  /// \return pair<v, P>
  /// \return v sorted vector
  /// \return P permutation matrix such that v = P^T u
  ///
  template<typename T>
  std::pair<Vector<T>, Tensor<T> >
  sort_permutation(Vector<T> const & u);

  ///
  /// Singular value decomposition (SVD)
  /// \param A tensor
  /// \return \f$ A = USV^T\f$
  ///
  template<typename T>
  boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
  svd(Tensor<T> const & A);

  ///
  /// Project to O(N) (Orthogonal Group) using a Newton-type algorithm.
  /// See Higham's Functions of Matrices p210 [2008]
  /// \param A tensor (often a deformation-gradient-like tensor)
  /// \return \f$ R = \argmin_Q \|A - Q\|\f$
  /// This algorithm projects a given tensor in GL(N) to O(N).
  /// The rotation/reflection obtained through this projection is
  /// the orthogonal component of the real polar decomposition
  ///
  template<typename T>
  Tensor<T>
  polar_rotation(Tensor<T> const & A);

  ///
  /// Left polar decomposition
  /// \param A tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = A \f$ with \f$ R \in SO(N) \f$ and \f$ V \in SPD(N) \f$
  ///
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_left(Tensor<T> const & A);

  ///
  /// Right polar decomposition
  /// \param A tensor (often a deformation-gradient-like tensor)
  /// \return \f$ RU = A \f$ with \f$ R \in SO(N) \f$ and \f$ U \in SPD(N) \f$
  ///
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_right(Tensor<T> const & A);

  ///
  /// Left polar decomposition computed with eigenvalue decomposition
  /// \param A tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = A \f$ with \f$ R \in SO(N) \f$ and \f$ V \in SPD(N) \f$
  ///
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_left_eig(Tensor<T> const & A);

  ///
  /// R^3 right polar decomposition
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ RU = F \f$ with \f$ R \in SO(N) \f$ and \f$ U \in SPD(N) \f$
  ///
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_right_eig(Tensor<T> const & A);

  ///
  /// Left polar decomposition with matrix logarithm for V
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = F \f$ with \f$ R \in SO(N) \f$ and V SPD, and log V
  ///
  template<typename T>
  boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
  polar_left_logV(Tensor<T> const & F);

  ///
  /// Left polar decomposition with matrix logarithm for V using eig_spd_cos
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = F \f$ with \f$ R \in SO(N) \f$ and V SPD, and log V
  ///
  template<typename T>
  boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
  polar_left_logV_lame(Tensor<T> const & F);

  ///
  /// Logarithmic map using BCH expansion (3 terms)
  /// \param v tensor
  /// \param r tensor
  /// \return Baker-Campbell-Hausdorff series up to 3 terms
  ///
  template<typename T>
  Tensor<T>
  bch(Tensor<T> const & v, Tensor<T> const & r);

  ///
  /// Symmetric Schur algorithm for R^2.
  /// \param \f$ A = [f, g; g, h] \in S(2) \f$
  /// \return \f$ c, s \rightarrow [c, -s; s, c]\f diagonalizes A$
  ///
  template<typename T>
  std::pair<T, T>
  schur_sym(const T f, const T g, const T h);

  ///
  /// Givens rotation. [c, -s; s, c] [a; b] = [r; 0]
  /// \param a, b
  /// \return c, s
  ///
  template<typename T>
  std::pair<T, T>
  givens(T const & a, T const & b);

  ///
  /// Eigenvalue decomposition for symmetric 2nd-order tensor
  /// \param A tensor
  /// \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  ///
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  eig_sym(Tensor<T> const & A);

  ///
  /// Eigenvalue decomposition for SPD 2nd-order tensor
  /// \param A tensor
  /// \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  ///
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  eig_spd(Tensor<T> const & A);

  ///
  /// Eigenvalue decomposition for SPD 2nd-order tensor
  /// \param A tensor
  /// \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  /// This algorithm comes from the journal article
  /// Scherzinger and Dohrmann, CMAME 197 (2008) 4007-4015
  ///
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  eig_spd_cos(Tensor<T> const & A);

  ///
  /// Cholesky decomposition, rank-1 update algorithm
  /// (Matrix Computations 3rd ed., Golub & Van Loan, p145)
  /// \param A assumed symetric tensor
  /// \return G Cholesky factor A = GG^T
  /// \return completed (bool) algorithm ran to completion
  ///
  template<typename T>
  std::pair<Tensor<T>, bool >
  cholesky(Tensor<T> const & A);

  ///
  /// 4th-order identity I1
  /// \return \f$ \delta_{ik} \delta_{jl} \f$ such that \f$ A = I_1 A \f$
  ///
  template<typename T>
  const Tensor4<T>
  identity_1(const Index N);

  ///
  /// 4th-order identity I2
  /// \return \f$ \delta_{il} \delta_{jk} \f$ such that \f$ A^T = I_2 A \f$
  ///
  template<typename T>
  const Tensor4<T>
  identity_2(const Index N);

  ///
  /// 4th-order identity I3
  /// \return \f$ \delta_{ij} \delta_{kl} \f$ such that \f$ I_A I = I_3 A \f$
  ///
  template<typename T>
  const Tensor4<T>
  identity_3(const Index N);

  ///
  /// 4th-order tensor vector dot product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ A dot u \f$ as \f$ B_{ijk}=A_{ijkl}u_{l} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot(Tensor4<T> const & A, Vector<T> const & u);

  ///
  /// vector 4th-order tensor dot product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ u dot A \f$ as \f$ B_{jkl}=u_{i} A_{ijkl} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot(Vector<T> const & u, Tensor4<T> const & A);

  ///
  /// 4th-order tensor vector dot2 product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ A dot2 u \f$ as \f$ B_{ijl}=A_{ijkl}u_{k} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot2(Tensor4<T> const & A, Vector<T> const & u);

  ///
  /// vector 4th-order tensor dot2 product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ u dot2 A \f$ as \f$ B_{ikl}=u_{j}A_{ijkl} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot2(Vector<T> const & u, Tensor4<T> const & A);

  ///
  /// 4th-order tensor 2nd-order tensor double dot product
  /// \param A 4th-order tensor
  /// \param B 2nd-order tensor
  /// \return 2nd-order tensor \f$ A:B \f$ as \f$ C_{ij}=A_{ijkl}B_{kl} \f$
  ///
  template<typename T>
  Tensor<T>
  dotdot(Tensor4<T> const & A, Tensor<T> const & B);

  ///
  /// 2nd-order tensor 4th-order tensor double dot product
  /// \param B 2nd-order tensor
  /// \param A 4th-order tensor
  /// \return 2nd-order tensor \f$ B:A \f$ as \f$ C_{kl}=A_{ijkl}B_{ij} \f$
  ///
  template<typename T>
  Tensor<T>
  dotdot(Tensor<T> const & B, Tensor4<T> const & A);

  ///
  /// 2nd-order tensor 2nd-order tensor tensor product
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \otimes B \f$
  ///
  template<typename T>
  Tensor4<T>
  tensor(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// odot operator useful for \f$ \frac{\partial A^{-1}}{\partial A} \f$
  /// see Holzapfel eqn 6.165
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \odot B \f$ which is
  /// \f$ C_{ijkl} = \frac{1}{2}(A_{ik} B_{jl} + A_{il} B_{jk}) \f$
  ///
  template<typename T>
  Tensor4<T>
  odot(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// 3rd-order tensor addition
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename T>
  Tensor3<T>
  operator+(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// 3rd-order tensor substraction
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename T>
  Tensor3<T>
  operator-(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// 3rd-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T>
  Tensor3<T>
  operator-(Tensor3<T> const & A);

  ///
  /// 3rd-order tensor equality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator==(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// 3rd-order tensor inequality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator!=(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// 4th-order tensor addition
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename T>
  Tensor4<T>
  operator+(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// 4th-order tensor substraction
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename T>
  Tensor4<T>
  operator-(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// 4th-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T>
  Tensor4<T>
  operator-(Tensor4<T> const & A);

  ///
  /// 4th-order equality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator==(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// 4th-order inequality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator!=(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// Vector input
  /// \param u vector
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Vector<T> & u);

  ///
  /// Vector output
  /// \param u vector
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Vector<T> const & u);

  ///
  /// Tensor input
  /// \param A tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor<T> & A);

  ///
  /// Tensor output
  /// \param A tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T> const & A);

  ///
  /// 3rd-order tensor input
  /// \param A 3rd-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor3<T> & A);

  ///
  /// 3rd-order tensor output
  /// \param A 3rd-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<T> const & A);

  ///
  /// 4th-order input
  /// \param A 4th-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor4<T> & A);

  ///
  /// 4th-order output
  /// \param A 4th-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<T> const & A);

} // namespace LCM

#include "Tensor.i.cc"
#include "Tensor.t.cc"

#endif //LCM_Tensor_h
