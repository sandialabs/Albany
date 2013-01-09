//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor_h)
#define tensor_Tensor_h

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/type_traits/is_complex.hpp>

#include "Vector.h"

namespace LCM {

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
    /// \param A the values of its components are copied to the new tensor
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
    /// \param A the values of its components are copied to this tensor
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
  /// Tensor tensor product C = A^T B
  /// \param A tensor
  /// \param B tensor
  /// \return a tensor \f$ A^T \cdot B \f$
  ///
  template<typename T>
  Tensor<T>
  t_dot(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Tensor tensor product C = A B^T
  /// \param A tensor
  /// \param B tensor
  /// \return a tensor \f$ A \cdot B^T \f$
  ///
  template<typename T>
  Tensor<T>
  dot_t(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// Tensor tensor double dot product (contraction)
  /// \param A tensor
  /// \param B tensor
  /// \return a scalar \f$ A : B \f$
  ///
  template<typename T>
  T
  dotdot(Tensor<T> const & A, Tensor<T> const & B);

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
  /// R^N 2nd-order tensor transpose
  ///
  template<typename T>
  Tensor<T>
  transpose(Tensor<T> const & A);

  ///
  /// R^N 2nd-order tensor transpose
  ///
  template<typename T>
  Tensor<T>
  transpose_1(Tensor<T> const & A);

  ///
  /// C^N 2nd-order tensor adjoint
  ///
  template<typename T>
  Tensor<T>
  adjoint(Tensor<T> const & A);

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

} // namespace LCM

#include "Tensor.i.cc"
#include "Tensor.t.cc"

#endif //tensor_Tensor_h
