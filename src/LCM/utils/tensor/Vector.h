//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Vector_h)
#define tensor_Vector_h

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "Definitions.h"
#include "Utilities.h"

namespace LCM {

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
    Vector(Index const N);

    // ///
    // /// Create vector from a scalar
    // /// \param N dimension
    // /// \param s all components are set equal to this value
    // ///
    // // Vector(Index const N, T const & s);

    ///
    /// Create vector specifying components
    /// \param s0, s1 are the vector components in the R^2 canonical basis
    ///
    Vector(T const & s0, T const & s1);

    ///
    /// Create vector specifying components
    /// the vector components in the R^3 canonical basis
    /// \param s0 
    /// \param s1 
    /// \param s2 
    ///
    Vector(T const & s0, T const & s1, T const & s2);

    ///
    /// Create vector from array - const version
    /// \param N dimension
    /// \param data_ptr pointer into the array
    ///
    Vector(Index const N, T const * data_ptr);

    ///
    /// Copy constructor
    /// \param v the values of its components are copied to the new vector
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
    operator()(Index const i) const;

    ///
    /// Vector indexing
    /// \param i the index
    ///
    T &
    operator()(Index const i);

    ///
    /// \return dimension
    ///
    Index
    get_dimension() const;

    ///
    /// \param N dimension of vector
    ///
    void
    set_dimension(Index const N);

    ///
    /// Fill components from array defined by pointer.
    /// \param data_ptr pointer into array for filling components
    ///
    void
    fill(T const * data_ptr);

    ///
    /// Copy assignment
    /// \param v the values of its components are copied to this vector
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

    Vector(Index const N, T const & s);

    ///
    /// Vector dimension
    ///
    Index
    dimension;

    ///
    /// Vector components
    ///
    T *
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
  /// Vector 2-norm square. Used for fast distance calculation.
  /// \return \f$ u \cdot u \f$
  ///
  template<typename T>
  T
  norm_square(Vector<T> const & u);

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

} // namespace LCM

#include "Vector.i.cc"
#include "Vector.t.cc"

#endif //tensor_Vector_h
