///
/// \file Tensor.h
/// First cut of LCM small tensor utilities. Declarations.
/// \author Alejandro Mota
/// \author Jake Ostien
///
#if !defined(LCM_Tensor_h)
#define LCM_Tensor_h

#include <cmath>
#include <cstdarg>
#include <iostream>
#include <vector>

#include <boost/tuple/tuple.hpp>

namespace LCM {

  ///
  /// Indexing type
  ///
  typedef unsigned int Index;

  ///
  /// Sign function
  ///
  template <typename T>
  int
  sgn(T const & s);

  ///
  /// Half angle cosine and sine. Useful for SVD
  /// in that it does not use any trigonometric
  /// functions, just square roots.
  /// \param catheti x, y
  /// \return cosine and sine of 0.5 * atan2(y, x)
  ///
  template <typename T>
  std::pair<T, T>
  half_angle(T const & x, T const & y);

  ///
  /// Vector in R^N provided just as a framework to
  /// specialize the R^2 and R^3 versions.
  ///
  template<typename T, Index N>
  class Vector
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Vector();

    ///
    /// Create vector from a scalar
    /// \param s all components are set equal to this value
    ///
    Vector(T const & s);

    ///
    /// Create vector specifying components
    /// \param s0,... are the vector components in the canonical basis
    ///
    Vector(T const & s0, ...);

    ///
    /// Create vector from array - const version
    /// \param data_ptr pointer into the array
    ///
    Vector(T const * data_ptr);

    ///
    /// Create vector from array
    /// \param data_ptr pointer into the array
    ///
    Vector(T* data_ptr);

    ///
    /// Copy constructor
    /// \param v the values of its componets are copied to the new vector
    ///
    Vector(Vector<T, N> const & v);

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
    /// Copy assignment
    /// \param v the values of its componets are copied to this vector
    ///
    Vector<T, N> &
    operator=(Vector<T, N> const & v);

    ///
    /// Vector increment
    /// \param v added to currrent vector
    ///
    Vector<T, N> &
    operator+=(Vector<T, N> const & v);

    ///
    /// Vector decrement
    /// \param v substracted from current vector
    ///
    Vector<T, N> &
    operator-=(Vector<T, N> const & v);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Vector components
    ///
    std::vector<T>
    e;

  };


  ///
  /// Vector in R^3
  ///
  template<typename T>
  class Vector<T, 3>
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Vector();

    ///
    /// Create vector from a scalar
    /// \param s all components are set equal to this value
    ///
    Vector(T const & s);

    ///
    /// Create vector specifying components
    /// \param s0,... are the vector components in the canonical basis
    ///
    Vector(T const & s0, T const & s1, T const & s2);

    ///
    /// Create vector from array - const version
    /// \param data_ptr pointer into the array
    ///
    Vector(const T * data_ptr);

    ///
    /// Create vector from array
    /// \param data_ptr pointer into the array
    ///
    Vector(T * data_ptr);

    ///
    /// Copy constructor
    /// \param v the values of its componets are copied to the new vector
    ///
    Vector(Vector<T, 3> const & v);

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
    /// Copy assignment
    /// \param v the values of its componets are copied to this vector
    ///
    Vector<T, 3> &
    operator=(Vector<T, 3> const & v);

    ///
    /// Vector increment
    /// \param v added to currrent vector
    ///
    Vector<T, 3> &
    operator+=(Vector<T, 3> const & v);

    ///
    /// Vector decrement
    /// \param v substracted from current vector
    ///
    Vector<T, 3> &
    operator-=(Vector<T, 3> const & v);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Vector components
    ///
    T
    e[3];

  };

  ///
  /// Vector in R^2
  ///
  template<typename T>
  class Vector<T, 2>
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Vector();

    ///
    /// Create vector from a scalar
    /// \param s all components are set equal to this value
    ///
    Vector(T const & s);

    ///
    /// Create vector specifying components
    /// \param s0,... are the vector components in the canonical basis
    ///
    Vector(T const & s0, T const & s1);

    ///
    /// Create vector from array - const version
    /// \param data_ptr pointer into the array
    ///
    Vector(T const * data_ptr);

    ///
    /// Create vector from array
    /// \param data_ptr pointer into the array
    ///
    Vector(T* data_ptr);

    ///
    /// Copy constructor
    /// \param v the values of its componets are copied to the new vector
    ///
    Vector(Vector<T, 2> const & v);

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
    /// Copy assignment
    /// \param v the values of its componets are copied to this vector
    ///
    Vector<T, 2> &
    operator=(Vector<T, 2> const & v);

    ///
    /// Vector increment
    /// \param v added to currrent vector
    ///
    Vector<T, 2> &
    operator+=(Vector<T, 2> const & v);

    ///
    /// Vector decrement
    /// \param v substracted from current vector
    ///
    Vector<T, 2> &
    operator-=(Vector<T, 2> const & v);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Vector components
    ///
    T
    e[2];

  };

  ///
  /// Second order tensor in R^N provided just as a framework to
  /// specialize the R^2 and R^3 versions.
  ///
  template<typename T, Index N>
  class Tensor
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Tensor();

    ///
    /// Create tensor from a scalar
    /// \param s all components are set equal to this value
    ///
    Tensor(T const & s);

    ///
    /// Create tensor specifying components
    /// The parameters are the components in the canonical basis
    /// \param s00 ...
    ///
    Tensor(T const & s00, ...);

    ///
    /// Create tensor from array - const version
    /// \param data_ptr pointer into the array
    ///
    Tensor(T const * data_ptr);

    ///
    /// Create tensor from array
    /// \param data_ptr pointer into the array
    ///
    Tensor(T* data_ptr);

    ///
    /// Copy constructor
    /// \param A the values of its componets are copied to the new tensor
    ///
    Tensor(Tensor<T, N> const & A);

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
    /// Copy assignment
    /// \param A the values of its componets are copied to this tensor
    ///
    Tensor<T, N> &
    operator=(Tensor<T, N> const & A);

    ///
    /// Tensor increment
    /// \param A added to current tensor
    ///
    Tensor<T, N> &
    operator+=(Tensor<T, N> const & A);

    ///
    /// Tensor decrement
    /// \param A substracted from current tensor
    ///
    Tensor<T, N> &
    operator-=(Tensor<T, N> const & A);

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
  /// Second order tensor in R^3
  ///
  template<typename T>
  class Tensor<T, 3>
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Tensor();

    ///
    /// Create tensor from a scalar
    /// \param s all components are set equal to this value
    ///
    Tensor(T const & s);

    ///
    /// Create tensor specifying components
    /// The parameters are the components in the canonical basis
    /// \param s00 ...
    ///
    Tensor(
        T const & s00, T const & s01, T const & s02,
        T const & s10, T const & s11, T const & s12,
        T const & s20, T const & s21, T const & s22);

    ///
    /// Create tensor from array - const version
    /// \param data_ptr pointer into the array
    ///
    Tensor(T const * data_ptr);

    ///
    /// Create tensor from array
    /// \param data_ptr pointer into the array
    ///
    Tensor(T* data_ptr);

    ///
    /// Copy constructor
    /// \param A the values of its componets are copied to the new tensor
    ///
    Tensor(Tensor<T, 3> const & A);

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
    /// Copy assignment
    /// \param A the values of its componets are copied to this tensor
    ///
    Tensor<T, 3> &
    operator=(Tensor<T, 3> const & A);

    ///
    /// Tensor increment
    /// \param A added to current tensor
    ///
    Tensor<T, 3> &
    operator+=(Tensor<T, 3> const & A);

    ///
    /// Tensor decrement
    /// \param A substracted from current tensor
    ///
    Tensor<T, 3> &
    operator-=(Tensor<T, 3> const & A);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    T
    e[3][3];

  };

  ///
  /// Second order tensor in R^2
  ///
  template<typename T>
  class Tensor<T, 2>
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Tensor();

    ///
    /// Create tensor from a scalar
    /// \param s all components are set equal to this value
    ///
    Tensor(T const & s);

    ///
    /// Create tensor specifying components
    /// The parameters are the components in the canonical basis
    /// \param s00 ...
    ///
    Tensor(
        T const & s00, T const & s01,
        T const & s10, T const & s11);

    ///
    /// Create tensor from array - const version
    /// \param data_ptr pointer into the array
    ///
    Tensor(T const * data_ptr);

    ///
    /// Create tensor from array
    /// \param data_ptr pointer into the array
    ///
    Tensor(T* data_ptr);

    ///
    /// Copy constructor
    /// \param A the values of its componets are copied to the new tensor
    ///
    Tensor(Tensor<T, 2> const & A);

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
    /// Copy assignment
    /// \param A the values of its componets are copied to this tensor
    ///
    Tensor<T, 2> &
    operator=(Tensor<T, 2> const & A);

    ///
    /// Tensor increment
    /// \param A added to current tensor
    ///
    Tensor<T, 2> &
    operator+=(Tensor<T, 2> const & A);

    ///
    /// Tensor decrement
    /// \param A substracted from current tensor
    ///
    Tensor<T, 2> &
    operator-=(Tensor<T, 2> const & A);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    T
    e[2][2];

  };

  ///
  /// Third order tensor in R^N provided just as a framework to
  /// specialize the R^2 and R^3 versions.
  ///
  template<typename T, Index N>
  class Tensor3
  {
  public:

    ///
    /// 3rd-order tensor constructor with NaNs
    ///
    Tensor3();

    ///
    /// 3rd-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor3(T const & s);

    ///
    /// Copy constructor
    /// 3rd-order tensor constructor from 3rd-order tensor
    /// \param A from which components are copied
    ///
    Tensor3(Tensor3<T, N> const & A);

    ///
    /// 3rd-order tensor simple destructor
    ///
    ~Tensor3();

    ///
    /// Indexing for constant 3rd order tensor
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
    /// 3rd-order tensor copy assignment
    ///
    Tensor3<T, N> &
    operator=(Tensor3<T, N> const & A);

    ///
    /// 3rd-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor3<T, N> &
    operator+=(Tensor3<T, N> const & A);

    ///
    /// 3rd-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor3<T, N> &
    operator-=(Tensor3<T, N> const & A);

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
  /// Third order tensor in R^3
  ///
  template<typename T>
  class Tensor3<T, 3>
  {
  public:

    ///
    /// 3rd-order tensor constructor with NaNs
    ///
    Tensor3();

    ///
    /// 3rd-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor3(T const & s);

    ///
    /// Copy constructor
    /// 3rd-order tensor constructor from 3rd-order tensor
    /// \param A from which components are copied
    ///
    Tensor3(Tensor3<T, 3> const & A);

    ///
    /// 3rd-order tensor simple destructor
    ///
    ~Tensor3();

    ///
    /// Indexing for constant 3rd order tensor
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
    /// 3rd-order tensor copy assignment
    ///
    Tensor3<T, 3> &
    operator=(Tensor3<T, 3> const & A);

    ///
    /// 3rd-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor3<T, 3> &
    operator+=(Tensor3<T, 3> const & A);

    ///
    /// 3rd-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor3<T, 3> &
    operator-=(Tensor3<T, 3> const & A);

    ///
    /// Fill 3rd-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    T
    e[3][3][3];

  };

  ///
  /// Third order tensor in R^2
  ///
  template<typename T>
  class Tensor3<T, 2>
  {
  public:

    ///
    /// 3rd-order tensor constructor with NaNs
    ///
    Tensor3();

    ///
    /// 3rd-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor3(T const & s);

    ///
    /// Copy constructor
    /// 3rd-order tensor constructor from 3rd-order tensor
    /// \param A from which components are copied
    ///
    Tensor3(Tensor3<T, 2> const & A);

    ///
    /// 3rd-order tensor simple destructor
    ///
    ~Tensor3();

    ///
    /// Indexing for constant 3rd order tensor
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
    /// 3rd-order tensor copy assignment
    ///
    Tensor3<T, 2> &
    operator=(Tensor3<T, 2> const & A);

    ///
    /// 3rd-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor3<T, 2> &
    operator+=(Tensor3<T, 2> const & A);

    ///
    /// 3rd-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor3<T, 2> &
    operator-=(Tensor3<T, 2> const & A);

    ///
    /// Fill 3rd-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    T
    e[2][2][2];

  };

  ///
  /// Fourth order tensor in R^N provided just as a framework to
  /// specialize the R^2 and R^3 versions.
  ///
  template<typename T, Index N>
  class Tensor4
  {
  public:

    ///
    /// 4th-order tensor constructor with NaNs
    ///
    Tensor4();

    ///
    /// 4th-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor4(T const & s);

    ///
    /// Copy constructor
    /// 4th-order tensor constructor with 4th-order tensor
    /// \param A from which components are copied
    ///
    Tensor4(Tensor4<T, N> const & A);

    ///
    /// 4th-order tensor simple destructor
    ///
    ~Tensor4();

    ///
    /// Indexing for constant 4th order tensor
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
    /// 4th-order tensor copy assignment
    ///
    Tensor4<T, N> &
    operator=(Tensor4<T, N> const & A);

    ///
    /// 4th-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor4<T, N> &
    operator+=(Tensor4<T, N> const & A);

    ///
    /// 4th-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor4<T, N> &
    operator-=(Tensor4<T, N> const & A);

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

  ///
  /// Fourth order tensor in R^3.
  ///
  template<typename T>
  class Tensor4<T, 3>
  {
  public:

    ///
    /// 4th-order tensor constructor with NaNs
    ///
    Tensor4();

    ///
    /// 4th-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor4(T const & s);

    ///
    /// Copy constructor
    /// 4th-order tensor constructor with 4th-order tensor
    /// \param A from which components are copied
    ///
    Tensor4(Tensor4<T, 3> const & A);

    ///
    /// 4th-order tensor simple destructor
    ///
    ~Tensor4();

    ///
    /// Indexing for constant 4th order tensor
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
    /// 4th-order tensor copy assignment
    ///
    Tensor4<T, 3> &
    operator=(Tensor4<T, 3> const & A);

    ///
    /// 4th-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor4<T, 3> &
    operator+=(Tensor4<T, 3> const & A);

    ///
    /// 4th-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor4<T, 3> &
    operator-=(Tensor4<T, 3> const & A);

    ///
    /// Fill 4th-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    T
    e[3][3][3][3];

  };

  ///
  /// Fourth order tensor in R^2.
  ///
  template<typename T>
  class Tensor4<T, 2>
  {
  public:

    ///
    /// 4th-order tensor constructor with NaNs
    ///
    Tensor4();

    ///
    /// 4th-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor4(T const & s);

    ///
    /// Copy constructor
    /// 4th-order tensor constructor with 4th-order tensor
    /// \param A from which components are copied
    ///
    Tensor4(Tensor4<T, 2> const & A);

    ///
    /// 4th-order tensor simple destructor
    ///
    ~Tensor4();

    ///
    /// Indexing for constant 4th order tensor
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
    /// 4th-order tensor copy assignment
    ///
    Tensor4<T, 2> &
    operator=(Tensor4<T, 2> const & A);

    ///
    /// 4th-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor4<T, 2> &
    operator+=(Tensor4<T, 2> const & A);

    ///
    /// 4th-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor4<T, 2> &
    operator-=(Tensor4<T, 2> const & A);

    ///
    /// Fill 4th-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    T
    e[2][2][2][2];

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
  template<typename T, Index N>
  Vector<T, N>
  operator+(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  Vector<T, 3>
  operator+(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  Vector<T, 2>
  operator+(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Vector substraction
  /// \param u
  /// \param v the operands
  /// \return \f$ u - v \f$
  ///
  template<typename T, Index N>
  Vector<T, N>
  operator-(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  Vector<T, 3>
  operator-(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  Vector<T, 2>
  operator-(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Vector minus
  /// \param u
  /// \return \f$ -u \f$
  ///
  template<typename T, Index N>
  Vector<T, N>
  operator-(Vector<T, N> const & u);

  template<typename T>
  Vector<T, 3>
  operator-(Vector<T, 3> const & u);

  template<typename T>
  Vector<T, 2>
  operator-(Vector<T, 2> const & u);

  ///
  /// Vector dot product
  /// \param u
  /// \param v the operands
  /// \return \f$ u \cdot v \f$
  ///
  template<typename T, Index N>
  T
  operator*(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  T
  operator*(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  T
  operator*(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Vector equality tested by components
  /// \param u
  /// \param v the operands
  /// \return \f$ u \equiv v \f$
  ///
  template<typename T, Index N>
  bool
  operator==(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  bool
  operator==(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  bool
  operator==(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Vector inequality tested by components
  /// \param u
  /// \param v the operands
  /// \return \f$ u \neq v \f$
  ///
  template<typename T, Index N>
  bool
  operator!=(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  bool
  operator!=(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  bool
  operator!=(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Scalar vector product
  /// \param s scalar factor
  /// \param u vector factor
  /// \return \f$ s u \f$
  ///
  template<typename T, Index N, typename S>
  Vector<T, N>
  operator*(S const & s, Vector<T, N> const & u);

  template<typename T, typename S>
  Vector<T, 3>
  operator*(S const & s, Vector<T, 3> const & u);

  template<typename T, typename S>
  Vector<T, 2>
  operator*(S const & s, Vector<T, 2> const & u);

  ///
  /// Vector scalar product
  /// \param u vector factor
  /// \param s scalar factor
  /// \return \f$ s u \f$
  ///
  template<typename T, Index N, typename S>
  Vector<T, N>
  operator*(Vector<T, N> const & u, S const & s);

  template<typename T, typename S>
  Vector<T, 3>
  operator*(Vector<T, 3> const & u, S const & s);

  template<typename T, typename S>
  Vector<T, 2>
  operator*(Vector<T, 2> const & u, S const & s);

  ///
  /// Vector scalar division
  /// \param u vector
  /// \param s scalar that divides each component of vector
  /// \return \f$ u / s \f$
  ///
  template<typename T, Index N, typename S>
  Vector<T, N>
  operator/(Vector<T, N> const & u, S const & s);

  template<typename T, typename S>
  Vector<T, 3>
  operator/(Vector<T, 3> const & u, S const & s);

  template<typename T, typename S>
  Vector<T, 2>
  operator/(Vector<T, 2> const & u, S const & s);

  ///
  /// Vector dot product
  /// \param u
  /// \param v operands
  /// \return \f$ u \cdot v \f$
  ///
  template<typename T, Index N>
  T
  dot(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  T
  dot(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  T
  dot(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Cross product only valid for R^3.
  /// R^N and R^2 will produce an error.
  /// \param u
  /// \param v operands
  /// \return \f$ u \times v \f$
  ///
  template<typename T, Index N>
  Vector<T, N>
  cross(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  Vector<T, 3>
  cross(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  Vector<T, 2>
  cross(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Vector 2-norm
  /// \return \f$ \sqrt{u \cdot u} \f$
  ///
  template<typename T, Index N>
  T
  norm(Vector<T, N> const & u);

  template<typename T>
  T
  norm(Vector<T, 3> const & u);

  template<typename T>
  T
  norm(Vector<T, 2> const & u);

  ///
  /// Vector 1-norm
  /// \return \f$ |u_0|+|u_1|+|u_2| \f$
  ///
  template<typename T, Index N>
  T
  norm_1(Vector<T, N> const & u);

  template<typename T>
  T
  norm_1(Vector<T, 3> const & u);

  template<typename T>
  T
  norm_1(Vector<T, 2> const & u);

  ///
  /// Vector infinity-norm
  /// \return \f$ \max(|u_0|,|u_1|,|u_2|) \f$
  ///
  template<typename T, Index N>
  T
  norm_infinity(Vector<T, N> const & u);

  template<typename T>
  T
  norm_infinity(Vector<T, 3> const & u);

  template<typename T>
  T
  norm_infinity(Vector<T, 2> const & u);

  ///
  /// Tensor addition
  /// \return \f$ A + B \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  operator+(Tensor<T, N> const & A, Tensor<T, N> const & B);

  template<typename T>
  Tensor<T, 3>
  operator+(Tensor<T, 3> const & A, Tensor<T, 3> const & B);

  template<typename T>
  Tensor<T, 2>
  operator+(Tensor<T, 2> const & A, Tensor<T, 2> const & B);

  ///
  /// Tensor substraction
  /// \return \f$ A - B \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  operator-(Tensor<T, N> const & A, Tensor<T, N> const & B);

  template<typename T>
  Tensor<T, 3>
  operator-(Tensor<T, 3> const & A, Tensor<T, 3> const & B);

  template<typename T>
  Tensor<T, 2>
  operator-(Tensor<T, 2> const & A, Tensor<T, 2> const & B);

  ///
  /// Tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  operator-(Tensor<T, N> const & A);

  template<typename T>
  Tensor<T, 3>
  operator-(Tensor<T, 3> const & A);

  template<typename T>
  Tensor<T, 2>
  operator-(Tensor<T, 2> const & A);

  ///
  /// Tensor equality
  /// Tested by components
  /// \return \f$ A \equiv B \f$
  ///
  template<typename T, Index N>
  bool
  operator==(Tensor<T, N> const & A, Tensor<T, N> const & B);

  template<typename T>
  bool
  operator==(Tensor<T, 3> const & A, Tensor<T, 3> const & B);

  template<typename T>
  bool
  operator==(Tensor<T, 2> const & A, Tensor<T, 2> const & B);

  ///
  /// Tensor inequality
  /// Tested by components
  /// \return \f$ A \neq B \f$
  ///
  template<typename T, Index N>
  bool
  operator!=(Tensor<T, N> const & A, Tensor<T, N> const & B);

  template<typename T>
  bool
  operator!=(Tensor<T, 3> const & A, Tensor<T, 3> const & B);

  template<typename T>
  bool
  operator!=(Tensor<T, 2> const & A, Tensor<T, 2> const & B);

  ///
  /// Scalar tensor product
  /// \param s scalar
  /// \param A tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, Index N, typename S>
  Tensor<T, N>
  operator*(S const & s, Tensor<T, N> const & A);

  template<typename T, typename S>
  Tensor<T, 3>
  operator*(S const & s, Tensor<T, 3> const & A);

  template<typename T, typename S>
  Tensor<T, 2>
  operator*(S const & s, Tensor<T, 2> const & A);

  ///
  /// Tensor scalar product
  /// \param A tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, Index N, typename S>
  Tensor<T, N>
  operator*(Tensor<T, N> const & A, S const & s);

  template<typename T, typename S>
  Tensor<T, 3>
  operator*(Tensor<T, 3> const & A, S const & s);

  template<typename T, typename S>
  Tensor<T, 2>
  operator*(Tensor<T, 2> const & A, S const & s);

  ///
  /// Scalar 3rd-order tensor product
  /// \param s scalar
  /// \param A 3rd-order tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, Index N, typename S>
  Tensor3<T, N>
  operator*(S const & s, Tensor3<T, N> const & A);

  // No specialization

  ///
  /// 3rd-order tensor scalar product
  /// \param A 3rd-order tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, Index N, typename S>
  Tensor3<T, N>
  operator*(Tensor3<T, N> const & A, S const & s);

  // No specialization

  ///
  /// 3rd-order tensor vector product
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dot(Tensor3<T, N> const & A, Vector<T, N> const & u);

  // No specialization

  ///
  /// vector 3rd-order tensor product
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ u A \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dot(Vector<T, N> const & u, Tensor3<T, N> const & A);

  // No specialization

  ///
  /// 3rd-order tensor vector product2 (contract 2nd index)
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dot2(Tensor3<T, N> const & A, Vector<T, N> const & u);

  // No specialization

  ///
  /// vector 3rd-order tensor product2 (contract 2nd index)
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ u A \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dot2(Vector<T, N> const & u, Tensor3<T, N> const & A);

  // No specialization

  ///
  /// Scalar 4th-order tensor product
  /// \param s scalar
  /// \param A 4th-order tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, Index N, typename S>
  Tensor4<T, N>
  operator*(S const & s, Tensor4<T, N> const & A);

  // No specialization

  ///
  /// 4th-order tensor scalar product
  /// \param A 4th-order tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, Index N, typename S>
  Tensor4<T, N>
  operator*(Tensor4<T, N> const & A, S const & s);

  // No specialization

  ///
  /// Tensor vector product v = A u
  /// \param A tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T, Index N>
  Vector<T, N>
  dot(Tensor<T, N> const & A, Vector<T, N> const & u);

  template<typename T>
  Vector<T, 3>
  dot(Tensor<T, 3> const & A, Vector<T, 3> const & u);

  template<typename T>
  Vector<T, 2>
  dot(Tensor<T, 2> const & A, Vector<T, 2> const & u);

  ///
  /// Vector tensor product v = u A
  /// \param A tensor
  /// \param u vector
  /// \return \f$ u A = A^T u \f$
  ///
  template<typename T, Index N>
  Vector<T, N>
  dot(Vector<T, N> const & u, Tensor<T, N> const & A);

  template<typename T>
  Vector<T, 3>
  dot(Vector<T, 3> const & u, Tensor<T, 3> const & A);

  template<typename T>
  Vector<T, 2>
  dot(Vector<T, 2> const & u, Tensor<T, 2> const & A);

  ///
  /// Tensor vector product v = A u
  /// \param A tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T, Index N>
  Vector<T, N>
  operator*(Tensor<T, N> const & A, Vector<T, N> const & u);

  template<typename T>
  Vector<T, 3>
  operator*(Tensor<T, 3> const & A, Vector<T, 3> const & u);

  template<typename T>
  Vector<T, 2>
  operator*(Tensor<T, 2> const & A, Vector<T, 2> const & u);

  ///
  /// Vector tensor product v = u A
  /// \param A tensor
  /// \param u vector
  /// \return \f$ u A = A^T u \f$
  ///
  template<typename T, Index N>
  Vector<T, N>
  operator*(Vector<T, N> const & u, Tensor<T, N> const & A);

  template<typename T>
  Vector<T, 3>
  operator*(Vector<T, 3> const & u, Tensor<T, 3> const & A);

  template<typename T>
  Vector<T, 2>
  operator*(Vector<T, 2> const & u, Tensor<T, 2> const & A);

  ///
  /// Tensor dot product C = A B
  /// \return \f$ A \cdot B \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  operator*(Tensor<T, N> const & A, Tensor<T, N> const & B);

  template<typename T>
  Tensor<T, 3>
  operator*(Tensor<T, 3> const & A, Tensor<T, 3> const & B);

  template<typename T>
  Tensor<T, 2>
  operator*(Tensor<T, 2> const & A, Tensor<T, 2> const & B);

  ///
  /// Tensor tensor product C = A B
  /// \param A tensor
  /// \param B tensor
  /// \return a tensor \f$ A \cdot B \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dot(Tensor<T, N> const & A, Tensor<T, N> const & B);

  template<typename T>
  Tensor<T, 3>
  dot(Tensor<T, 3> const & A, Tensor<T, 3> const & B);

  template<typename T>
  Tensor<T, 2>
  dot(Tensor<T, 2> const & A, Tensor<T, 2> const & B);

  ///
  /// Tensor tensor double dot product (contraction)
  /// \param A tensor
  /// \param B tensor
  /// \return a scalar \f$ A : B \f$
  ///
  template<typename T, Index N>
  T
  dotdot(Tensor<T, N> const & A, Tensor<T, N> const & B);

  template<typename T>
  T
  dotdot(Tensor<T, 3> const & A, Tensor<T, 3> const & B);

  template<typename T>
  T
  dotdot(Tensor<T, 2> const & A, Tensor<T, 2> const & B);

  ///
  /// Tensor Frobenius norm
  /// \return \f$ \sqrt{A:A} \f$
  ///
  template<typename T, Index N>
  T
  norm(Tensor<T, N> const & A);

  template<typename T>
  T
  norm(Tensor<T, 3> const & A);

  template<typename T>
  T
  norm(Tensor<T, 2> const & A);

  ///
  /// Tensor 1-norm
  /// \return \f$ \max_{j \in {0,1,2}}\Sigma_{i=0}^2 |A_{ij}| \f$
  ///
  template<typename T, Index N>
  T
  norm_1(Tensor<T, N> const & A);

  template<typename T>
  T
  norm_1(Tensor<T, 3> const & A);

  template<typename T>
  T
  norm_1(Tensor<T, 2> const & A);

  ///
  /// Tensor infinity-norm
  /// \return \f$ \max_{i \in {0,1,2}}\Sigma_{j=0}^2 |A_{ij}| \f$
  ///
  template<typename T, Index N>
  T
  norm_infinity(Tensor<T, N> const & A);

  template<typename T>
  T
  norm_infinity(Tensor<T, 3> const & A);

  template<typename T>
  T
  norm_infinity(Tensor<T, 2> const & A);

  ///
  /// Dyad
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dyad(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  Tensor<T, 3>
  dyad(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  Tensor<T, 2>
  dyad(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Bun operator, just for Jay
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  bun(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  Tensor<T, 3>
  bun(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  Tensor<T, 2>
  bun(Vector<T, 2> const & u, Vector<T, 2> const & v);

  ///
  /// Tensor product
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  tensor(Vector<T, N> const & u, Vector<T, N> const & v);

  template<typename T>
  Tensor<T, 3>
  tensor(Vector<T, 3> const & u, Vector<T, 3> const & v);

  template<typename T>
  Tensor<T, 2>
  tensor(Vector<T, 2> const & u, Vector<T, 2> const & v);


  ///
  /// Diagonal tensor from vector
  /// \param v vector
  /// \return A = diag(v)
  ///
  template<typename T, Index N>
  Tensor<T, N>
  diag(Vector<T, N> const & v);

  template<typename T>
  Tensor<T, 3>
  diag(Vector<T, 3> const & v);

  template<typename T>
  Tensor<T, 2>
  diag(Vector<T, 2> const & v);

  ///
  /// Diagonal of tensor in a vector
  /// \param A tensor
  /// \return v = diag(A)
  ///
  template<typename T, Index N>
  Vector<T, N>
  diag(Tensor<T, N> const & A);

  template<typename T>
  Vector<T, 3>
  diag(Tensor<T, 3> const & A);

  template<typename T>
  Vector<T, 2>
  diag(Tensor<T, 2> const & A);

  ///
  /// Zero 2nd-order tensor
  /// All components are zero
  ///
  template<typename T, Index N>
  const Tensor<T, N>
  zero();

  template<typename T>
  const Tensor<T, 3>
  zero();

  template<typename T>
  const Tensor<T, 2>
  zero();

  ///
  /// 2nd-order identity tensor
  ///
  template<typename T, Index N>
  const Tensor<T, N>
  identity();

  template<typename T>
  const Tensor<T, 3>
  identity();

  template<typename T>
  const Tensor<T, 2>
  identity();

  ///
  /// 2nd-order identity tensor, à la Matlab
  ///
  template<typename T, Index N>
  const Tensor<T, N>
  eye();

  template<typename T>
  const Tensor<T, 3>
  eye();

  template<typename T>
  const Tensor<T, 2>
  eye();

  ///
  /// 2nd-order tensor transpose
  ///
  template<typename T, Index N>
  Tensor<T, N>
  transpose(Tensor<T, N> const & A);

  template<typename T>
  Tensor<T, 3>
  transpose(Tensor<T, 3> const & A);

  template<typename T>
  Tensor<T, 2>
  transpose(Tensor<T, 2> const & A);

  ///
  /// Symmetric part of 2nd-order tensor
  /// \return \f$ \frac{1}{2}(A + A^T) \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  symm(Tensor<T, N> const & A);

  template<typename T>
  Tensor<T, 3>
  symm(Tensor<T, 3> const & A);

  template<typename T>
  Tensor<T, 2>
  symm(Tensor<T, 2> const & A);

  ///
  /// Skew symmetric part of 2nd-order tensor
  /// \return \f$ \frac{1}{2}(A - A^T) \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  skew(Tensor<T, N> const & A);

  template<typename T>
  Tensor<T, 3>
  skew(Tensor<T, 3> const & A);

  template<typename T>
  Tensor<T, 2>
  skew(Tensor<T, 2> const & A);

  ///
  /// Skew symmetric 2nd-order tensor from vector valid for R^3 only.
  /// R^N and R^2 will produce an error
  /// \param u vector
  /// \return \f$ {{0, -u_2, u_1}, {u_2, 0, -u_0}, {-u_1, u+0, 0}} \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  skew(Vector<T, N> const & u);

  template<typename T>
  Tensor<T, 3>
  skew(Vector<T, 3> const & u);

  template<typename T>
  Tensor<T, 2>
  skew(Vector<T, 2> const & u);

  ///
  /// Volumetric part of 2nd-order tensor  
  /// \param A tensor
  /// \return \f$ \frac{1}{3} \mathrm{tr}\:A I \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  vol(Tensor<T, N> const & A);

  template<typename T>
  Tensor<T, 3>
  vol(Tensor<T, 3> const & A);

  template<typename T>
  Tensor<T, 2>
  vol(Tensor<T, 2> const & A);

  ///
  /// Deviatoric part of 2nd-order tensor
  /// \param A tensor
  /// \return \f$ A - vol(A) \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dev(Tensor<T, N> const & A);

  template<typename T>
  Tensor<T, 3>
  dev(Tensor<T, 3> const & A);

  template<typename T>
  Tensor<T, 2>
  dev(Tensor<T, 2> const & A);

  ///
  /// 2nd-order tensor inverse
  /// \param A nonsingular tensor
  /// \return \f$ A^{-1} \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  inverse(Tensor<T, N> const & A);

  template<typename T>
  Tensor<T, 3>
  inverse(Tensor<T, 3> const & A);

  template<typename T>
  Tensor<T, 2>
  inverse(Tensor<T, 2> const & A);

  ///
  /// Subtensor
  /// \param A tensor
  /// \param i index
  /// \param j index
  /// \return Subtensor with i-row and j-col deleted.
  ///
  template<typename T, Index N>
  Tensor<T, N - 1>
  subtensor(Tensor<T, N> const & A, Index i, Index j);

  template<typename T>
  Tensor<T, 2>
  subtensor(Tensor<T, 3> const & A, Index i, Index j);

  template<typename T>
  Tensor<T, 1>
  subtensor(Tensor<T, 2> const & A, Index i, Index j);

  ///
  /// Determinant
  /// \param A tensor
  /// \return \f$ \det A \f$
  ///
  template<typename T, Index N>
  T
  det(Tensor<T, N> const & A);

  template<typename T>
  T
  det(Tensor<T, 3> const & A);

  template<typename T>
  T
  det(Tensor<T, 2> const & A);

  ///
  /// Trace
  /// \param A tensor
  /// \return \f$ A:I \f$
  ///
  template<typename T, Index N>
  T
  trace(Tensor<T, N> const & A);

  template<typename T>
  T
  trace(Tensor<T, 3> const & A);

  template<typename T>
  T
  trace(Tensor<T, 2> const & A);

  ///
  /// First invariant, trace
  /// \param A tensor
  /// \return \f$ I_A = A:I \f$
  ///
  template<typename T, Index N>
  T
  I1(Tensor<T, N> const & A);

  template<typename T>
  T
  I1(Tensor<T, 3> const & A);

  template<typename T>
  T
  I1(Tensor<T, 2> const & A);

  ///
  /// Second invariant
  /// \param A tensor
  /// \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
  ///
  template<typename T, Index N>
  T
  I2(Tensor<T, N> const & A);

  template<typename T>
  T
  I2(Tensor<T, 3> const & A);

  template<typename T>
  T
  I2(Tensor<T, 2> const & A);

  ///
  /// Third invariant
  /// \param A tensor
  /// \return \f$ III_A = \det A \f$
  ///
  template<typename T, Index N>
  T
  I3(Tensor<T, N> const & A);

  template<typename T>
  T
  I3(Tensor<T, 3> const & A);

  template<typename T>
  T
  I3(Tensor<T, 2> const & A);

  ///
  /// Exponential map by Taylor series, radius of convergence is infinity
  /// \param A tensor
  /// \return \f$ \exp A \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  exp(Tensor<T, N> const & A);

  // No specialization for the moment. Series applies to all dimensions.

  ///
  /// Logarithmic map by Taylor series, converges for \f$ |A-I| < 1 \f$
  /// \param A tensor
  /// \return \f$ \log A \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  log(Tensor<T, N> const & A);

  // No specialization for the moment. Series applies to all dimensions.

  ///
  /// Logarithmic map of a rotation
  /// \param R with \f$ R \in SO(3) \f$
  /// \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  log_rotation(Tensor<T, N> const & R);

  template<typename T>
  Tensor<T, 3>
  log_rotation(Tensor<T, 3> const & R);

  template<typename T>
  Tensor<T, 2>
  log_rotation(Tensor<T, 2> const & R);

  ///
  /// Logarithmic map of a 180 degree rotation
  /// \param R with \f$ R \in SO(3) \f$
  /// \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  log_rotation_pi(Tensor<T, N> const & R);

  template<typename T>
  Tensor<T, 3>
  log_rotation_pi(Tensor<T, 3> const & R);

  template<typename T>
  Tensor<T, 2>
  log_rotation_pi(Tensor<T, 2> const & R);

  /// Gaussian Elimination with partial pivot
  /// \param A
  /// \return \f$ xvec \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  gaussian_elimination(Tensor<T, N> const & A);

  // No specialization for the time being.

  /// Apply Givens-Jacobi rotation on the left in place.
  /// \param c and s for a rotation G in form [c, s; -s, c]
  /// \param A
  ///
  template<typename T, Index N>
  void
  givens_left(T const & c, T const & s, Index i, Index k, Tensor<T, N> & A);

  // No specialization for the time being.

  /// Apply Givens-Jacobi rotation on the right in place.
  /// \param A
  /// \param c and s for a rotation G in form [c, s; -s, c]
  ///
  template<typename T, Index N>
  void
  givens_right(T const & c, T const & s, Index i, Index k, Tensor<T, N> & A);

  // No specialization for the time being.

  ///
  /// Exponential map of a skew-symmetric tensor
  /// \param r \f$ r \in so(3) \f$
  /// \return \f$ R = \exp R \f$ with \f$ R \in SO(3) \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  exp_skew_symmetric(Tensor<T, N> const & r);

  template<typename T>
  Tensor<T, 3>
  exp_skew_symmetric(Tensor<T, 3> const & r);

  template<typename T>
  Tensor<T, 2>
  exp_skew_symmetric(Tensor<T, 2> const & r);

  ///
  /// Off-diagonal norm. Useful for SVD and other algorithms
  /// that rely on Jacobi-type procedures.
  /// \param A
  /// \return \f$ \sqrt(\sum_i \sum_{j, j\neq i} a_{ij}^2) \f$
  ///
  template<typename T, Index N>
  T
  norm_off_diagonal(Tensor<T, N> const & A);

  template<typename T>
  T
  norm_off_diagonal(Tensor<T, 3> const & A);

  template<typename T>
  T
  norm_off_diagonal(Tensor<T, 2> const & A);

  ///
  /// Arg max off-diagonal. Useful for SVD and other algorithms
  /// that rely on Jacobi-type procedures.
  /// \param A
  /// \return \f$ (p,q) = arg max_{i \neq j} |a_{ij}| \f$
  ///
  template<typename T, Index N>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T, N> const & A);

  template<typename T>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T, 3> const & A);

  template<typename T>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T, 2> const & A);

  ///
  /// Singular value decomposition (SVD) for 2x2
  /// bidiagonal matrix. Used for general 2x2 SVD
  /// \param f, g, h where A = [f, g; 0, h]
  /// \return \f$ A = USV^T\f$
  ///
  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
  svd_bidiagonal(T f, T g, T h);

  ///
  /// Singular value decomposition (SVD)
  /// \param A tensor
  /// \return \f$ A = USV^T\f$
  ///
  template<typename T, Index N>
  boost::tuple<Tensor<T, N>, Tensor<T, N>, Tensor<T, N> >
  svd(Tensor<T, N> const & A);

  // No specialization for R^3

  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
  svd(Tensor<T, 2> const & A);

  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
  svd2(Tensor<T, 2> const & A);

  ///
  /// Left polar decomposition
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD
  ///
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N> >
  polar_left(Tensor<T, N> const & F);

  template<typename T>
  std::pair<Tensor<T, 3>, Tensor<T, 3> >
  polar_left(Tensor<T, 3> const & F);

  template<typename T>
  std::pair<Tensor<T, 2>, Tensor<T, 2> >
  polar_left(Tensor<T, 2> const & F);

  ///
  /// Right polar decomposition
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ RU = F \f$ with \f$ R \in SO(3) \f$ and U SPD
  ///
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N> >
  polar_right(Tensor<T, N> const & F);

  template<typename T>
  std::pair<Tensor<T, 3>, Tensor<T, 3> >
  polar_right(Tensor<T, 3> const & F);

  template<typename T>
  std::pair<Tensor<T, 2>, Tensor<T, 2> >
  polar_right(Tensor<T, 2> const & F);

  ///
  /// Left polar decomposition with matrix logarithm for V
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD, and log V
  ///
  template<typename T, Index N>
  boost::tuple<Tensor<T, N>, Tensor<T, N>, Tensor<T, N> >
  polar_left_logV(Tensor<T, N> const & F);

  template<typename T>
  boost::tuple<Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3> >
  polar_left_logV(Tensor<T, 3> const & F);

  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
  polar_left_logV(Tensor<T, 2> const & F);

  ///
  /// Logarithmic map using BCH expansion (3 terms)
  /// \param v tensor
  /// \param r tensor
  /// \return Baker-Campbell-Hausdorff series up to 3 terms
  ///
  template<typename T, Index N>
  Tensor<T, N>
  bch(Tensor<T, N> const & v, Tensor<T, N> const & r);

  // No specialization

  ///
  /// Symmetric Schur algorithm for R^2.
  /// \param \f$ A \in S(2) \f$
  /// \return \f$ c, s \rightarrow [c, -s; s, c]\f diagonalizes A$
  ///
  template<typename T>
  std::pair<T, T>
  schur_sym(Tensor<T, 2> const & A);

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
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N> >
  eig_sym(Tensor<T, N> const & A);

  // No specialization for R^3

  template<typename T>
  std::pair<Tensor<T, 2>, Tensor<T, 2> >
  eig_sym(Tensor<T, 2> const & A);

  ///
  /// Eigenvalue decomposition for SPD 2nd-order tensor
  /// \param A tensor
  /// \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  ///
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N> >
  eig_spd(Tensor<T, N> const & A);

  template<typename T>
  std::pair<Tensor<T, 3>, Tensor<T, 3> >
  eig_spd(Tensor<T, 3> const & A);

  template<typename T>
  std::pair<Tensor<T, 2>, Tensor<T, 2> >
  eig_spd(Tensor<T, 2> const & A);

  ///
  /// 4th-order identity I1
  /// \return \f$ \delta_{ik} \delta_{jl} \f$ such that \f$ A = I_1 A \f$
  ///
  template<typename T, Index N>
  const Tensor4<T, N>
  identity_1();

  // No specialization

  ///
  /// 4th-order identity I2
  /// \return \f$ \delta_{il} \delta_{jk} \f$ such that \f$ A^T = I_2 A \f$
  ///
  template<typename T, Index N>
  const Tensor4<T, N>
  identity_2();

  // No specialization

  ///
  /// 4th-order identity I3
  /// \return \f$ \delta_{ij} \delta_{kl} \f$ such that \f$ I_A I = I_3 A \f$
  ///
  template<typename T, Index N>
  const Tensor4<T, N>
  identity_3();

  // No specialization

  ///
  /// 4th-order tensor vector dot product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ A dot u \f$ as \f$ B_{ijk}=A_{ijkl}u_{l} \f$
  ///
  template<typename T, Index N>
  Tensor3<T, N>
  dot(Tensor4<T, N> const & A, Vector<T, N> const & u);

  // No specialization

  ///
  /// vector 4th-order tensor dot product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ u dot A \f$ as \f$ B_{jkl}=u_{i} A_{ijkl} \f$
  ///
  template<typename T, Index N>
  Tensor3<T, N>
  dot(Vector<T, N> const & u, Tensor4<T, N> const & A);

  // No specialization

  ///
  /// 4th-order tensor vector dot2 product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ A dot2 u \f$ as \f$ B_{ijl}=A_{ijkl}u_{k} \f$
  ///
  template<typename T, Index N>
  Tensor3<T, N>
  dot2(Tensor4<T, N> const & A, Vector<T, N> const & u);

  // No specialization

  ///
  /// vector 4th-order tensor dot2 product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ u dot2 A \f$ as \f$ B_{ikl}=u_{j}A_{ijkl} \f$
  ///
  template<typename T, Index N>
  Tensor3<T, N>
  dot2(Vector<T, N> const & u, Tensor4<T, N> const & A);

  // No specialization

  ///
  /// 4th-order tensor 2nd-order tensor double dot product
  /// \param A 4th-order tensor
  /// \param B 2nd-order tensor
  /// \return 2nd-order tensor \f$ A:B \f$ as \f$ C_{ij}=A_{ijkl}B_{kl} \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dotdot(Tensor4<T, N> const & A, Tensor<T, N> const & B);

  // No specialization

  ///
  /// 2nd-order tensor 4th-order tensor double dot product
  /// \param B 2nd-order tensor
  /// \param A 4th-order tensor
  /// \return 2nd-order tensor \f$ B:A \f$ as \f$ C_{kl}=A_{ijkl}B_{ij} \f$
  ///
  template<typename T, Index N>
  Tensor<T, N>
  dotdot(Tensor<T, N> const & B, Tensor4<T, N> const & A);

  // No specialization

  ///
  /// 2nd-order tensor 2nd-order tensor tensor product
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \otimes B \f$
  ///
  template<typename T, Index N>
  Tensor4<T, N>
  tensor(Tensor<T, N> const & A, Tensor<T, N> const & B);

  // No specialization

  ///
  /// odot operator useful for \f$ \frac{\partial A^{-1}}{\partial A} \f$
  /// see Holzapfel eqn 6.165
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \odot B \f$ which is
  /// \f$ C_{ijkl} = \frac{1}{2}(A_{ik} B_{jl} + A_{il} B_{jk}) \f$
  ///
  template<typename T, Index N>
  Tensor4<T, N>
  odot(Tensor<T, N> const & A, Tensor<T, N> const & B);

  // No specialization

  ///
  /// 3rd-order tensor addition
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename T, Index N>
  Tensor3<T, N>
  operator+(Tensor3<T, N> const & A, Tensor3<T, N> const & B);

  ///
  /// 3rd-order tensor substraction
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename T, Index N>
  Tensor3<T, N>
  operator-(Tensor3<T, N> const & A, Tensor3<T, N> const & B);

  ///
  /// 3rd-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T, Index N>
  Tensor3<T, N>
  operator-(Tensor3<T, N> const & A);

  ///
  /// 3rd-order tensor equality
  /// Tested by components
  ///
  template<typename T, Index N>
  bool
  operator==(Tensor3<T, N> const & A, Tensor3<T, N> const & B);

  ///
  /// 3rd-order tensor inequality
  /// Tested by components
  ///
  template<typename T, Index N>
  bool
  operator!=(Tensor3<T, N> const & A, Tensor3<T, N> const & B);

  ///
  /// 4th-order tensor addition
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename T, Index N>
  Tensor4<T, N>
  operator+(Tensor4<T, N> const & A, Tensor4<T, N> const & B);

  ///
  /// 4th-order tensor substraction
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename T, Index N>
  Tensor4<T, N>
  operator-(Tensor4<T, N> const & A, Tensor4<T, N> const & B);

  ///
  /// 4th-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T, Index N>
  Tensor4<T, N>
  operator-(Tensor4<T, N> const & A);

  ///
  /// 4th-order equality
  /// Tested by components
  ///
  template<typename T, Index N>
  bool
  operator==(Tensor4<T, N> const & A, Tensor4<T, N> const & B);

  ///
  /// 4th-order inequality
  /// Tested by components
  ///
  template<typename T, Index N>
  bool
  operator!=(Tensor4<T, N> const & A, Tensor4<T, N> const & B);

  ///
  /// Vector input
  /// \param u vector
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Vector<T, N> & u);

  template<typename T>
  std::istream &
  operator>>(std::istream & is, Vector<T, 3> & u);

  template<typename T>
  std::istream &
  operator>>(std::istream & is, Vector<T, 2> & u);


  ///
  /// Vector output
  /// \param u vector
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Vector<T, N> const & u);

  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Vector<T, 3> const & u);

  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Vector<T, 2> const & u);

  ///
  /// Tensor input
  /// \param A tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Tensor<T, N> & A);

  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor<T, 3> & A);

  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor<T, 2> & A);

  ///
  /// Tensor output
  /// \param A tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T, N> const & A);

  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T, 3> const & A);

  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T, 2> const & A);

  ///
  /// 3rd-order tensor input
  /// \param A 3rd-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Tensor3<T, N> & A);

  // No specialization

  ///
  /// 3rd-order tensor output
  /// \param A 3rd-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<T, N> const & A);

  // No specialization

  ///
  /// 4th-order input
  /// \param A 4th-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Tensor4<T, N> & A);

  // No specialization

  ///
  /// 4th-order output
  /// \param A 4th-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<T, N> const & A);

  // No specialization

} // namespace LCM

#include "Tensor.i.cc"
#include "Tensor.t.cc"

#endif //LCM_Tensor_h
