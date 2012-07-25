///
/// \file Tensor.i.cc
/// First cut of LCM small tensor utilities. Inline functions.
/// \author Alejandro Mota
/// \author Jake Ostien
///
#if !defined(LCM_Tensor_i_cc)
#define LCM_Tensor_i_cc

namespace LCM {

  //
  // Sign function
  //
  template <typename T> int sgn(T const & s) {
    return int(T(0) < s) - int(s < T(0));
  }

  //
  // NaN function. Necessary to choose the proper underlying NaN
  // for non-floating-point types.
  // Assumption: non-floating-point types have a typedef that
  // determines the underlying floating-point type.
  //
  template<typename T>
  typename boost::enable_if<boost::is_floating_point<T> >::type
  not_a_number()
  {
    return std::numeric_limits<T>::quiet_NaN();
  }

  template<typename T>
  typename boost::disable_if<boost::is_floating_point<T> >::type
  not_a_number()
  {
    return std::numeric_limits<typename T::value_type>::quiet_NaN();
  }

  //
  // Machine epsilon function. Necessary to choose the proper underlying
  // machine epsilon for non-floating-point types.
  // Assumption: non-floating-point types have a typedef that
  // determines the underlying floating-point type.
  //
  template<typename T>
  typename boost::enable_if<boost::is_floating_point<T> >::type
  machine_epsilon()
  {
    return std::numeric_limits<T>::epsilon();
  }

  template<typename T>
  typename boost::disable_if<boost::is_floating_point<T> >::type
  machine_epsilon()
  {
    return std::numeric_limits<typename T::value_type>::epsilon();
  }

  //
  // R^N default constructor that initializes to NaNs
  //
  template<typename T, Index N>
  inline
  Vector<T, N>::Vector()
  {
    e.resize(N);
    for (Index i =0; i < N; ++i) {
      e[i] = std::numeric_limits<T>::quiet_NaN();
    }

    return;
  }

  //
  // R^3 default constructor that initializes to NaNs
  //
  template<typename T>
  inline
  Vector<T, 3>::Vector()
  {
    e[0] = std::numeric_limits<T>::quiet_NaN();
    e[1] = std::numeric_limits<T>::quiet_NaN();
    e[2] = std::numeric_limits<T>::quiet_NaN();

    return;
  }

  //
  // R^2 default constructor that initializes to NaNs
  //
  template<typename T>
  inline
  Vector<T, 2>::Vector()
  {
    e[0] = std::numeric_limits<T>::quiet_NaN();
    e[1] = std::numeric_limits<T>::quiet_NaN();

    return;
  }

  //
  // R^N create vector from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T, Index N>
  inline
  Vector<T, N>::Vector(T const & s)
  {
    e.resize(N);
    for (Index i =0; i < N; ++i) {
      e[i] = s;
    }

    return;
  }

  //
  // R^3 create vector from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T>
  inline
  Vector<T, 3>::Vector(T const & s)
  {
    e[0] = s;
    e[1] = s;
    e[2] = s;

    return;
  }

  //
  // R^2 create vector from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T>
  inline
  Vector<T, 2>::Vector(T const & s)
  {
    e[0] = s;
    e[1] = s;

    return;
  }

  //
  // R^N create vector specifying components
  // \param s0, ... are the vector components in the canonical basis
  //
  template<typename T, Index N>
  inline
  Vector<T, N>::Vector(T const & s0, ...)
  {

    e.resize(N);
    e[0] = s0;

    va_list arg_list;
    va_start(arg_list, s0);
    for (Index i = 1; i < N; ++i) {
      T const & s = va_arg(arg_list, T);
      e[i] = s;
    }
    va_end(arg_list);

    return;
  }

  //
  // R^3 create vector specifying components
  // \param s0, ... are the vector components in the canonical basis
  //
  template<typename T>
  inline
  Vector<T, 3>::Vector(T const & s0, T const & s1, T const & s2)
  {
    e[0] = s0;
    e[1] = s1;
    e[2] = s2;

    return;
  }

  //
  // R^2 create vector specifying components
  // \param s0, ... are the vector components in the canonical basis
  //
  template<typename T>
  inline
  Vector<T, 2>::Vector(T const & s0, T const & s1)
  {
    e[0] = s0;
    e[1] = s1;

    return;
  }

  //
  // R^N create vector from array - const version
  // \param data_ptr
  //
  template<typename T, Index N>
  inline
  Vector<T, N>::Vector(T const * data_ptr)
  {
    assert(data_ptr != NULL);

    for (Index i = 0; i < N; ++i) {
      e[i] = data_ptr[i];
    }

    return;
  }

  //
  // R^3 create vector from array - const version
  // \param data_ptr
  //
  template<typename T>
  inline
  Vector<T, 3>::Vector(T const * data_ptr)
  {
    assert(data_ptr != NULL);
    e[0] = data_ptr[0];
    e[1] = data_ptr[1];
    e[2] = data_ptr[2];

    return;
  }

  //
  // R^2 create vector from array - const version
  // \param data_ptr
  //
  template<typename T>
  inline
  Vector<T, 2>::Vector(T const * data_ptr)
  {
    assert(data_ptr != NULL);
    e[0] = data_ptr[0];
    e[1] = data_ptr[1];

    return;
  }

  //
  // R^N create vector from array
  // \param data_ptr
  //
  template<typename T, Index N>
  inline
  Vector<T, N>::Vector(T * data_ptr)
  {
    assert(data_ptr != NULL);

    for (Index i = 0; i < N; ++i) {
      e[i] = data_ptr[i];
    }

    return;
  }

  //
  // R^3 create vector from array
  // \param data_ptr
  //
  template<typename T>
  inline
  Vector<T, 3>::Vector(T * data_ptr)
  {
    assert(data_ptr != NULL);
    e[0] = data_ptr[0];
    e[1] = data_ptr[1];
    e[2] = data_ptr[2];

    return;
  }

  //
  // R^2 Create vector from array
  // \param data_ptr
  //
  template<typename T>
  inline
  Vector<T, 2>::Vector(T * data_ptr)
  {
    assert(data_ptr != NULL);
    e[0] = data_ptr[0];
    e[1] = data_ptr[1];

    return;
  }

  //
  // R^N copy constructor
  // \param v the values of its componets are copied to the new vector
  //
  template<typename T, Index N>
  inline
  Vector<T, N>::Vector(Vector<T, N> const & v)
  {
    for (Index i = 0; i < N; ++i) {
      e[i] = v.e[i];
    }

    return;
  }

  //
  // R^3 copy constructor
  // \param v the values of its componets are copied to the new vector
  //
  template<typename T>
  inline
  Vector<T, 3>::Vector(Vector<T, 3> const & v)
  {
    e[0] = v.e[0];
    e[1] = v.e[1];
    e[2] = v.e[2];

    return;
  }

  //
  // R^2 copy constructor
  // \param v the values of its componets are copied to the new vector
  //
  template<typename T>
  inline
  Vector<T, 2>::Vector(Vector<T, 2> const & v)
  {
    e[0] = v.e[0];
    e[1] = v.e[1];

    return;
  }

  //
  // R^N simple destructor
  //
  template<typename T, Index N>
  inline
  Vector<T, N>::~Vector()
  {
    return;
  }

  //
  // R^3 simple destructor
  //
  template<typename T>
  inline
  Vector<T, 3>::~Vector()
  {
    return;
  }

  //
  // R^2 simple destructor
  //
  template<typename T>
  inline
  Vector<T, 2>::~Vector()
  {
    return;
  }

  //
  // R^N indexing for constant vector
  // \param i the index
  //
  template<typename T, Index N>
  inline
  const T &
  Vector<T, N>::operator()(const Index i) const
  {
    assert(i < N);
    return e[i];
  }

  //
  // R^3 indexing for constant vector
  // \param i the index
  //
  template<typename T>
  inline
  const T &
  Vector<T, 3>::operator()(const Index i) const
  {
    assert(i < 3);
    return e[i];
  }

  //
  // R^2 indexing for constant vector
  // \param i the index
  //
  template<typename T>
  inline
  const T &
  Vector<T, 2>::operator()(const Index i) const
  {
    assert(i < 2);
    return e[i];
  }

  //
  // R^N vector indexing
  // \param i the index
  //
  template<typename T, Index N>
  inline
  T &
  Vector<T, N>::operator()(const Index i)
  {
    assert(i < N);
    return e[i];
  }

  //
  // R^3 vector indexing
  // \param i the index
  //
  template<typename T>
  inline
  T &
  Vector<T, 3>::operator()(const Index i)
  {
    assert(i < 3);
    return e[i];
  }

  //
  // R^2 vector indexing
  // \param i the index
  //
  template<typename T>
  inline
  T &
  Vector<T, 2>::operator()(const Index i)
  {
    assert(i < 2);
    return e[i];
  }

  //
  // R^N copy assignment
  // \param v the values of its componets are copied to this vector
  //
  template<typename T, Index N>
  inline
  Vector<T, N> &
  Vector<T, N>::operator=(Vector<T, N> const & v)
  {
    if (this != &v) {
      for (Index i = 0; i < N; ++i) {
        e[i] = v.e[i];
      }
    }
    return *this;
  }

  //
  // R^3 copy assignment
  // \param v the values of its componets are copied to this vector
  //
  template<typename T>
  inline
  Vector<T, 3> &
  Vector<T, 3>::operator=(Vector<T, 3> const & v)
  {
    if (this != &v) {
      e[0] = v.e[0];
      e[1] = v.e[1];
      e[2] = v.e[2];
    }
    return *this;
  }

  //
  // R^2 copy assignment
  // \param v the values of its componets are copied to this vector
  //
  template<typename T>
  inline
  Vector<T, 2> &
  Vector<T, 2>::operator=(Vector<T, 2> const & v)
  {
    if (this != &v) {
      e[0] = v.e[0];
      e[1] = v.e[1];
    }
    return *this;
  }

  //
  // R^N vector increment
  // \param v added to currrent vector
  //
  template<typename T, Index N>
  inline
  Vector<T, N> &
  Vector<T, N>::operator+=(Vector<T, N> const & v)
  {
    for (Index i = 0; i < N; ++i) {
      e[i] += v.e[i];
    }

    return *this;
  }

  //
  // R^3 vector increment
  // \param v added to currrent vector
  //
  template<typename T>
  inline
  Vector<T, 3> &
  Vector<T, 3>::operator+=(Vector<T, 3> const & v)
  {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];

    return *this;
  }

  //
  // R^2 vector increment
  // \param v added to currrent vector
  //
  template<typename T>
  inline
  Vector<T, 2> &
  Vector<T, 2>::operator+=(Vector<T, 2> const & v)
  {
    e[0] += v.e[0];
    e[1] += v.e[1];

    return *this;
  }

  //
  // R^N vector decrement
  // \param v substracted from current vector
  //
  template<typename T, Index N>
  inline
  Vector<T, N> &
  Vector<T, N>::operator-=(Vector<T, N> const & v)
  {
    for (Index i = 0; i < N; ++i) {
      e[i] -= v.e[i];
    }

    return *this;
  }

  //
  // R^3 vector decrement
  // \param v substracted from current vector
  //
  template<typename T>
  inline
  Vector<T, 3> &
  Vector<T, 3>::operator-=(Vector<T, 3> const & v)
  {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];

    return *this;
  }

  //
  // R^2 vector decrement
  // \param v substracted from current vector
  //
  template<typename T>
  inline
  Vector<T, 2> &
  Vector<T, 2>::operator-=(Vector<T, 2> const & v)
  {
    e[0] -= v.e[0];
    e[1] -= v.e[1];

    return *this;
  }

  //
  // R^N fill with zeros
  //
  template<typename T, Index N>
  inline
  void
  Vector<T, N>::clear()
  {
    for (Index i = 0; i < N; ++i) {
      e[i] -= 0.0;
    }

    return;
  }

  //
  // R^3 fill with zeros
  //
  template<typename T>
  inline
  void
  Vector<T, 3>::clear()
  {
    e[0] = 0.0;
    e[1] = 0.0;
    e[2] = 0.0;

    return;
  }

  //
  // R^2 fill with zeros
  //
  template<typename T>
  inline
  void
  Vector<T, 2>::clear()
  {
    e[0] = 0.0;
    e[1] = 0.0;

    return;
  }

  //
  // R^N vector addition
  // \param u
  // \param v the operands
  // \return \f$ u + v \f$
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  operator+(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    Vector<T, N> s;

    for (Index i = 0; i < N; ++i) {
      s(i) = u(i) + v(i);
    }

    return s;
  }

  //
  // R^3 vector addition
  // \param u
  // \param v the operands
  // \return \f$ u + v \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  operator+(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    Vector<T, 3> s;

    s(0) = u(0) + v(0);
    s(1) = u(1) + v(1);
    s(2) = u(2) + v(2);

    return s;
  }

  //
  // R^2 vector addition
  // \param u
  // \param v the operands
  // \return \f$ u + v \f$
  //
  template<typename T>
  inline
  Vector<T, 2>
  operator+(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    Vector<T, 2> s;

    s(0) = u(0) + v(0);
    s(1) = u(1) + v(1);

    return s;
  }

  //
  // R^N vector substraction
  // \param u
  // \param v the operands
  // \return \f$ u - v \f$
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  operator-(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    Vector<T, N> s;

    for (Index i = 0; i < N; ++i) {
      s(i) = u(i) - v(i);
    }

    return s;
  }

  //
  // R^3 vector substraction
  // \param u
  // \param v the operands
  // \return \f$ u - v \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  operator-(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    Vector<T, 3> s;

    s(0) = u(0) - v(0);
    s(1) = u(1) - v(1);
    s(2) = u(2) - v(2);

    return s;
  }

  //
  // R^2 vector substraction
  // \param u
  // \param v the operands
  // \return \f$ u - v \f$
  //
  template<typename T>
  inline
  Vector<T, 2>
  operator-(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    Vector<T, 2> s;

    s(0) = u(0) - v(0);
    s(1) = u(1) - v(1);

    return s;
  }

  //
  // R^N vector minus
  // \param u
  // \return \f$ -u \f$
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  operator-(Vector<T, N> const & u)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      v(i) = - u(i);
    }

    return v;
  }

  //
  // R^3 vector minus
  // \param u
  // \return \f$ -u \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  operator-(Vector<T, 3> const & u)
  {
    return Vector<T, 3>(-u(0), -u(1), -u(2));
  }

  //
  // R^2 vector minus
  // \param u
  // \return \f$ -u \f$
  //
  template<typename T>
  inline
  Vector<T, 2>
  operator-(Vector<T, 2> const & u)
  {
    return Vector<T, 2>(-u(0), -u(1));
  }

  //
  // R^N vector dot product
  // \param u
  // \param v the operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T, Index N>
  inline
  T
  operator*(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    return dot(u, v);
  }

  //
  // R^3 vector dot product
  // \param u
  // \param v the operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T>
  inline
  T
  operator*(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    return dot(u, v);
  }

  //
  // R^2 vector dot product
  // \param u
  // \param v the operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T>
  inline
  T
  operator*(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    return dot(u, v);
  }

  //
  // R^N vector equality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \equiv v \f$
  //
  template<typename T, Index N>
  inline
  bool
  operator==(Vector<T, N> const & u, Vector<T, N > const & v)
  {
    bool equal = true;

    for (Index i = 0; i < N; ++i) {
      equal = equal && (v(i) == u(i));
      if (equal == false) break;
    }

    return equal;
  }

  //
  // R^3 vector equality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \equiv v \f$
  //
  template<typename T>
  inline
  bool
  operator==(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    return u(0)==v(0) && u(1)==v(1) && u(2)==v(2);
  }

  //
  // R^2 vector equality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \equiv v \f$
  //
  template<typename T>
  inline
  bool
  operator==(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    return u(0)==v(0) && u(1)==v(1);
  }

  //
  // R^N, vector inequality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \neq v \f$
  //
  template<typename T, Index N>
  inline
  bool
  operator!=(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    return !(u==v);
  }

  //
  // R^3 vector inequality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \neq v \f$
  //
  template<typename T>
  inline
  bool
  operator!=(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    return !(u==v);
  }

  //
  // R^2 vector inequality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \neq v \f$
  //
  template<typename T>
  inline
  bool
  operator!=(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    return !(u==v);
  }

  //
  // R^N scalar vector product
  // \param s scalar factor
  // \param u vector factor
  // \return \f$ s u \f$
  //
  template<typename T, Index N, typename S>
  inline
  Vector<T, N>
  operator*(S const & s, Vector<T, N> const & u)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      v(i) = s * u(i);
    }

    return v;
  }

  //
  // R^3 scalar vector product
  // \param s scalar factor
  // \param u vector factor
  // \return \f$ s u \f$
  //
  template<typename T, typename S>
  inline
  Vector<T, 3>
  operator*(S const & s, Vector<T, 3> const & u)
  {
    return Vector<T, 3>(s * u(0), s * u(1), s * u(2));
  }

  //
  // R^2 scalar vector product
  // \param s scalar factor
  // \param u vector factor
  // \return \f$ s u \f$
  //
  template<typename T, typename S>
  inline
  Vector<T, 2>
  operator*(S const & s, Vector<T, 2> const & u)
  {
    return Vector<T, 2>(s * u(0), s * u(1));
  }

  //
  // R^N vector scalar product
  // \param u vector factor
  // \param s scalar factor
  // \return \f$ s u \f$
  //
  template<typename T, Index N, typename S>
  inline
  Vector<T, N>
  operator*(Vector<T, N> const & u, S const & s)
  {
    return s * u;
  }

  //
  // R^3 vector scalar product
  // \param u vector factor
  // \param s scalar factor
  // \return \f$ s u \f$
  //
  template<typename T, typename S>
  inline
  Vector<T, 3>
  operator*(Vector<T, 3> const & u, S const & s)
  {
    return s * u;
  }

  //
  // R^2 vector scalar product
  // \param u vector factor
  // \param s scalar factor
  // \return \f$ s u \f$
  //
  template<typename T, typename S>
  inline
  Vector<T, 2>
  operator*(Vector<T, 2> const & u, S const & s)
  {
    return s * u;
  }

  //
  // R^N vector scalar division
  // \param u vector
  // \param s scalar that divides each component of vector
  // \return \f$ u / s \f$
  //
  template<typename T, Index N, typename S>
  inline
  Vector<T, N>
  operator/(Vector<T, N> const & u, S const & s)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      v(i) = u(i) / s;
    }

    return v;
  }

  //
  // R^3 vector scalar division
  // \param u vector
  // \param s scalar that divides each component of vector
  // \return \f$ u / s \f$
  //
  template<typename T, typename S>
  inline
  Vector<T, 3>
  operator/(Vector<T, 3> const & u, S const & s)
  {
    return Vector<T, 3>(u(0) / s, u(1) / s, u(2) / s);
  }

  //
  // R^2 vector scalar division
  // \param u vector
  // \param s scalar that divides each component of vector
  // \return \f$ u / s \f$
  //
  template<typename T, typename S>
  inline
  Vector<T, 2>
  operator/(Vector<T, 2> const & u, S const & s)
  {
    return Vector<T, 2>(u(0) / s, u(1) / s);
  }

  //
  // R^N vector dot product
  // \param u
  // \param v operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T, Index N>
  inline
  T
  dot(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    T s = 0.0;

    for (Index i = 0; i < N; ++i) {
      s += u(i) * v(i);
    }

    return s;
  }

  //
  // R^3 vector dot product
  // \param u
  // \param v operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T>
  inline
  T
  dot(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    return u(0)*v(0) + u(1)*v(1) + u(2)*v(2);
  }

  //
  // R^2 vector dot product
  // \param u
  // \param v operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T>
  inline
  T
  dot(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    return u(0)*v(0) + u(1)*v(1);
  }

  //
  // R^N cross product, undefined.
  // \param u
  // \param v operands
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  cross(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    Vector<T, N> w(0.0);

    std::cerr << "ERROR: Cross product undefined for R^" << N << std::endl;
    exit(1);

    return w;
  }

  //
  // R^3 cross product
  // \param u
  // \param v operands
  // \return \f$ u \times v \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  cross(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    Vector<T, 3> w;

    w(0) = u(1)*v(2) - u(2)*v(1);
    w(1) = u(2)*v(0) - u(0)*v(2);
    w(2) = u(0)*v(1) - u(1)*v(0);

    return w;
  }

  //
  // R^2 cross product, undefined.
  // \param u
  // \param v operands
  //
  template<typename T>
  inline
  Vector<T, 2>
  cross(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    Vector<T, 2> w(0.0);

    std::cerr << "ERROR: Cross product undefined for R^2" << std::endl;
    exit(1);

    return w;
  }

  //
  // R^N vector 2-norm
  // \return \f$ \sqrt{u \cdot u} \f$
  //
  template<typename T, Index N>
  inline
  T
  norm(Vector<T, N> const & u)
  {
    return sqrt(dot(u, u));
  }

  //
  // R^3 vector 2-norm
  // \return \f$ \sqrt{u \cdot u} \f$
  //
  template<typename T>
  inline
  T
  norm(Vector<T, 3> const & u)
  {
    return sqrt(u(0)*u(0) + u(1)*u(1) + u(2)*u(2));
  }

  //
  // R^2 vector 2-norm
  // \return \f$ \sqrt{u \cdot u} \f$
  //
  template<typename T>
  inline
  T
  norm(Vector<T, 2> const & u)
  {
    return sqrt(u(0)*u(0) + u(1)*u(1));
  }

  //
  // R^N vector 1-norm
  // \return \f$ \sum_i |u_i| \f$
  //
  template<typename T, Index N>
  inline
  T
  norm_1(Vector<T, N> const & u)
  {
    T s = 0.0;

    for (Index i = 0; i < N; ++i) {
      s += std::abs(u(i));
    }

    return s;
  }

  //
  // R^3 vector 1-norm
  // \return \f$ |u_0|+|u_1|+|u_2| \f$
  //
  template<typename T>
  inline
  T
  norm_1(Vector<T, 3> const & u)
  {
    return std::abs(u(0)) + std::abs(u(1)) + std::abs(u(2));
  }

  //
  // R^2 vector 1-norm
  // \return \f$ |u_0|+|u_1| \f$
  //
  template<typename T>
  inline
  T
  norm_1(Vector<T, 2> const & u)
  {
    return std::abs(u(0)) + std::abs(u(1));
  }

  //
  // R^N vector infinity-norm
  // \return \f$ \max(|u_0|,...|u_i|,...|u_N|) \f$
  //
  template<typename T, Index N>
  inline
  T
  norm_infinity(Vector<T, N> const & u)
  {
    T s = u(0);

    for (Index i = 1; i < N; ++i) {
      s = std::max(s, u(i));
    }

    return s;
  }

  //
  // R^3 vector infinity-norm
  // \return \f$ \max(|u_0|,|u_1|,|u_2|) \f$
  //
  template<typename T>
  inline
  T
  norm_infinity(Vector<T, 3> const & u)
  {
    return std::max(std::max(std::abs(u(0)),std::abs(u(1))),std::abs(u(2)));
  }

  //
  // R^2 vector infinity-norm
  // \return \f$ \max(|u_0|,|u_1|) \f$
  //
  template<typename T>
  inline
  T
  norm_infinity(Vector<T, 2> const & u)
  {
    return std::max(std::abs(u(0)),std::abs(u(1)));
  }

  //
  // R^N default constructor that initializes to NaNs
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>::Tensor()
  {
    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; j++) {
        e[i][j] = std::numeric_limits<T>::quiet_NaN();
      }
    }

    return;
  }

  //
  // R^3 default constructor that initializes to NaNs
  //
  template<typename T>
  inline
  Tensor<T, 3>::Tensor()
  {
    e[0][0] = std::numeric_limits<T>::quiet_NaN();
    e[0][1] = std::numeric_limits<T>::quiet_NaN();
    e[0][2] = std::numeric_limits<T>::quiet_NaN();

    e[1][0] = std::numeric_limits<T>::quiet_NaN();
    e[1][1] = std::numeric_limits<T>::quiet_NaN();
    e[1][2] = std::numeric_limits<T>::quiet_NaN();

    e[2][0] = std::numeric_limits<T>::quiet_NaN();
    e[2][1] = std::numeric_limits<T>::quiet_NaN();
    e[2][2] = std::numeric_limits<T>::quiet_NaN();

    return;
  }

  //
  // R^2 default constructor that initializes to NaNs
  //
  template<typename T>
  inline
  Tensor<T, 2>::Tensor()
  {
    e[0][0] = std::numeric_limits<T>::quiet_NaN();
    e[0][1] = std::numeric_limits<T>::quiet_NaN();

    e[1][0] = std::numeric_limits<T>::quiet_NaN();
    e[1][1] = std::numeric_limits<T>::quiet_NaN();

    return;
  }

  //
  // R^N create tensor from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>::Tensor(T const & s)
  {
    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; j++) {
        e[i][j] = s;
      }
    }

    return;
  }

  //
  // R^3 create tensor from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T>
  inline
  Tensor<T, 3>::Tensor(T const & s)
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
  // R^2 create tensor from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T>
  inline
  Tensor<T, 2>::Tensor(T const & s)
  {
    e[0][0] = s;
    e[0][1] = s;

    e[1][0] = s;
    e[1][1] = s;

    return;
  }

  //
  // R^N create tensor specifying components
  // The parameters are the components in the canonical basis
  // \param s00 ...
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>::Tensor(T const & s00, ...)
  {
    e.resize(N);
    e[0].resize(N);

    e[0][0] = s00;

    va_list arg_list;
    va_start(arg_list, s00);
    for (Index j = 1; j < N; ++j) {
      T const & s = va_arg(arg_list, T);
      e[0][j] = s;
    }
    for (Index i = 1; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        T const & s = va_arg(arg_list, T);
        e[i][j] = s;
      }
    }
    va_end(arg_list);

    return;
  }

  //
  // R^3 create tensor specifying components
  // The parameters are the components in the canonical basis
  // \param s00 ...
  //
  template<typename T>
  inline
  Tensor<T, 3>::Tensor(
      T const & s00, T const & s01, T const & s02,
      T const & s10, T const & s11, T const & s12,
      T const & s20, T const & s21, T const & s22)
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
  // R^2 create tensor specifying components
  // The parameters are the components in the canonical basis
  // \param s00 ...
  //
  template<typename T>
  inline
  Tensor<T, 2>::Tensor(
      T const & s00, T const & s01,
      T const & s10, T const & s11)
  {
    e[0][0] = s00;
    e[0][1] = s01;

    e[1][0] = s10;
    e[1][1] = s11;

    return;
  }

  //
  // R^N create tensor from array - const version
  // \param data_ptr
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>::Tensor(T const * data_ptr)
  {
    assert(data_ptr != NULL);

    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j] = data_ptr[i * N + j];
      }
    }

    return;
  }

  //
  // R^3 create tensor from array - const version
  // \param data_ptr
  //
  template<typename T>
  inline
  Tensor<T, 3>::Tensor(T const * data_ptr)
  {
    assert(data_ptr != NULL);

    e[0][0] = data_ptr[0];
    e[0][1] = data_ptr[1];
    e[0][2] = data_ptr[2];

    e[1][0] = data_ptr[3];
    e[1][1] = data_ptr[4];
    e[1][2] = data_ptr[5];

    e[2][0] = data_ptr[6];
    e[2][1] = data_ptr[7];
    e[2][2] = data_ptr[8];

    return;
  }

  //
  // R^2 create tensor from array - const version
  // \param data_ptr
  //
  template<typename T>
  inline
  Tensor<T, 2>::Tensor(T const * data_ptr)
  {
    assert(data_ptr != NULL);

    e[0][0] = data_ptr[0];
    e[0][1] = data_ptr[1];

    e[1][0] = data_ptr[2];
    e[1][1] = data_ptr[3];

    return;
  }

  //
  // R^N create vector from array
  // \param data_ptr
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>::Tensor(T * data_ptr)
  {
    assert(data_ptr != NULL);

    e.resize(N);

    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j] = data_ptr[i * N + j];
      }
    }

    return;
  }


  //
  // R^3 create vector from array
  // \param data_ptr
  //
  template<typename T>
  inline
  Tensor<T, 3>::Tensor(T * data_ptr)
  {
    assert(data_ptr != NULL);

    e[0][0] = data_ptr[0];
    e[0][1] = data_ptr[1];
    e[0][2] = data_ptr[2];

    e[1][0] = data_ptr[3];
    e[1][1] = data_ptr[4];
    e[1][2] = data_ptr[5];

    e[2][0] = data_ptr[6];
    e[2][1] = data_ptr[7];
    e[2][2] = data_ptr[8];

    return;
  }

  //
  // R^2 create vector from array
  // \param data_ptr
  //
  template<typename T>
  inline
  Tensor<T, 2>::Tensor(T * data_ptr)
  {
    assert(data_ptr != NULL);

    e[0][0] = data_ptr[0];
    e[0][1] = data_ptr[1];

    e[1][0] = data_ptr[2];
    e[1][1] = data_ptr[3];

    return;
  }

  //
  // R^N copy constructor
  // \param A the values of its componets are copied to the new tensor
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>::Tensor(Tensor<T, N> const & A)
  {
    e.resize(N);

    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j] = A.e[i][j];
      }
    }

    return;
  }

  //
  // R^3 copy constructor
  // \param A the values of its componets are copied to the new tensor
  //
  template<typename T>
  inline
  Tensor<T, 3>::Tensor(Tensor<T, 3> const & A)
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
  // R^2 copy constructor
  // \param A the values of its componets are copied to the new tensor
  //
  template<typename T>
  inline
  Tensor<T, 2>::Tensor(Tensor<T, 2> const & A)
  {
    e[0][0] = A.e[0][0];
    e[0][1] = A.e[0][1];

    e[1][0] = A.e[1][0];
    e[1][1] = A.e[1][1];

    return;
  }

  //
  // R^N simple destructor
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>::~Tensor()
  {
    return;
  }

  //
  // R^3 simple destructor
  //
  template<typename T>
  inline
  Tensor<T, 3>::~Tensor()
  {
    return;
  }

  //
  // R^2 simple destructor
  //
  template<typename T>
  inline
  Tensor<T, 2>::~Tensor()
  {
    return;
  }

  //
  // R^N indexing for constant tensor
  // \param i index
  // \param j index
  //
  template<typename T, Index N>
  inline
  T const &
  Tensor<T, N>::operator()(const Index i, const Index j) const
  {
    assert(i < N);
    assert(j < N);
    return e[i][j];
  }

  //
  // R^3 indexing for constant tensor
  // \param i index
  // \param j index
  //
  template<typename T>
  inline
  T const &
  Tensor<T, 3>::operator()(const Index i, const Index j) const
  {
    assert(i < 3);
    assert(j < 3);
    return e[i][j];
  }

  //
  // R^2 indexing for constant tensor
  // \param i index
  // \param j index
  //
  template<typename T>
  inline
  T const &
  Tensor<T, 2>::operator()(const Index i, const Index j) const
  {
    assert(i < 2);
    assert(j < 2);
    return e[i][j];
  }

  //
  // R^N tensor indexing
  // \param i index
  // \param j index
  //
  template<typename T, Index N>
  inline
  T &
  Tensor<T, N>::operator()(const Index i, const Index j)
  {
    assert(i < N);
    assert(j < N);
    return e[i][j];
  }

  //
  // R^3 tensor indexing
  // \param i index
  // \param j index
  //
  template<typename T>
  inline
  T &
  Tensor<T, 3>::operator()(const Index i, const Index j)
  {
    assert(i < 3);
    assert(j < 3);
    return e[i][j];
  }

  //
  // R^2 tensor indexing
  // \param i index
  // \param j index
  //
  template<typename T>
  inline
  T &
  Tensor<T, 2>::operator()(const Index i, const Index j)
  {
    assert(i < 2);
    assert(j < 2);
    return e[i][j];
  }

  //
  // R^N copy assignment
  // \param A the values of its componets are copied to this tensor
  //
  template<typename T, Index N>
  inline
  Tensor<T, N> &
  Tensor<T, N>::operator=(Tensor<T, N> const & A)
  {
    if (this != &A) {

      e.resize(N);
      for (Index i = 0; i < N; ++i) {
        e[i].resize(N);
        for (Index j = 0; j < N; ++j) {
          e[i][j] = A.e[i][j];
        }
      }

    }

    return *this;
  }

  //
  // R^3 copy assignment
  // \param A the values of its componets are copied to this tensor
  //
  template<typename T>
  inline
  Tensor<T, 3> &
  Tensor<T, 3>::operator=(Tensor<T, 3> const & A)
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
  // R^2 copy assignment
  // \param A the values of its componets are copied to this tensor
  //
  template<typename T>
  inline
  Tensor<T, 2> &
  Tensor<T, 2>::operator=(Tensor<T, 2> const & A)
  {
    if (this != &A) {
      e[0][0] = A.e[0][0];
      e[0][1] = A.e[0][1];

      e[1][0] = A.e[1][0];
      e[1][1] = A.e[1][1];
    }
    return *this;
  }

  //
  // R^N tensor increment
  // \param A added to current tensor
  //
  template<typename T, Index N>
  inline
  Tensor<T, N> &
  Tensor<T, N>::operator+=(Tensor<T, N> const & A)
  {
    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        e[i][j] += A.e[i][j];
      }
    }

    return *this;
  }

  //
  // R^3 tensor increment
  // \param A added to current tensor
  //
  template<typename T>
  inline
  Tensor<T, 3> &
  Tensor<T, 3>::operator+=(Tensor<T, 3> const & A)
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
  // R^2 tensor increment
  // \param A added to current tensor
  //
  template<typename T>
  inline
  Tensor<T, 2> &
  Tensor<T, 2>::operator+=(Tensor<T, 2> const & A)
  {
    e[0][0] += A.e[0][0];
    e[0][1] += A.e[0][1];

    e[1][0] += A.e[1][0];
    e[1][1] += A.e[1][1];

    return *this;
  }

  //
  // R^N tensor decrement
  // \param A substracted from current tensor
  //
  template<typename T, Index N>
  inline
  Tensor<T, N> &
  Tensor<T, N>::operator-=(Tensor<T, N> const & A)
  {
    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        e[i][j] -= A.e[i][j];
      }
    }

    return *this;
  }

  //
  // R^3 tensor decrement
  // \param A substracted from current tensor
  //
  template<typename T>
  inline
  Tensor<T, 3> &
  Tensor<T, 3>::operator-=(Tensor<T, 3> const & A)
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
  // R^2 tensor decrement
  // \param A substracted from current tensor
  //
  template<typename T>
  inline
  Tensor<T, 2> &
  Tensor<T, 2>::operator-=(Tensor<T, 2> const & A)
  {
    e[0][0] -= A.e[0][0];
    e[0][1] -= A.e[0][1];

    e[1][0] -= A.e[1][0];
    e[1][1] -= A.e[1][1];

    return *this;
  }

  //
  // R^N fill with zeros
  //
  template<typename T, Index N>
  inline
  void
  Tensor<T, N>::clear()
  {
    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j] = 0.0;
      }
    }

    return;
  }

  //
  // R^3 fill with zeros
  //
  template<typename T>
  inline
  void
  Tensor<T, 3>::clear()
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
  // R^2 fill with zeros
  //
  template<typename T>
  inline
  void
  Tensor<T, 2>::clear()
  {
    e[0][0] = 0.0;
    e[0][1] = 0.0;

    e[1][0] = 0.0;
    e[1][1] = 0.0;

    return;
  }

  //
  // R^N tensor addition
  // \return \f$ A + B \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  operator+(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    Tensor<T, N> S;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        S(i, j) = A(i, j) + B(i, j);
      }
    }

    return S;
  }

  //
  // R^3 tensor addition
  // \return \f$ A + B \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  operator+(Tensor<T, 3> const & A, Tensor<T, 3> const & B)
  {
    Tensor<T, 3> S;

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
  // R^2 tensor addition
  // \return \f$ A + B \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  operator+(Tensor<T, 2> const & A, Tensor<T, 2> const & B)
  {
    Tensor<T, 2> S;

    S(0,0) = A(0,0) + B(0,0);
    S(0,1) = A(0,1) + B(0,1);

    S(1,0) = A(1,0) + B(1,0);
    S(1,1) = A(1,1) + B(1,1);

    return S;
  }

  //
  // R^N Tensor substraction
  // \return \f$ A - B \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  operator-(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    Tensor<T, N> S;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        S(i, j) = A(i, j) - B(i, j);
      }
    }

    return S;
  }


  //
  // R^3 tensor substraction
  // \return \f$ A - B \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  operator-(Tensor<T, 3> const & A, Tensor<T, 3> const & B)
  {
    Tensor<T, 3> S;

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
  // R^2 tensor substraction
  // \return \f$ A - B \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  operator-(Tensor<T, 2> const & A, Tensor<T, 2> const & B)
  {
    Tensor<T, 2> S;

    S(0,0) = A(0,0) - B(0,0);
    S(0,1) = A(0,1) - B(0,1);

    S(1,0) = A(1,0) - B(1,0);
    S(1,1) = A(1,1) - B(1,1);

    return S;
  }

  //
  // R^N tensor minus
  // \return \f$ -A \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  operator-(Tensor<T, N> const & A)
  {
    Tensor<T, N> S;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        S(i, j) = - A(i, j);
      }
    }

    return S;
  }

  //
  // R^3 tensor minus
  // \return \f$ -A \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  operator-(Tensor<T, 3> const & A)
  {
    return Tensor<T, 3>(
        -A(0,0),-A(0,1),-A(0,2),
        -A(1,0),-A(1,1),-A(1,2),
        -A(2,0),-A(2,1),-A(2,2));
  }

  //
  // R^2 tensor minus
  // \return \f$ -A \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  operator-(Tensor<T, 2> const & A)
  {
    return Tensor<T, 2>(
        -A(0,0),-A(0,1),
        -A(1,0),-A(1,1));
  }

  //
  // R^N tensor equality
  // Tested by components
  // \return \f$ A \equiv B \f$
  //
  template<typename T, Index N>
  inline
  bool
  operator==(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    bool equal = true;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        equal = equal && (A(i, j) == B(i, j));
        if (equal == false) break;
      }
      if (equal == false) break;
    }

    return equal;
  }

  //
  // R^3 tensor equality
  // Tested by components
  // \return \f$ A \equiv B \f$
  //
  template<typename T>
  inline
  bool
  operator==(Tensor<T, 3> const & A, Tensor<T, 3> const & B)
  {
    return
      A(0,0)==B(0,0) && A(0,1)==B(0,1) && A(0,2)==B(0,2) &&
      A(1,0)==B(1,0) && A(1,1)==B(1,1) && A(1,2)==B(1,2) &&
      A(2,0)==B(2,0) && A(2,1)==B(2,1) && A(2,2)==B(2,2);
  }

  //
  // R^2 tensor equality
  // Tested by components
  // \return \f$ A \equiv B \f$
  //
  template<typename T>
  inline
  bool
  operator==(Tensor<T, 2> const & A, Tensor<T, 2> const & B)
  {
    return
      A(0,0)==B(0,0) && A(0,1)==B(0,1) &&
      A(1,0)==B(1,0) && A(1,1)==B(1,1);
  }

  //
  // R^N tensor inequality
  // Tested by components
  // \return \f$ A \neq B \f$
  //
  template<typename T, Index N>
  inline
  bool
  operator!=(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    return !(A == B);
  }

  //
  // R^3 tensor inequality
  // Tested by components
  // \return \f$ A \neq B \f$
  //
  template<typename T>
  inline
  bool
  operator!=(Tensor<T, 3> const & A, Tensor<T, 3> const & B)
  {
    return !(A == B);
  }

  //
  // R^2 tensor inequality
  // Tested by components
  // \return \f$ A \neq B \f$
  //
  template<typename T>
  inline
  bool
  operator!=(Tensor<T, 2> const & A, Tensor<T, 2> const & B)
  {
    return !(A == B);
  }

  //
  // R^N scalar tensor product
  // \param s scalar
  // \param A tensor
  // \return \f$ s A \f$
  //
  template<typename T, Index N, typename S>
  inline
  Tensor<T, N>
  operator*(S const & s, Tensor<T, N> const & A)
  {
    Tensor<T, N> B;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        B(i, j) = s * A(i, j);
      }
    }

    return B;
  }

  //
  // R^3 scalar tensor product
  // \param s scalar
  // \param A tensor
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  inline
  Tensor<T, 3>
  operator*(S const & s, Tensor<T, 3> const & A)
  {
    return Tensor<T, 3>(
        s*A(0,0), s*A(0,1), s*A(0,2),
        s*A(1,0), s*A(1,1), s*A(1,2),
        s*A(2,0), s*A(2,1), s*A(2,2));
  }

  //
  // R^2 scalar tensor product
  // \param s scalar
  // \param A tensor
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  inline
  Tensor<T, 2>
  operator*(S const & s, Tensor<T, 2> const & A)
  {
    return Tensor<T, 2>(
        s*A(0,0), s*A(0,1),
        s*A(1,0), s*A(1,1));
  }

  //
  // R^N tensor scalar product
  // \param A tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename T, Index N, typename S>
  inline
  Tensor<T, N>
  operator*(Tensor<T, N> const & A, S const & s)
  {
    return s * A;
  }

  //
  // R^3 tensor scalar product
  // \param A tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  inline
  Tensor<T, 3>
  operator*(Tensor<T, 3> const & A, S const & s)
  {
    return s * A;
  }

  //
  // R^2 tensor scalar product
  // \param A tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  inline
  Tensor<T, 2>
  operator*(Tensor<T, 2> const & A, S const & s)
  {
    return s * A;
  }

  //
  // R^N tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  operator*(Tensor<T, N> const & A, Vector<T, N> const & u)
  {
    return dot(A, u);
  }

  //
  // R^3 tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  operator*(Tensor<T, 3> const & A, Vector<T, 3> const & u)
  {
    return dot(A, u);
  }

  //
  // R^2 tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  inline
  Vector<T, 2>
  operator*(Tensor<T, 2> const & A, Vector<T, 2> const & u)
  {
    return dot(A, u);
  }

  //
  // R^N vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  operator*(Vector<T, N> const & u, Tensor<T, N> const & A)
  {
    return dot(u, A);
  }

  //
  // R^3 vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  operator*(Vector<T, 3> const & u, Tensor<T, 3> const & A)
  {
    return dot(u, A);
  }

  //
  // R^2 vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T>
  inline
  Vector<T, 2>
  operator*(Vector<T, 2> const & u, Tensor<T, 2> const & A)
  {
    return dot(u, A);
  }

  //
  // R^N tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  dot(Tensor<T, N> const & A, Vector<T, N> const & u)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      T s = 0.0;
      for (Index j = 0; j < N; ++j) {
        s += A(i, j) * u(j);
      }
      v(i) = s;
    }

    return v;
  }

  //
  // R^3 tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  dot(Tensor<T, 3> const & A, Vector<T, 3> const & u)
  {
    Vector<T, 3> v;

    v(0) = A(0,0)*u(0) + A(0,1)*u(1) + A(0,2)*u(2);
    v(1) = A(1,0)*u(0) + A(1,1)*u(1) + A(1,2)*u(2);
    v(2) = A(2,0)*u(0) + A(2,1)*u(1) + A(2,2)*u(2);

    return v;
  }

  //
  // R^2 tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  inline
  Vector<T, 2>
  dot(Tensor<T, 2> const & A, Vector<T, 2> const & u)
  {
    Vector<T, 2> v;

    v(0) = A(0,0)*u(0) + A(0,1)*u(1);
    v(1) = A(1,0)*u(0) + A(1,1)*u(1);

    return v;
  }

  //
  // R^N vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T, Index N>
  inline
  Vector<T, N>
  dot(Vector<T, N> const & u, Tensor<T, N> const & A)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      T s = 0.0;
      for (Index j = 0; j < N; ++j) {
        s += A(j, i) * u(j);
      }
      v(i) = s;
    }

    return v;
  }

  //
  // R^3 vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T>
  inline
  Vector<T, 3>
  dot(Vector<T, 3> const & u, Tensor<T, 3> const & A)
  {
    Vector<T, 3> v;

    v(0) = A(0,0)*u(0) + A(1,0)*u(1) + A(2,0)*u(2);
    v(1) = A(0,1)*u(0) + A(1,1)*u(1) + A(2,1)*u(2);
    v(2) = A(0,2)*u(0) + A(1,2)*u(1) + A(2,2)*u(2);

    return v;
  }

  //
  // R^2 vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T>
  inline
  Vector<T, 2>
  dot(Vector<T, 2> const & u, Tensor<T, 2> const & A)
  {
    Vector<T, 2> v;

    v(0) = A(0,0)*u(0) + A(1,0)*u(1);
    v(1) = A(0,1)*u(0) + A(1,1)*u(1);

    return v;
  }

  //
  // R^N tensor dot product C = A B
  // \return \f$ A \cdot B \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  operator*(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    return dot(A, B);
  }

  //
  // R^3 tensor dot product C = A B
  // \return \f$ A \cdot B \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  operator*(Tensor<T, 3> const & A, Tensor<T, 3> const & B)
  {
    return dot(A, B);
  }

  //
  // R^2 tensor dot product C = A B
  // \return \f$ A \cdot B \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  operator*(Tensor<T, 2> const & A, Tensor<T, 2> const & B)
  {
    return dot(A, B);
  }

  //
  // R^N tensor tensor product C = A B
  // \param A tensor
  // \param B tensor
  // \return a tensor \f$ A \cdot B \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  dot(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    Tensor<T, N> C;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        T s = 0.0;
        for (Index k = 0; k < N; ++k) {
          s += A(i, k) * B(k, j);
        }
        C(i, j) = s;
      }
    }

    return C;
  }

  //
  // R^3 tensor tensor product C = A B
  // \param A tensor
  // \param B tensor
  // \return a tensor \f$ A \cdot B \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  dot(Tensor<T, 3> const & A, Tensor<T, 3> const & B)
  {
    Tensor<T, 3> C;

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
  // R^2 tensor tensor product C = A B
  // \param A tensor
  // \param B tensor
  // \return a tensor \f$ A \cdot B \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  dot(Tensor<T, 2> const & A, Tensor<T, 2> const & B)
  {
    Tensor<T, 2> C;

    C(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0);
    C(0,1) = A(0,0)*B(0,1) + A(0,1)*B(1,1);

    C(1,0) = A(1,0)*B(0,0) + A(1,1)*B(1,0);
    C(1,1) = A(1,0)*B(0,1) + A(1,1)*B(1,1);

    return C;
  }

  //
  // R^N tensor tensor double dot product (contraction)
  // \param A tensor
  // \param B tensor
  // \return a scalar \f$ A : B \f$
  //
  template<typename T, Index N>
  inline
  T
  dotdot(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    T s = 0.0;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        s += A(i, j) * B(i, j);
      }
    }

    return s;
  }

  //
  // R^3 tensor tensor double dot product (contraction)
  // \param A tensor
  // \param B tensor
  // \return a scalar \f$ A : B \f$
  //
  template<typename T>
  inline
  T
  dotdot(Tensor<T, 3> const & A, Tensor<T, 3> const & B)
  {
    T s = 0.0;

    s+= A(0,0)*B(0,0) + A(0,1)*B(0,1) + A(0,2)*B(0,2);
    s+= A(1,0)*B(1,0) + A(1,1)*B(1,1) + A(1,2)*B(1,2);
    s+= A(2,0)*B(2,0) + A(2,1)*B(2,1) + A(2,2)*B(2,2);

    return s;
  }

  //
  // R^2 tensor tensor double dot product (contraction)
  // \param A tensor
  // \param B tensor
  // \return a scalar \f$ A : B \f$
  //
  template<typename T>
  inline
  T
  dotdot(Tensor<T, 2> const & A, Tensor<T, 2> const & B)
  {
    T s = 0.0;

    s+= A(0,0)*B(0,0) + A(0,1)*B(0,1);
    s+= A(1,0)*B(1,0) + A(1,1)*B(1,1);

    return s;
  }

  //
  // R^N tensor Frobenius norm
  // \return \f$ \sqrt{A:A} \f$
  //
  template<typename T, Index N>
  inline
  T
  norm(Tensor<T, N> const & A)
  {
    return sqrt(dotdot(A, A));
  }

  //
  // R^3 tensor Frobenius norm
  // \return \f$ \sqrt{A:A} \f$
  //
  template<typename T>
  inline
  T
  norm(Tensor<T, 3> const & A)
  {
    T s = 0.0;

    s+= A(0,0)*A(0,0) + A(0,1)*A(0,1) + A(0,2)*A(0,2);
    s+= A(1,0)*A(1,0) + A(1,1)*A(1,1) + A(1,2)*A(1,2);
    s+= A(2,0)*A(2,0) + A(2,1)*A(2,1) + A(2,2)*A(2,2);

    return sqrt(s);
  }

  //
  // R^2 tensor Frobenius norm
  // \return \f$ \sqrt{A:A} \f$
  //
  template<typename T>
  inline
  T
  norm(Tensor<T, 2> const & A)
  {
    T s = 0.0;

    s+= A(0,0)*A(0,0) + A(0,1)*A(0,1);
    s+= A(1,0)*A(1,0) + A(1,1)*A(1,1);

    return sqrt(s);
  }

  //
  // R^N tensor 1-norm
  // \return \f$ \max_{j \in {0,1,2}}\Sigma_{i=0}^2 |A_{ij}| \f$
  //
  template<typename T, Index N>
  inline
  T
  norm_1(Tensor<T, N> const & A)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      T s = 0.0;
      for (Index j = 0; j < N; ++j) {
        s += std::abs(A(j, i));
      }
      v(i) = s;
    }

    T s = v(0);

    for (Index i = 1; i < N; ++i) {
      s = std::max(s, v(i));
    }

    return s;
  }

  //
  // R^3 tensor 1-norm
  // \return \f$ \max_{j \in {0,1,2}}\Sigma_{i=0}^2 |A_{ij}| \f$
  //
  template<typename T>
  inline
  T
  norm_1(Tensor<T, 3> const & A)
  {
    const T s0 = std::abs(A(0,0)) + std::abs(A(1,0)) + std::abs(A(2,0));
    const T s1 = std::abs(A(0,1)) + std::abs(A(1,1)) + std::abs(A(2,1));
    const T s2 = std::abs(A(0,2)) + std::abs(A(1,2)) + std::abs(A(2,2));

    return std::max(std::max(s0,s1),s2);
  }

  //
  // R^2 tensor 1-norm
  // \return \f$ \max_{j \in {0,1,2}}\Sigma_{i=0}^2 |A_{ij}| \f$
  //
  template<typename T>
  inline
  T
  norm_1(Tensor<T, 2> const & A)
  {
    const T s0 = std::abs(A(0,0)) + std::abs(A(1,0));
    const T s1 = std::abs(A(0,1)) + std::abs(A(1,1));

    return std::max(s0,s1);
  }

  //
  // R^N tensor infinity-norm
  // \return \f$ \max_{i \in {0,1,2}}\Sigma_{j=0}^2 |A_{ij}| \f$
  //
  template<typename T, Index N>
  inline
  T
  norm_infinity(Tensor<T, N> const & A)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      T s = 0.0;
      for (Index j = 0; j < N; ++j) {
        s += std::abs(A(i, j));
      }
      v(i) = s;
    }

    T s = v(0);

    for (Index i = 1; i < N; ++i) {
      s = std::max(s, v(i));
    }

    return s;
  }

  //
  // R^3 tensor infinity-norm
  // \return \f$ \max_{i \in {0,1,2}}\Sigma_{j=0}^2 |A_{ij}| \f$
  //
  template<typename T>
  inline
  T
  norm_infinity(Tensor<T, 3> const & A)
  {
    const T s0 = std::abs(A(0,0)) + std::abs(A(0,1)) + std::abs(A(0,2));
    const T s1 = std::abs(A(1,0)) + std::abs(A(1,1)) + std::abs(A(1,2));
    const T s2 = std::abs(A(2,0)) + std::abs(A(2,1)) + std::abs(A(2,2));

    return std::max(std::max(s0,s1),s2);
  }

  //
  // R^2 tensor infinity-norm
  // \return \f$ \max_{i \in {0,1,2}}\Sigma_{j=0}^2 |A_{ij}| \f$
  //
  template<typename T>
  inline
  T
  norm_infinity(Tensor<T, 2> const & A)
  {
    const T s0 = std::abs(A(0,0)) + std::abs(A(0,1));
    const T s1 = std::abs(A(1,0)) + std::abs(A(1,1));

    return std::max(s0,s1);
  }

  //
  // R^N dyad
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  dyad(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    Tensor<T, N> A;

    for (Index i = 0; i < N; ++i) {
      const T s = u(i);
      for (Index j = 0; j < N; ++j) {
        A(i, j) = s * v(j);
      }
    }

    return A;
  }

  //
  // R^3 dyad
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  dyad(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    Tensor<T, 3> A;

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
  // R^2 dyad
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  dyad(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    Tensor<T, 2> A;

    A(0,0) = u(0) * v(0);
    A(0,1) = u(0) * v(1);

    A(1,0) = u(1) * v(0);
    A(1,1) = u(1) * v(1);

    return A;
  }

  //
  // R^N bun operator, just for Jay
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  bun(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^3 bun operator, just for Jay
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  bun(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^2 bun operator, just for Jay
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  bun(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^N tensor product
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  tensor(Vector<T, N> const & u, Vector<T, N> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^3 tensor product
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  tensor(Vector<T, 3> const & u, Vector<T, 3> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^2 tensor product
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  tensor(Vector<T, 2> const & u, Vector<T, 2> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^N diagonal tensor from vector
  // \param v vector
  // \return A = diag(v)
  //
  template<typename T, Index N>
  Tensor<T, N>
  diag(Vector<T, N> const & v)
  {
    Tensor<T, N> A = zero<T, N>();

    for (Index i = 0; i < N; ++i) {
      A(i, i) = v(i);
    }

    return A;
  }

  //
  // R^3 diagonal tensor from vector
  // \param v vector
  // \return A = diag(v)
  //
  template<typename T>
  Tensor<T, 3>
  diag(Vector<T, 3> const & v)
  {
    return Tensor<T, 3>(v(0), 0.0, 0.0, 0.0, v(1), 0.0, 0.0, 0.0, v(2));
  }

  //
  // R^2 diagonal tensor from vector
  // \param v vector
  // \return A = diag(v)
  //
  template<typename T>
  Tensor<T, 2>
  diag(Vector<T, 2> const & v)
  {
    return Tensor<T, 2>(v(0), 0.0, 0.0, v(1));
  }

  //
  // R^N diagonal of tensor in a vector
  // \param A tensor
  // \return v = diag(A)
  //
  template<typename T, Index N>
  Vector<T, N>
  diag(Tensor<T, N> const & A)
  {
    Vector<T, N> v;

    for (Index i = 0; i < N; ++i) {
      v(i) = A(i, i);
    }

    return v;
  }

  //
  // R^3 diagonal of tensor in a vector
  // \param A tensor
  // \return v = diag(A)
  //
  template<typename T>
  Vector<T, 3>
  diag(Tensor<T, 3> const & A)
  {
    return Vector<T, 3>(A(0,0), A(1,1), A(2,2));
  }

  //
  // R^2 diagonal of tensor in a vector
  // \param A tensor
  // \return v = diag(A)
  //
  template<typename T>
  Vector<T, 2>
  diag(Tensor<T, 2> const & A)
  {
    return Vector<T, 2>(A(0,0), A(1,1));
  }

  //
  // R^N zero 2nd-order tensor
  // All components are zero
  //
  template<typename T, Index N>
  inline
  const Tensor<T, N>
  zero()
  {
    Tensor<T, N> A;
    A.clear();
    return A;
  }

  //
  // R^3 zero 2nd-order tensor
  // All components are zero
  //
  template<typename T>
  inline
  const Tensor<T, 3>
  zero()
  {
    return Tensor<T, 3>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
  }

  //
  // R^2 zero 2nd-order tensor
  // All components are zero
  //
  template<typename T>
  inline
  const Tensor<T, 2>
  zero()
  {
    return Tensor<T, 2>(0.0,0.0,0.0,0.0);
  }

  //
  // R^N 2nd-order identity tensor
  //
  template<typename T, Index N>
  inline
  const Tensor<T, N>
  identity()
  {
    Tensor<T, N> A;
    A.clear();
    for (Index i = 0; i < N; ++i) {
      A(i, i) = 1.0;
    }
    return A;
  }

  //
  // R^3 2nd-order identity tensor
  //
  template<typename T>
  inline
  const Tensor<T, 3>
  identity()
  {
    return Tensor<T, 3>(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
  }

  //
  // R^2 2nd-order identity tensor
  //
  template<typename T>
  inline
  const Tensor<T, 2>
  identity()
  {
    return Tensor<T, 2>(1.0,0.0,0,0,1.0);
  }

  //
  // R^N 2nd-order identity tensor, à la Matlab
  //
  template<typename T, Index N>
  inline
  const Tensor<T, N>
  eye()
  {
    return identity<T, N>();
  }

  //
  // R^3 2nd-order identity tensor, à la Matlab
  //
  template<typename T>
  inline
  const Tensor<T, 3>
  eye()
  {
    return identity<T, 3>();
  }

  //
  // R^2 2nd-order identity tensor, à la Matlab
  //
  template<typename T>
  inline
  const Tensor<T, 2>
  eye()
  {
    return identity<T, 2>();
  }

  //
  // R^N 2nd-order tensor transpose
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  transpose(Tensor<T, N> const & A)
  {
    Tensor<T, N> B;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        B(i, j) = A(j, i);
      }
    }

    return B;
  }

  //
  // R^3 2nd-order tensor transpose
  //
  template<typename T>
  inline
  Tensor<T, 3>
  transpose(Tensor<T, 3> const & A)
  {
    return Tensor<T, 3>(
        A(0,0),A(1,0),A(2,0),
        A(0,1),A(1,1),A(2,1),
        A(0,2),A(1,2),A(2,2));
  }

  //
  // R^2 2nd-order tensor transpose
  //
  template<typename T>
  inline
  Tensor<T, 2>
  transpose(Tensor<T, 2> const & A)
  {
    return Tensor<T, 2>(
        A(0,0),A(1,0),
        A(0,1),A(1,1));
  }

  //
  // R^N symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A + A^T) \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  symm(Tensor<T, N> const & A)
  {
    return 0.5 * (A + transpose(A));
  }

  //
  // R^3 symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A + A^T) \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  symm(Tensor<T, 3> const & A)
  {
    T const & s00 = A(0,0);
    T const & s11 = A(1,1);
    T const & s22 = A(2,2);

    T const & s01 = 0.5*(A(0,1)+A(1,0));
    T const & s02 = 0.5*(A(0,2)+A(2,0));
    T const & s12 = 0.5*(A(1,2)+A(2,1));

    return Tensor<T, 3>(
        s00, s01, s02,
        s01, s11, s12,
        s02, s12, s22);
  }

  //
  // R^2 symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A + A^T) \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  symm(Tensor<T, 2> const & A)
  {
    T const & s00 = A(0,0);
    T const & s11 = A(1,1);

    T const & s01 = 0.5*(A(0,1)+A(1,0));

    return Tensor<T, 2>(
        s00, s01,
        s01, s11);
  }

  //
  // R^N skew symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A - A^T) \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  skew(Tensor<T, N> const & A)
  {
    return 0.5 * (A - transpose(A));
  }

  //
  // R^3 skew symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A - A^T) \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  skew(Tensor<T, 3> const & A)
  {
    T const & s01 = 0.5*(A(0,1)-A(1,0));
    T const & s02 = 0.5*(A(0,2)-A(2,0));
    T const & s12 = 0.5*(A(1,2)-A(2,1));

    return Tensor<T, 3>(
         0.0,  s01,  s02,
        -s01,  0.0,  s12,
        -s02, -s12,  0.0);
  }

  //
  // R^2 skew symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A - A^T) \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  skew(Tensor<T, 2> const & A)
  {
    T const & s01 = 0.5*(A(0,1)-A(1,0));

    return Tensor<T, 2>(
         0.0,  s01,
        -s01,  0.0);
  }

  //
  // R^N skew symmetric 2nd-order tensor from vector, undefined
  // \param u vector
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  skew(Vector<T, N> const & u)
  {
    Tensor<T, N> A;

    std::cerr << "ERROR: Skew from vector undefined for R^" << N << std::endl;
    exit(1);

    return A;
  }

  //
  // R^3 skew symmetric 2nd-order tensor from vector
  // \param u vector
  // \return \f$ {{0, -u_2, u_1}, {u_2, 0, -u_0}, {-u_1, u+0, 0}} \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  skew(Vector<T, 3> const & u)
  {
    return Tensor<T, 3>(
         0.0, -u(2),  u(1),
        u(2),   0.0, -u(0),
       -u(1),  u(0),   0.0);
  }

  //
  // R^2 skew symmetric 2nd-order tensor from vector, undefined
  // \param u vector
  //
  template<typename T>
  inline
  Tensor<T, 2>
  skew(Vector<T, 2> const & u)
  {
    Tensor<T, 2> A;

    std::cerr << "ERROR: Skew from vector undefined for R^2" << std::endl;
    exit(1);

    return A;
  }

  //
  // R^N volumetric part of 2nd-order tensor
  // \return \f$ \frac{1}{N} \mathrm{tr}\:(A) I \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  vol(Tensor<T, N> const & A)
  {
    const T tr = (1.0/T(N)) * trace(A);

    return tr * eye<T, N>();
  }

  //
  // R^3 volumetric part of 2nd-order tensor
  // \return \f$ \frac{1}{3} \mathrm{tr}\:(A) I \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  vol(Tensor<T, 3> const & A)
  {
    const T tr = (1.0/3.0) * trace(A);

    return tr * eye<T, 3>();
  }

  //
  // R^2 volumetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2} \mathrm{tr}\:(A) I \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  vol(Tensor<T, 2> const & A)
  {
    const T tr = 0.5 * trace(A);

    return tr * eye<T, 2>();
  }

  //
  // R^N deviatoric part of 2nd-order tensor
  // \return \f$ A - vol(A) \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  dev(Tensor<T, N> const & A)
  {
    return A - vol(A);
  }

  //
  // R^3 deviatoric part of 2nd-order tensor
  // \return \f$ A - vol(A) \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  dev(Tensor<T, 3> const & A)
  {
    return A - vol(A);
  }

  //
  // R^2 deviatoric part of 2nd-order tensor
  // \return \f$ A - vol(A) \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  dev(Tensor<T, 2> const & A)
  {
    return A - vol(A);
  }

  //
  // R^N 2nd-order tensor inverse
  // Gauss-Jordan elimination. Warning: full pivoting
  // for small tensors. Use Teuchos LAPACK interface for
  // more efficient and robust techniques.
  // \param A nonsingular tensor
  // \return \f$ A^{-1} \f$
  //
  template<typename T, Index N>
  inline
  Tensor<T, N>
  inverse(Tensor<T, N> const & A)
  {
    Tensor<T, N> S = A;
    Tensor<T, N> B = identity<T, N>();
    Vector<Index, N> p, q;

    p.clear();
    q.clear();

    // Determine full pivot
    for (Index k = 0; k < N; ++k) {

      Index m = k;
      Index n = k;

      T s = fabs(S(m, n));

      for (Index i = k; i < N; ++i) {
        for (Index j = k; j < N; ++j) {
          if (fabs(S(i, j)) > s) {
            m = i;
            n = j;
            s = fabs(S(i, j));
          }
        }
      }

      // Swap rows and columns for pivoting
      swap_row(S, k, m);
      swap_row(B, k, m);

      swap_col(S, k, n);
      swap_col(B, k, n);

      p(k) = m;
      q(k) = n;

      // Gauss-Jordan elimination
      const T t = S(k, k);

      if (t == 0.0) {
        std::cerr << "ERROR: Inverse of singular tensor." << std::endl;
        exit(1);
      }

      for (Index j = 0; j < N; ++j) {
        S(k, j) /= t;
        B(k, j) /= t;
      }

      for (Index i = 0; i < N; ++i) {
        if (i == k) continue;

        const T c = S(i, k);

        for (Index j = 0; j < N; ++j) {
          S(i, j) -= c * S(k, j);
          B(i, j) -= c * B(k, j);
        }
      }

    }

    // Restore order of rows and columns
    for (Index k = N - 1; k > 0; --k) {

      Index m = p(k);
      Index n = q(k);

      swap_row(B, k, m);
      swap_col(B, k, n);

    }

    return B;
  }

  //
  // R^3 2nd-order tensor inverse
  // \param A nonsingular tensor
  // \return \f$ A^{-1} \f$
  //
  template<typename T>
  inline
  Tensor<T, 3>
  inverse(Tensor<T, 3> const & A)
  {
    const T d = det(A);
    assert(d != 0.0);
    Tensor<T, 3> B(
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
    return T(1.0 / d) * B;
  }

  //
  // R^2 2nd-order tensor inverse
  // \param A nonsingular tensor
  // \return \f$ A^{-1} \f$
  //
  template<typename T>
  inline
  Tensor<T, 2>
  inverse(Tensor<T, 2> const & A)
  {
    const T d = det(A);
    assert(d != 0.0);
    Tensor<T, 2> B(A(1,1), -A(0,1), -A(1,0), A(0,0));
    return T(1.0 / d) * B;
  }

  //
  // R^N Subtensor
  // \param A tensor
  // \param i index
  // \param j index
  // \return Subtensor with i-row and j-col deleted.
  //
  template<typename T, Index N>
  inline
  Tensor<T, N - 1>
  subtensor(Tensor<T, N> const & A, Index i, Index j)
  {
    assert(i < N);
    assert(j < N);

    Tensor<T, N - 1> B;

    Index p = 0;
    Index q = 0;
    for (Index m = 0; m < N; ++m) {
      if (m == i) continue;
      for (Index n = 0; n < N; ++n) {
        if (n == j) continue;
        B(p, q) = A(m, n);
        ++q;
      }
      ++p;
    }

    return B;
  }

  //
  // R^3 Subtensor
  // \param A tensor
  // \param i index
  // \param j index
  // \return Subtensor with i-row and j-col deleted.
  //
  template<typename T>
  inline
  Tensor<T, 2>
  subtensor(Tensor<T, 3> const & A, Index i, Index j)
  {
    assert(i < 3);
    assert(j < 3);

    Tensor<T, 2> B;

    Index p = 0;
    Index q = 0;
    for (Index m = 0; m < 3; ++m) {
      if (m == i) continue;
      for (Index n = 0; n < 3; ++n) {
        if (n == j) continue;
        B(p, q) = A(m, n);
        ++q;
      }
      ++p;
    }

    return B;
  }

  //
  // R^2 Subtensor
  // \param A tensor
  // \param i index
  // \param j index
  // \return Subtensor with i-row and j-col deleted.
  //
  template<typename T>
  inline
  Tensor<T, 1>
  subtensor(Tensor<T, 2> const & A, Index i, Index j)
  {
    assert(i < 2);
    assert(j < 2);

    Tensor<T, 1> B;

    Index m = 1 - i;
    Index n = 1 - j;

    B(0, 0) = A(m, n);

    return B;

  }

  //
  // Swap row. Echange rows i and j in place
  // \param A tensor
  // \param i index
  // \param j index
  //
  template<typename T, Index N>
  void
  swap_row(Tensor<T, N> & A, Index i, Index j)
  {
    if (i != j) {
      for (Index k = 0; k < N; ++k) {
        std::swap(A(i, k), A(j, k));
      }
    }
    return;
  }

  //
  // Swap column. Echange columns i and j in place
  // \param A tensor
  // \param i index
  // \param j index
  //
  template<typename T, Index N>
  void
  swap_col(Tensor<T, N> & A, Index i, Index j)
  {
    if (i != j) {
      for (Index k = 0; k < N; ++k) {
        std::swap(A(k, i), A(k, j));
      }
    }
    return;
  }

  //
  // R^N determinant
  // Laplace expansion. Warning: no pivoting.
  // Casual use only. Use Teuchos LAPACK interface for
  // more efficient and robust techniques.
  // \param A tensor
  // \return \f$ \det A \f$
  //
  template<typename T, Index N>
  inline
  T
  det(Tensor<T, N> const & A)
  {
    T s = 0.0;

    int sign = 1;
    for (Index i = 0; i < N; ++i) {
      const T d = det(subtensor(A, i, 1));
      s += sign * d * A(i, 1);
      sign *= -1;
    }
    return s;
  }

  //
  // R^3 determinant
  // \param A tensor
  // \return \f$ \det A \f$
  //
  template<typename T>
  inline
  T
  det(Tensor<T, 3> const & A)
  {
    return
        -A(0,2)*A(1,1)*A(2,0) + A(0,1)*A(1,2)*A(2,0) +
         A(0,2)*A(1,0)*A(2,1) - A(0,0)*A(1,2)*A(2,1) -
         A(0,1)*A(1,0)*A(2,2) + A(0,0)*A(1,1)*A(2,2);
  }

  //
  // R^2 determinant
  // \param A tensor
  // \return \f$ \det A \f$
  //
  template<typename T>
  inline
  T
  det(Tensor<T, 2> const & A)
  {
    return A(0,0) * A(1,1) - A(1,0) * A(0,1);
  }

  //
  // R^N trace
  // \param A tensor
  // \return \f$ A:I \f$
  //
  template<typename T, Index N>
  inline
  T
  trace(Tensor<T, N> const & A)
  {
    T s = 0.0;

    for (Index i = 0; i < N; ++i) {
      s += A(i,i);
    }

    return s;
  }

  //
  // R^3 trace
  // \param A tensor
  // \return \f$ A:I \f$
  //
  template<typename T>
  inline
  T
  trace(Tensor<T, 3> const & A)
  {
    return A(0,0) + A(1,1) + A(2,2);
  }

  //
  // R^2 trace
  // \param A tensor
  // \return \f$ A:I \f$
  //
  template<typename T>
  inline
  T
  trace(Tensor<T, 2> const & A)
  {
    return A(0,0) + A(1,1);
  }

  //
  // R^N first invariant, trace
  // \param A tensor
  // \return \f$ I_A = A:I \f$
  //
  template<typename T, Index N>
  inline
  T
  I1(Tensor<T, N> const & A)
  {
    return trace(A);
  }

  //
  // R^3 first invariant, trace
  // \param A tensor
  // \return \f$ I_A = A:I \f$
  //
  template<typename T>
  inline
  T
  I1(Tensor<T, 3> const & A)
  {
    return trace(A);
  }

  //
  // R^2 first invariant, trace
  // \param A tensor
  // \return \f$ I_A = A:I \f$
  //
  template<typename T>
  inline
  T
  I1(Tensor<T, 2> const & A)
  {
    return trace(A);
  }

  //
  // R^N second invariant
  // \param A tensor
  // \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
  //
  template<typename T, Index N>
  inline
  T
  I2(Tensor<T, N> const & A)
  {
    const T trA = trace(A);

    return 0.5 * (trA * trA - trace(A * A));
  }

  //
  // R^3 second invariant
  // \param A tensor
  // \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
  //
  template<typename T>
  inline
  T
  I2(Tensor<T, 3> const & A)
  {
    const T trA = trace(A);

    return 0.5 * (trA*trA - A(0,0)*A(0,0) - A(1,1)*A(1,1) - A(2,2)*A(2,2)) -
        A(0,1)*A(1,0) - A(0,2)*A(2,0) - A(1,2)*A(2,1);
  }

  //
  // R^2 second invariant
  // \param A tensor
  // \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
  //
  template<typename T>
  inline
  T
  I2(Tensor<T, 2> const & A)
  {
    const T trA = trace(A);

    return 0.5 * (trA * trA - trace(A * A));
  }

  //
  // R^N third invariant
  // \param A tensor
  // \return \f$ III_A = \det A \f$
  //
  template<typename T, Index N>
  inline
  T
  I3(Tensor<T, N> const & A)
  {
    return det(A);
  }

  //
  // R^3 third invariant
  // \param A tensor
  // \return \f$ III_A = \det A \f$
  //
  template<typename T>
  inline
  T
  I3(Tensor<T, 3> const & A)
  {
    return det(A);
  }

  //
  // R^2 third invariant
  // \param A tensor
  // \return \f$ III_A = \det A \f$
  //
  template<typename T>
  inline
  T
  I3(Tensor<T, 2> const & A)
  {
    return det(A);
  }

  //
  // R^N Indexing for constant 3rd order tensor
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T, Index N>
  inline
  const T &
  Tensor3<T, N>::operator()(
      const Index i,
      const Index j,
      const Index k) const
  {
    assert(i < N);
    assert(j < N);
    assert(k < N);
    return e[i][j][k];
  }

  //
  // R^3 indexing for constant 3rd order tensor
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  const T &
  Tensor3<T, 3>::operator()(
      const Index i,
      const Index j,
      const Index k) const
  {
    assert(i < 3);
    assert(j < 3);
    assert(k < 3);
    return e[i][j][k];
  }

  //
  // R^2 indexing for constant 3rd order tensor
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  const T &
  Tensor3<T, 2>::operator()(
      const Index i,
      const Index j,
      const Index k) const
  {
    assert(i < 2);
    assert(j < 2);
    assert(k < 2);
    return e[i][j][k];
  }

  //
  // R^N 3rd-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T, Index N>
  inline
  T &
  Tensor3<T, N>::operator()(const Index i, const Index j, const Index k)
  {
    assert(i < N);
    assert(j < N);
    assert(k < N);
    return e[i][j][k];
  }

  //
  // R^3 3rd-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  T &
  Tensor3<T, 3>::operator()(const Index i, const Index j, const Index k)
  {
    assert(i < 3);
    assert(j < 3);
    assert(k < 3);
    return e[i][j][k];
  }

  //
  // R^2 3rd-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  T &
  Tensor3<T, 2>::operator()(const Index i, const Index j, const Index k)
  {
    assert(i < 2);
    assert(j < 2);
    assert(k < 2);
    return e[i][j][k];
  }

  //
  // R^N indexing for constant 4th order tensor
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T, Index N>
  inline
  const T &
  Tensor4<T, N>::operator()(
      const Index i, const Index j, const Index k, const Index l) const
  {
    assert(i < N);
    assert(j < N);
    assert(k < N);
    assert(l < N);
    return e[i][j][k][l];
  }

  //
  // R^3 indexing for constant 4th order tensor
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  const T &
  Tensor4<T, 3>::operator()(
      const Index i, const Index j, const Index k, const Index l) const
  {
    assert(i < 3);
    assert(j < 3);
    assert(k < 3);
    assert(l < 3);
    return e[i][j][k][l];
  }

  //
  // R^2 indexing for constant 4th order tensor
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  const T &
  Tensor4<T, 2>::operator()(
      const Index i, const Index j, const Index k, const Index l) const
  {
    assert(i < 2);
    assert(j < 2);
    assert(k < 2);
    assert(l < 2);
    return e[i][j][k][l];
  }

  //
  // R^N 4th-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T, Index N>
  inline
  T &
  Tensor4<T, N>::operator()(
      const Index i, const Index j, const Index k, const Index l)
  {
    assert(i < N);
    assert(j < N);
    assert(k < N);
    assert(l < N);
    return e[i][j][k][l];
  }

  //
  // R^3 4th-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  T &
  Tensor4<T, 3>::operator()(
      const Index i, const Index j, const Index k, const Index l)
  {
    assert(i < 3);
    assert(j < 3);
    assert(k < 3);
    assert(l < 3);
    return e[i][j][k][l];
  }

  //
  // R^2 4th-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  T &
  Tensor4<T, 2>::operator()(
      const Index i, const Index j, const Index k, const Index l)
  {
    assert(i < 2);
    assert(j < 2);
    assert(k < 2);
    assert(l < 2);
    return e[i][j][k][l];
  }

} // namespace LCM

#endif // LCM_Tensor_i_cc
