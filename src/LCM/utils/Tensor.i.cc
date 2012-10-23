//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Tensor_i_cc)
#define LCM_Tensor_i_cc

namespace LCM {

  //
  // Sign function
  //
  template <typename T> int sgn(T const & s) {
    return (T(0) < s) - (s < T(0));
  }

  //
  // NaN function. Necessary to choose the proper underlying NaN
  // for non-floating-point types.
  // Assumption: non-floating-point types have a typedef that
  // determines the underlying floating-point type.
  //
  template<typename T>
  typename Sacado::ScalarType<T>::type
  not_a_number()
  {
    return
        std::numeric_limits<typename Sacado::ScalarType<T>::type>::quiet_NaN();
  }

  //
  // Machine epsilon function. Necessary to choose the proper underlying
  // machine epsilon for non-floating-point types.
  // Assumption: non-floating-point types have a typedef that
  // determines the underlying floating-point type.
  //
  template<typename T>
  typename Sacado::ScalarType<T>::type
  machine_epsilon()
  {
    return
        std::numeric_limits<typename Sacado::ScalarType<T>::type>::epsilon();
  }

  //
  // return dimension
  //
  template<typename T>
  inline
  Index
  Vector<T>::get_dimension() const
  {
    return e.size();
  }

  //
  // set dimension
  //
  template<typename T>
  inline
  void
  Vector<T>::set_dimension(const Index N)
  {
    e.resize(N);
    return;
  }

  //
  // default constructor
  //
  template<typename T>
  inline
  Vector<T>::Vector()
  {
    return;
  }

  //
  // constructor that initializes to NaNs
  //
  template<typename T>
  inline
  Vector<T>::Vector(const Index N)
  {

    set_dimension(N);

    switch (N) {

    default:
      for (Index i =0; i < N; ++i) {
        e[i] = not_a_number<T>();
      }
      break;

    case 3:
      e[0] = not_a_number<T>();
      e[1] = not_a_number<T>();
      e[2] = not_a_number<T>();
      break;

    case 2:
      e[0] = not_a_number<T>();
      e[1] = not_a_number<T>();
      break;

    }

    return;
  }

  //
  // R^N create vector from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T>
  inline
  Vector<T>::Vector(const Index N, T const & s)
  {
    set_dimension(N);

    switch (N) {

    default:
      for (Index i =0; i < N; ++i) {
        e[i] = s;
      }
      break;

    case 3:
      e[0] = s;
      e[1] = s;
      e[2] = s;
      break;

    case 2:
      e[0] = s;
      e[1] = s;
      break;

    }

    return;
  }

  //
  // Create vector specifying components
  // \param N dimension
  // \param s0, s1 are the vector components in the R^2 canonical basis
  //
  template<typename T>
  inline
  Vector<T>::Vector(T const & s0, T const & s1)
  {
    set_dimension(2);

    e[0] = s0;
    e[1] = s1;

    return;
  }

  //
  // Create vector specifying components
  // \param N dimension
  // \param s0, s1, s2 are the vector components in the R^3 canonical basis
  //
  template<typename T>
  inline
  Vector<T>::Vector(T const & s0, T const & s1, T const & s2)
  {
    set_dimension(3);

    e[0] = s0;
    e[1] = s1;
    e[2] = s2;

    return;
  }

  //
  // R^N create vector from array - const version
  // \param N dimension
  // \param data_ptr
  //
  template<typename T>
  inline
  Vector<T>::Vector(const Index N, T const * data_ptr)
  {
    assert(data_ptr != NULL);

    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        e[i] = data_ptr[i];
      }
      break;

    case 3:
      e[0] = data_ptr[0];
      e[1] = data_ptr[1];
      e[2] = data_ptr[2];
      break;

    case 2:
      e[0] = data_ptr[0];
      e[1] = data_ptr[1];
      break;

    }

    return;
  }

  //
  // R^N create vector from array
  // \param N dimension
  // \param data_ptr
  //
  template<typename T>
  inline
  Vector<T>::Vector(const Index N, T * data_ptr)
  {
    assert(data_ptr != NULL);

    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        e[i] = data_ptr[i];
      }
      break;

    case 3:
      e[0] = data_ptr[0];
      e[1] = data_ptr[1];
      e[2] = data_ptr[2];
      break;

    case 2:
      e[0] = data_ptr[0];
      e[1] = data_ptr[1];
      break;

    }

    return;
  }

  //
  // R^N copy constructor
  // \param v the values of its componets are copied to the new vector
  //
  template<typename T>
  inline
  Vector<T>::Vector(Vector<T> const & v)
  {
    const Index
    N = v.get_dimension();

    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        e[i] = v.e[i];
      }
      break;

    case 3:
      e[0] = v.e[0];
      e[1] = v.e[1];
      e[2] = v.e[2];
      break;

    case 2:
      e[0] = v.e[0];
      e[1] = v.e[1];
      break;

    }

    return;
  }

  //
  // R^N simple destructor
  //
  template<typename T>
  inline
  Vector<T>::~Vector()
  {
    return;
  }

  //
  // R^N indexing for constant vector
  // \param i the index
  //
  template<typename T>
  inline
  const T &
  Vector<T>::operator()(const Index i) const
  {
    assert(i < get_dimension());
    return e[i];
  }

  //
  // R^N vector indexing
  // \param i the index
  //
  template<typename T>
  inline
  T &
  Vector<T>::operator()(const Index i)
  {
    assert(i < get_dimension());
    return e[i];
  }

  //
  // R^N copy assignment
  // \param v the values of its componets are copied to this vector
  //
  template<typename T>
  inline
  Vector<T> &
  Vector<T>::operator=(Vector<T> const & v)
  {
    if (this != &v) {

      const Index
      N = v.get_dimension();

      set_dimension(N);

      switch (N) {

      default:
        for (Index i = 0; i < N; ++i) {
          e[i] = v.e[i];
        }
        break;

      case 3:
        e[0] = v.e[0];
        e[1] = v.e[1];
        e[2] = v.e[2];
        break;

      case 2:
        e[0] = v.e[0];
        e[1] = v.e[1];
        break;

      }
    }

    return *this;
  }

  //
  // R^N vector increment
  // \param v added to currrent vector
  //
  template<typename T>
  inline
  Vector<T> &
  Vector<T>::operator+=(Vector<T> const & v)
  {
    const Index
    N = get_dimension();

    assert(v.get_dimension() == N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        e[i] += v.e[i];
      }
      break;

    case 3:
      e[0] += v.e[0];
      e[1] += v.e[1];
      e[2] += v.e[2];
      break;

    case 2:
      e[0] += v.e[0];
      e[1] += v.e[1];
      break;

    }

    return *this;
  }

  //
  // R^N vector decrement
  // \param v substracted from current vector
  //
  template<typename T>
  inline
  Vector<T> &
  Vector<T>::operator-=(Vector<T> const & v)
  {
    const Index
    N = get_dimension();

    assert(v.get_dimension() == N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        e[i] -= v.e[i];
      }
      break;

    case 3:
      e[0] -= v.e[0];
      e[1] -= v.e[1];
      e[2] -= v.e[2];
      break;

    case 2:
      e[0] -= v.e[0];
      e[1] -= v.e[1];
      break;

    }

    return *this;
  }

  //
  // R^N fill with zeros
  //
  template<typename T>
  inline
  void
  Vector<T>::clear()
  {
    const Index
    N = get_dimension();

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        e[i] -= 0.0;
      }
      break;

    case 3:
      e[0] = 0.0;
      e[1] = 0.0;
      e[2] = 0.0;
      break;

    case 2:
      e[0] = 0.0;
      e[1] = 0.0;
      break;

    }

    return;
  }

  //
  // R^N vector addition
  // \param u
  // \param v the operands
  // \return \f$ u + v \f$
  //
  template<typename T>
  inline
  Vector<T>
  operator+(Vector<T> const & u, Vector<T> const & v)
  {
    const Index
    N = u.get_dimension();

    assert(v.get_dimension() == N);

    Vector<T> s(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        s(i) = u(i) + v(i);
      }
      break;

    case 3:
      s(0) = u(0) + v(0);
      s(1) = u(1) + v(1);
      s(2) = u(2) + v(2);
      break;

    case 2:
      s(0) = u(0) + v(0);
      s(1) = u(1) + v(1);
      break;

    }

    return s;
  }

  //
  // R^N vector substraction
  // \param u
  // \param v the operands
  // \return \f$ u - v \f$
  //
  template<typename T>
  inline
  Vector<T>
  operator-(Vector<T> const & u, Vector<T> const & v)
  {
    const Index
    N = u.get_dimension();

    assert(v.get_dimension() == N);

    Vector<T> s(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        s(i) = u(i) - v(i);
      }
      break;

    case 3:
      s(0) = u(0) - v(0);
      s(1) = u(1) - v(1);
      s(2) = u(2) - v(2);
      break;

    case 2:
      s(0) = u(0) - v(0);
      s(1) = u(1) - v(1);
      break;

    }

    return s;
  }

  //
  // R^N vector minus
  // \param u
  // \return \f$ -u \f$
  //
  template<typename T>
  inline
  Vector<T>
  operator-(Vector<T> const & u)
  {
    const Index
    N = u.get_dimension();

    Vector<T> v(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        v(i) = - u(i);
      }
      break;

    case 3:
      v(0) = - u(0);
      v(1) = - u(1);
      v(2) = - u(2);
      break;

    case 2:
      v(0) = - u(0);
      v(1) = - u(1);
      break;

    }

    return v;
  }

  //
  // R^N vector dot product
  // \param u
  // \param v the operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T>
  inline
  T
  operator*(Vector<T> const & u, Vector<T> const & v)
  {
    return dot(u, v);
  }

  //
  // R^N vector equality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \equiv v \f$
  //
  template<typename T>
  inline
  bool
  operator==(Vector<T> const & u, Vector<T> const & v)
  {
    const Index
    N = u.get_dimension();

    assert(v.get_dimension() == N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        if (v(i) != u(i)) {
          return false;
        }
      }
      break;

    case 3:
      return u(0)==v(0) && u(1)==v(1) && u(2)==v(2);
      break;

    case 2:
      return u(0)==v(0) && u(1)==v(1);
      break;

    }

    return true;
  }

  //
  // R^N, vector inequality tested by components
  // \param u
  // \param v the operands
  // \return \f$ u \neq v \f$
  //
  template<typename T>
  inline
  bool
  operator!=(Vector<T> const & u, Vector<T> const & v)
  {
    return !(u==v);
  }

  //
  // R^N scalar vector product
  // \param s scalar factor
  // \param u vector factor
  // \return \f$ s u \f$
  //
  template<typename T, typename S>
  inline
  Vector<T>
  operator*(S const & s, Vector<T> const & u)
  {
    const Index
    N = u.get_dimension();

    Vector<T> v(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        v(i) = s * u(i);
      }
      break;

    case 3:
      v(0) = s * u(0);
      v(1) = s * u(1);
      v(2) = s * u(2);
      break;

    case 2:
      v(0) = s * u(0);
      v(1) = s * u(1);
      break;

    }

    return v;
  }

  //
  // R^N vector scalar product
  // \param u vector factor
  // \param s scalar factor
  // \return \f$ s u \f$
  //
  template<typename T, typename S>
  inline
  Vector<T>
  operator*(Vector<T> const & u, S const & s)
  {
    return s * u;
  }

  //
  // R^N vector scalar division
  // \param u vector
  // \param s scalar that divides each component of vector
  // \return \f$ u / s \f$
  //
  template<typename T, typename S>
  inline
  Vector<T>
  operator/(Vector<T> const & u, S const & s)
  {
    const Index
    N = u.get_dimension();

    Vector<T> v(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        v(i) = u(i) / s;
      }
      break;

    case 3:
      v(0) = u(0) / s;
      v(1) = u(1) / s;
      v(2) = u(2) / s;
      break;

    case 2:
      v(0) = u(0) / s;
      v(1) = u(1) / s;
      break;

    }

    return v;
  }

  //
  // R^N vector dot product
  // \param u
  // \param v operands
  // \return \f$ u \cdot v \f$
  //
  template<typename T>
  inline
  T
  dot(Vector<T> const & u, Vector<T> const & v)
  {
    const Index
    N = u.get_dimension();

    assert(v.get_dimension() == N);

    T s = 0.0;

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        s += u(i) * v(i);
      }
      break;

    case 3:
      s = u(0)*v(0) + u(1)*v(1) + u(2)*v(2);
      break;

    case 2:
      s = u(0)*v(0) + u(1)*v(1);
      break;

    }

    return s;
  }

  //
  // Cross product only valid for R^3.
  // R^N with N != 3 will produce an error.
  // \param u
  // \param v operands
  // \return \f$ u \times v \f$
  //
  template<typename T>
  inline
  Vector<T>
  cross(Vector<T> const & u, Vector<T> const & v)
  {
    const Index
    N = u.get_dimension();

    assert(v.get_dimension() == N);

    Vector<T> w(N);

    switch (N) {

    default:
      std::cerr << "ERROR: Cross product undefined for R^" << N << std::endl;
      exit(1);
      break;

    case 3:
      w(0) = u(1)*v(2) - u(2)*v(1);
      w(1) = u(2)*v(0) - u(0)*v(2);
      w(2) = u(0)*v(1) - u(1)*v(0);
      break;

    }

    return w;
  }

  //
  // R^N vector 2-norm
  // \return \f$ \sqrt{u \cdot u} \f$
  //
  template<typename T>
  inline
  T
  norm(Vector<T> const & u)
  {
    const Index
    N = u.get_dimension();

    T s = 0.0;

    switch (N) {

    default:
      s = sqrt(dot(u, u));
      break;

    case 3:
      s = sqrt(u(0)*u(0) + u(1)*u(1) + u(2)*u(2));
      break;

    case 2:
      s = sqrt(u(0)*u(0) + u(1)*u(1));
      break;

    }

    return s;
  }

  //
  // R^N vector 1-norm
  // \return \f$ \sum_i |u_i| \f$
  //
  template<typename T>
  inline
  T
  norm_1(Vector<T> const & u)
  {
    const Index
    N = u.get_dimension();

    T s = 0.0;

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        s += std::abs(u(i));
      }
      break;

    case 3:
      s = std::abs(u(0)) + std::abs(u(1)) + std::abs(u(2));
      break;

    case 2:
      s = std::abs(u(0)) + std::abs(u(1));
      break;

    }

    return s;
  }

  //
  // R^N vector infinity-norm
  // \return \f$ \max(|u_0|,...|u_i|,...|u_N|) \f$
  //
  template<typename T>
  inline
  T
  norm_infinity(Vector<T> const & u)
  {
    const Index
    N = u.get_dimension();

    T s = 0.0;

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        s = std::max(s, std::abs(u(i)));
      }
      break;

    case 3:
      s = std::max(std::max(std::abs(u(0)),std::abs(u(1))),std::abs(u(2)));
      break;

    case 2:
      s = std::max(std::abs(u(0)),std::abs(u(1)));
      break;

    }

    return s;
  }

  namespace {

    template<typename S>
    bool
    greater_than(S const & a, S const & b)
    {
      return a.first > b.first;
    }

  } // anonymous namespace

  //
  // Sort and index in descending order. Useful for ordering singular values
  // and eigenvalues and corresponding vectors in the
  // respective decompositions.
  // \param u vector to sort
  // \return pair<v, P>
  // \return v sorted vector
  // \return P permutation matrix such that v = P^T u
  //
  template<typename T>
  std::pair<Vector<T>, Tensor<T> >
  sort_permutation(Vector<T> const & u)
  {

    const Index
    N = u.get_dimension();

    std::vector<std::pair<T, Index > >
    s(N);

    for (Index i = 0; i < N; ++i) {
      s[i].first = u(i);
      s[i].second = i;
    }

    std::sort(s.begin(), s.end(), greater_than< std::pair<T, Index > > );

    Vector<T> v(N);

    Tensor<T>
    P = zero<T>(N);

    for (Index i = 0; i < N; ++i) {
      v(i) = s[i].first;
      P(s[i].second, i) = 1.0;
    }

    return std::make_pair(v, P);

  }

  //
  // Dimension
  // get dimension
  //
  template<typename T>
  inline
  Index
  Tensor<T>::get_dimension() const
  {
    return e.size();
  }

  //
  // set dimension
  //
  //
  template<typename T>
  inline
  void
  Tensor<T>::set_dimension(const Index N)
  {

    switch (N) {

    default:
      e.resize(N);
      for (Index i = 0; i < N; ++i) {
        e[i].resize(N);
      }
      break;

    case 3:
      e.resize(3);
      e[0].resize(3);
      e[1].resize(3);
      e[2].resize(3);
      break;

    case 2:
      e.resize(2);
      e[0].resize(2);
      e[1].resize(2);
      break;

    }

    return;
  }

  //
  // default constructor
  //
  template<typename T>
  inline
  Tensor<T>::Tensor()
  {
    return;
  }

  //
  // constructor that initializes to NaNs
  //
  template<typename T>
  inline
  Tensor<T>::Tensor(const Index N)
  {
    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; j++) {
          e[i][j] = not_a_number<T>();
        }
      }
      break;

    case 3:
      e[0][0] = not_a_number<T>();
      e[0][1] = not_a_number<T>();
      e[0][2] = not_a_number<T>();

      e[1][0] = not_a_number<T>();
      e[1][1] = not_a_number<T>();
      e[1][2] = not_a_number<T>();

      e[2][0] = not_a_number<T>();
      e[2][1] = not_a_number<T>();
      e[2][2] = not_a_number<T>();
      break;

    case 2:
      e[0][0] = not_a_number<T>();
      e[0][1] = not_a_number<T>();

      e[1][0] = not_a_number<T>();
      e[1][1] = not_a_number<T>();
      break;

    }
    return;
  }

  //
  // R^N create tensor from a scalar
  // \param s all components are set equal to this value
  //
  template<typename T>
  inline
  Tensor<T>::Tensor(const Index N, T const & s)
  {
    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; j++) {
          e[i][j] = s;
        }
      }
      break;

    case 3:
      e[0][0] = s;
      e[0][1] = s;
      e[0][2] = s;

      e[1][0] = s;
      e[1][1] = s;
      e[1][2] = s;

      e[2][0] = s;
      e[2][1] = s;
      e[2][2] = s;
      break;

    case 2:
      e[0][0] = s;
      e[0][1] = s;

      e[1][0] = s;
      e[1][1] = s;
      break;

    }

    return;
  }

  //
  // Create tensor specifying components
  // \param N dimension
  // \param  s00, s01, ... components in the R^2 canonical basis
  //
  template<typename T>
  inline
  Tensor<T>::Tensor(T const & s00, T const & s01, T const & s10, T const & s11)
  {
    set_dimension(2);

    e[0][0] = s00;
    e[0][1] = s01;

    e[1][0] = s10;
    e[1][1] = s11;

    return;
  }

  //
  // Create tensor specifying components
  // \param N dimension
  // \param  s00, s01, ... components in the R^3 canonical basis
  //
  template<typename T>
  inline
  Tensor<T>::Tensor(
      T const & s00, T const & s01, T const & s02,
      T const & s10, T const & s11, T const & s12,
      T const & s20, T const & s21, T const & s22)
  {
    set_dimension(3);

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
  // R^N create tensor from array - const version
  // \param data_ptr
  //
  template<typename T>
  inline
  Tensor<T>::Tensor(const Index N, T const * data_ptr)
  {
    assert(data_ptr != NULL);

    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          e[i][j] = data_ptr[i * N + j];
        }
      }
      break;

    case 3:
      e[0][0] = data_ptr[0];
      e[0][1] = data_ptr[1];
      e[0][2] = data_ptr[2];

      e[1][0] = data_ptr[3];
      e[1][1] = data_ptr[4];
      e[1][2] = data_ptr[5];

      e[2][0] = data_ptr[6];
      e[2][1] = data_ptr[7];
      e[2][2] = data_ptr[8];
      break;

    case 2:
      e[0][0] = data_ptr[0];
      e[0][1] = data_ptr[1];

      e[1][0] = data_ptr[2];
      e[1][1] = data_ptr[3];
      break;

    }

    return;
  }

  //
  // R^N create tensor from array
  // \param data_ptr
  //
  template<typename T>
  inline
  Tensor<T>::Tensor(const Index N, T * data_ptr)
  {
    assert(data_ptr != NULL);

    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          e[i][j] = data_ptr[i * N + j];
        }
      }
      break;

    case 3:
      e[0][0] = data_ptr[0];
      e[0][1] = data_ptr[1];
      e[0][2] = data_ptr[2];

      e[1][0] = data_ptr[3];
      e[1][1] = data_ptr[4];
      e[1][2] = data_ptr[5];

      e[2][0] = data_ptr[6];
      e[2][1] = data_ptr[7];
      e[2][2] = data_ptr[8];
      break;

    case 2:
      e[0][0] = data_ptr[0];
      e[0][1] = data_ptr[1];

      e[1][0] = data_ptr[2];
      e[1][1] = data_ptr[3];
      break;

    }

    return;
  }

  //
  // R^N copy constructor
  // \param A the values of its componets are copied to the new tensor
  //
  template<typename T>
  inline
  Tensor<T>::Tensor(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    set_dimension(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          e[i][j] = A.e[i][j];
        }
      }
      break;

    case 3:
      e[0][0] = A.e[0][0];
      e[0][1] = A.e[0][1];
      e[0][2] = A.e[0][2];

      e[1][0] = A.e[1][0];
      e[1][1] = A.e[1][1];
      e[1][2] = A.e[1][2];

      e[2][0] = A.e[2][0];
      e[2][1] = A.e[2][1];
      e[2][2] = A.e[2][2];
      break;

    case 2:
      e[0][0] = A.e[0][0];
      e[0][1] = A.e[0][1];

      e[1][0] = A.e[1][0];
      e[1][1] = A.e[1][1];
      break;

    }

    return;
  }

  //
  // R^N simple destructor
  //
  template<typename T>
  inline
  Tensor<T>::~Tensor()
  {
    return;
  }

  //
  // R^N indexing for constant tensor
  // \param i index
  // \param j index
  //
  template<typename T>
  inline
  T const &
  Tensor<T>::operator()(const Index i, const Index j) const
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    return e[i][j];
  }

  //
  // R^N tensor indexing
  // \param i index
  // \param j index
  //
  template<typename T>
  inline
  T &
  Tensor<T>::operator()(const Index i, const Index j)
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    return e[i][j];
  }

  //
  // R^N copy assignment
  // \param A the values of its componets are copied to this tensor
  //
  template<typename T>
  inline
  Tensor<T> &
  Tensor<T>::operator=(Tensor<T> const & A)
  {
    if (this != &A) {

      const Index
      N = A.get_dimension();

      set_dimension(N);

      switch (N) {

      default:
        for (Index i = 0; i < N; ++i) {
          for (Index j = 0; j < N; ++j) {
            e[i][j] = A.e[i][j];
          }
        }
        break;

      case 3:
        e[0][0] = A.e[0][0];
        e[0][1] = A.e[0][1];
        e[0][2] = A.e[0][2];

        e[1][0] = A.e[1][0];
        e[1][1] = A.e[1][1];
        e[1][2] = A.e[1][2];

        e[2][0] = A.e[2][0];
        e[2][1] = A.e[2][1];
        e[2][2] = A.e[2][2];
        break;

      case 2:
        e[0][0] = A.e[0][0];
        e[0][1] = A.e[0][1];

        e[1][0] = A.e[1][0];
        e[1][1] = A.e[1][1];
        break;

      }

    }

    return *this;
  }

  //
  // R^N tensor increment
  // \param A added to current tensor
  //
  template<typename T>
  inline
  Tensor<T> &
  Tensor<T>::operator+=(Tensor<T> const & A)
  {
    const Index
    N = get_dimension();

    assert(A.get_dimension() == N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          e[i][j] += A.e[i][j];
        }
      }
      break;

    case 3:
      e[0][0] += A.e[0][0];
      e[0][1] += A.e[0][1];
      e[0][2] += A.e[0][2];

      e[1][0] += A.e[1][0];
      e[1][1] += A.e[1][1];
      e[1][2] += A.e[1][2];

      e[2][0] += A.e[2][0];
      e[2][1] += A.e[2][1];
      e[2][2] += A.e[2][2];
      break;

    case 2:
      e[0][0] += A.e[0][0];
      e[0][1] += A.e[0][1];

      e[1][0] += A.e[1][0];
      e[1][1] += A.e[1][1];
      break;

    }

    return *this;
  }

  //
  // R^N tensor decrement
  // \param A substracted from current tensor
  //
  template<typename T>
  inline
  Tensor<T> &
  Tensor<T>::operator-=(Tensor<T> const & A)
  {
    const Index
    N = get_dimension();

    assert(A.get_dimension() == N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          e[i][j] -= A.e[i][j];
        }
      }
      break;

    case 3:
      e[0][0] -= A.e[0][0];
      e[0][1] -= A.e[0][1];
      e[0][2] -= A.e[0][2];

      e[1][0] -= A.e[1][0];
      e[1][1] -= A.e[1][1];
      e[1][2] -= A.e[1][2];

      e[2][0] -= A.e[2][0];
      e[2][1] -= A.e[2][1];
      e[2][2] -= A.e[2][2];
      break;

    case 2:
      e[0][0] -= A.e[0][0];
      e[0][1] -= A.e[0][1];

      e[1][0] -= A.e[1][0];
      e[1][1] -= A.e[1][1];
      break;

    }

    return *this;
  }

  //
  // R^N fill with zeros
  //
  template<typename T>
  inline
  void
  Tensor<T>::clear()
  {
    const Index
    N = get_dimension();

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          e[i][j] = 0.0;
        }
      }
      break;

    case 3:
      e[0][0] = 0.0;
      e[0][1] = 0.0;
      e[0][2] = 0.0;

      e[1][0] = 0.0;
      e[1][1] = 0.0;
      e[1][2] = 0.0;

      e[2][0] = 0.0;
      e[2][1] = 0.0;
      e[2][2] = 0.0;
      break;

    case 2:
      e[0][0] = 0.0;
      e[0][1] = 0.0;

      e[1][0] = 0.0;
      e[1][1] = 0.0;
      break;

    }

    return;
  }

  //
  // R^N tensor addition
  // \return \f$ A + B \f$
  //
  template<typename T>
  inline
  Tensor<T>
  operator+(Tensor<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor<T> S(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          S(i, j) = A(i, j) + B(i, j);
        }
      }
      break;

    case 3:
      S(0,0) = A(0,0) + B(0,0);
      S(0,1) = A(0,1) + B(0,1);
      S(0,2) = A(0,2) + B(0,2);

      S(1,0) = A(1,0) + B(1,0);
      S(1,1) = A(1,1) + B(1,1);
      S(1,2) = A(1,2) + B(1,2);

      S(2,0) = A(2,0) + B(2,0);
      S(2,1) = A(2,1) + B(2,1);
      S(2,2) = A(2,2) + B(2,2);
      break;

    case 2:
      S(0,0) = A(0,0) + B(0,0);
      S(0,1) = A(0,1) + B(0,1);

      S(1,0) = A(1,0) + B(1,0);
      S(1,1) = A(1,1) + B(1,1);
      break;

    }

    return S;
  }

  //
  // R^N Tensor substraction
  // \return \f$ A - B \f$
  //
  template<typename T>
  inline
  Tensor<T>
  operator-(Tensor<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor<T> S(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          S(i, j) = A(i, j) - B(i, j);
        }
      }
      break;

    case 3:
      S(0,0) = A(0,0) - B(0,0);
      S(0,1) = A(0,1) - B(0,1);
      S(0,2) = A(0,2) - B(0,2);

      S(1,0) = A(1,0) - B(1,0);
      S(1,1) = A(1,1) - B(1,1);
      S(1,2) = A(1,2) - B(1,2);

      S(2,0) = A(2,0) - B(2,0);
      S(2,1) = A(2,1) - B(2,1);
      S(2,2) = A(2,2) - B(2,2);
      break;

    case 2:
      S(0,0) = A(0,0) - B(0,0);
      S(0,1) = A(0,1) - B(0,1);

      S(1,0) = A(1,0) - B(1,0);
      S(1,1) = A(1,1) - B(1,1);
      break;

    }

    return S;
  }

  //
  // R^N tensor minus
  // \return \f$ -A \f$
  //
  template<typename T>
  inline
  Tensor<T>
  operator-(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T> S(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          S(i, j) = - A(i, j);
        }
      }
      break;

    case 3:
      S(0,0) = -A(0,0);
      S(0,1) = -A(0,1);
      S(0,2) = -A(0,2);

      S(1,0) = -A(1,0);
      S(1,1) = -A(1,1);
      S(1,2) = -A(1,2);

      S(2,0) = -A(2,0);
      S(2,1) = -A(2,1);
      S(2,2) = -A(2,2);
      break;

    case 2:
      S(0,0) = -A(0,0);
      S(0,1) = -A(0,1);

      S(1,0) = -A(1,0);
      S(1,1) = -A(1,1);
      break;

    }

    return S;
  }

  //
  // R^N tensor equality
  // Tested by components
  // \return \f$ A \equiv B \f$
  //
  template<typename T>
  inline
  bool
  operator==(Tensor<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
           if (A(i, j) != B(i, j)) {
             return false;
           }
        }
      }
      break;

    case 3:
      return
          A(0,0)==B(0,0) && A(0,1)==B(0,1) && A(0,2)==B(0,2) &&
          A(1,0)==B(1,0) && A(1,1)==B(1,1) && A(1,2)==B(1,2) &&
          A(2,0)==B(2,0) && A(2,1)==B(2,1) && A(2,2)==B(2,2);
      break;

    case 2:
      return
          A(0,0)==B(0,0) && A(0,1)==B(0,1) &&
          A(1,0)==B(1,0) && A(1,1)==B(1,1);
      break;

    }

    return true;
  }

  //
  // R^N tensor inequality
  // Tested by components
  // \return \f$ A \neq B \f$
  //
  template<typename T>
  inline
  bool
  operator!=(Tensor<T> const & A, Tensor<T> const & B)
  {
    return !(A == B);
  }

  //
  // R^N scalar tensor product
  // \param s scalar
  // \param A tensor
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  inline
  Tensor<T>
  operator*(S const & s, Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T> B(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          B(i, j) = s * A(i, j);
        }
      }
      break;

    case 3:
      B(0,0) = s * A(0,0);
      B(0,1) = s * A(0,1);
      B(0,2) = s * A(0,2);

      B(1,0) = s * A(1,0);
      B(1,1) = s * A(1,1);
      B(1,2) = s * A(1,2);

      B(2,0) = s * A(2,0);
      B(2,1) = s * A(2,1);
      B(2,2) = s * A(2,2);
      break;

    case 2:
      B(0,0) = s * A(0,0);
      B(0,1) = s * A(0,1);

      B(1,0) = s * A(1,0);
      B(1,1) = s * A(1,1);
      break;

    }

    return B;
  }

  //
  // R^N tensor scalar product
  // \param A tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  inline
  Tensor<T>
  operator*(Tensor<T> const & A, S const & s)
  {
    return s * A;
  }

  //
  // R^N tensor scalar division
  // \param A tensor
  // \param s scalar
  // \return \f$ A / s \f$
  //
  template<typename T, typename S>
  inline
  Tensor<T>
  operator/(Tensor<T> const & A, S const & s)
  {
    const Index
    N = A.get_dimension();

    Tensor<T> B(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          B(i, j) = A(i, j) / s;
        }
      }
      break;

    case 3:
      B(0,0) = A(0,0) / s;
      B(0,1) = A(0,1) / s;
      B(0,2) = A(0,2) / s;

      B(1,0) = A(1,0) / s;
      B(1,1) = A(1,1) / s;
      B(1,2) = A(1,2) / s;

      B(2,0) = A(2,0) / s;
      B(2,1) = A(2,1) / s;
      B(2,2) = A(2,2) / s;
      break;

    case 2:
      B(0,0) = A(0,0) / s;
      B(0,1) = A(0,1) / s;

      B(1,0) = A(1,0) / s;
      B(1,1) = A(1,1) / s;
      break;

    }

    return B;
  }

  //
  // R^N tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  inline
  Vector<T>
  operator*(Tensor<T> const & A, Vector<T> const & u)
  {
    return dot(A, u);
  }

  //
  // R^N vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T>
  inline
  Vector<T>
  operator*(Vector<T> const & u, Tensor<T> const & A)
  {
    return dot(u, A);
  }

  //
  // R^N tensor vector product v = A u
  // \param A tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  inline
  Vector<T>
  dot(Tensor<T> const & A, Vector<T> const & u)
  {
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Vector<T> v(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        T s = 0.0;
        for (Index j = 0; j < N; ++j) {
          s += A(i, j) * u(j);
        }
        v(i) = s;
      }
      break;

    case 3:
      v(0) = A(0,0)*u(0) + A(0,1)*u(1) + A(0,2)*u(2);
      v(1) = A(1,0)*u(0) + A(1,1)*u(1) + A(1,2)*u(2);
      v(2) = A(2,0)*u(0) + A(2,1)*u(1) + A(2,2)*u(2);
      break;

    case 2:
      v(0) = A(0,0)*u(0) + A(0,1)*u(1);
      v(1) = A(1,0)*u(0) + A(1,1)*u(1);
      break;

    }

    return v;
  }

  //
  // R^N vector tensor product v = u A
  // \param A tensor
  // \param u vector
  // \return \f$ u A = A^T u \f$
  //
  template<typename T>
  inline
  Vector<T>
  dot(Vector<T> const & u, Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Vector<T> v(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        T s = 0.0;
        for (Index j = 0; j < N; ++j) {
          s += A(j, i) * u(j);
        }
        v(i) = s;
      }
      break;

    case 3:
      v(0) = A(0,0)*u(0) + A(1,0)*u(1) + A(2,0)*u(2);
      v(1) = A(0,1)*u(0) + A(1,1)*u(1) + A(2,1)*u(2);
      v(2) = A(0,2)*u(0) + A(1,2)*u(1) + A(2,2)*u(2);
      break;

    case 2:
      v(0) = A(0,0)*u(0) + A(1,0)*u(1);
      v(1) = A(0,1)*u(0) + A(1,1)*u(1);
      break;

    }

    return v;
  }

  //
  // R^N tensor dot product C = A B
  // \return \f$ A \cdot B \f$
  //
  template<typename T>
  inline
  Tensor<T>
  operator*(Tensor<T> const & A, Tensor<T> const & B)
  {
    return dot(A, B);
  }

  //
  // R^N tensor tensor product C = A B
  // \param A tensor
  // \param B tensor
  // \return a tensor \f$ A \cdot B \f$
  //
  template<typename T>
  inline
  Tensor<T>
  dot(Tensor<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor<T> C(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          T s = 0.0;
          for (Index k = 0; k < N; ++k) {
            s += A(i, k) * B(k, j);
          }
          C(i, j) = s;
        }
      }
      break;

    case 3:
      C(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0) + A(0,2)*B(2,0);
      C(0,1) = A(0,0)*B(0,1) + A(0,1)*B(1,1) + A(0,2)*B(2,1);
      C(0,2) = A(0,0)*B(0,2) + A(0,1)*B(1,2) + A(0,2)*B(2,2);

      C(1,0) = A(1,0)*B(0,0) + A(1,1)*B(1,0) + A(1,2)*B(2,0);
      C(1,1) = A(1,0)*B(0,1) + A(1,1)*B(1,1) + A(1,2)*B(2,1);
      C(1,2) = A(1,0)*B(0,2) + A(1,1)*B(1,2) + A(1,2)*B(2,2);

      C(2,0) = A(2,0)*B(0,0) + A(2,1)*B(1,0) + A(2,2)*B(2,0);
      C(2,1) = A(2,0)*B(0,1) + A(2,1)*B(1,1) + A(2,2)*B(2,1);
      C(2,2) = A(2,0)*B(0,2) + A(2,1)*B(1,2) + A(2,2)*B(2,2);
      break;

    case 2:
      C(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0);
      C(0,1) = A(0,0)*B(0,1) + A(0,1)*B(1,1);

      C(1,0) = A(1,0)*B(0,0) + A(1,1)*B(1,0);
      C(1,1) = A(1,0)*B(0,1) + A(1,1)*B(1,1);
      break;

    }

    return C;
  }

  //
  // R^N tensor tensor double dot product (contraction)
  // \param A tensor
  // \param B tensor
  // \return a scalar \f$ A : B \f$
  //
  template<typename T>
  inline
  T
  dotdot(Tensor<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    T s = 0.0;

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          s += A(i, j) * B(i, j);
        }
      }
      break;

    case 3:
      s+= A(0,0)*B(0,0) + A(0,1)*B(0,1) + A(0,2)*B(0,2);
      s+= A(1,0)*B(1,0) + A(1,1)*B(1,1) + A(1,2)*B(1,2);
      s+= A(2,0)*B(2,0) + A(2,1)*B(2,1) + A(2,2)*B(2,2);
      break;

    case 2:
      s+= A(0,0)*B(0,0) + A(0,1)*B(0,1);
      s+= A(1,0)*B(1,0) + A(1,1)*B(1,1);
      break;

    }

    return s;
  }

  //
  // R^N tensor Frobenius norm
  // \return \f$ \sqrt{A:A} \f$
  //
  template<typename T>
  inline
  T
  norm(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    T s = 0.0;

    switch (N) {

    default:
      s = dotdot(A, A);
      break;

    case 3:
      s+= A(0,0)*A(0,0) + A(0,1)*A(0,1) + A(0,2)*A(0,2);
      s+= A(1,0)*A(1,0) + A(1,1)*A(1,1) + A(1,2)*A(1,2);
      s+= A(2,0)*A(2,0) + A(2,1)*A(2,1) + A(2,2)*A(2,2);
      break;

    case 2:
      s+= A(0,0)*A(0,0) + A(0,1)*A(0,1);
      s+= A(1,0)*A(1,0) + A(1,1)*A(1,1);
      break;

    }

    return sqrt(s);
  }

  //
  // R^N tensor 1-norm
  // \return \f$ \max_{j \in {0,\cdots,N}}\Sigma_{i=0}^N |A_{ij}| \f$
  //
  template<typename T>
  inline
  T
  norm_1(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Vector<T> v(N);

    T s = 0.0;

    switch (N) {

    default:

      for (Index i = 0; i < N; ++i) {
        T t = 0.0;
        for (Index j = 0; j < N; ++j) {
          t += std::abs(A(j, i));
        }
        v(i) = t;
      }

      for (Index i = 0; i < N; ++i) {
        s = std::max(s, v(i));
      }
      break;

    case 3:
      v(0) = std::abs(A(0,0)) + std::abs(A(1,0)) + std::abs(A(2,0));
      v(1) = std::abs(A(0,1)) + std::abs(A(1,1)) + std::abs(A(2,1));
      v(2) = std::abs(A(0,2)) + std::abs(A(1,2)) + std::abs(A(2,2));

      s = std::max(std::max(v(0),v(1)),v(2));
      break;

    case 2:
      v(0) = std::abs(A(0,0)) + std::abs(A(1,0));
      v(1) = std::abs(A(0,1)) + std::abs(A(1,1));

      s = std::max(v(0),v(1));
      break;

    }

    return s;
  }

  //
  // R^N tensor infinity-norm
  // \return \f$ \max_{i \in {0,\cdots,N}}\Sigma_{j=0}^N |A_{ij}| \f$
  //
  template<typename T>
  inline
  T
  norm_infinity(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Vector<T> v(N);

    T s = 0.0;

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        T t = 0.0;
        for (Index j = 0; j < N; ++j) {
          t += std::abs(A(i, j));
        }
        v(i) = t;
      }

      for (Index i = 0; i < N; ++i) {
        s = std::max(s, v(i));
      }
      break;

    case 3:
      v(0) = std::abs(A(0,0)) + std::abs(A(0,1)) + std::abs(A(0,2));
      v(1) = std::abs(A(1,0)) + std::abs(A(1,1)) + std::abs(A(1,2));
      v(2) = std::abs(A(2,0)) + std::abs(A(2,1)) + std::abs(A(2,2));

      s = std::max(std::max(v(0),v(1)),v(2));
      break;

    case 2:
      v(0) = std::abs(A(0,0)) + std::abs(A(0,1));
      v(1) = std::abs(A(1,0)) + std::abs(A(1,1));

      s = std::max(v(0),v(1));
      break;

    }

    return s;
  }

  //
  // R^N dyad
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T>
  dyad(Vector<T> const & u, Vector<T> const & v)
  {
    const Index
    N = u.get_dimension();

    assert(v.get_dimension() == N);

    Tensor<T> A(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        const T s = u(i);
        for (Index j = 0; j < N; ++j) {
          A(i, j) = s * v(j);
        }
      }
      break;

    case 3:
      A(0,0) = u(0) * v(0);
      A(0,1) = u(0) * v(1);
      A(0,2) = u(0) * v(2);

      A(1,0) = u(1) * v(0);
      A(1,1) = u(1) * v(1);
      A(1,2) = u(1) * v(2);

      A(2,0) = u(2) * v(0);
      A(2,1) = u(2) * v(1);
      A(2,2) = u(2) * v(2);
      break;

    case 2:
      A(0,0) = u(0) * v(0);
      A(0,1) = u(0) * v(1);

      A(1,0) = u(1) * v(0);
      A(1,1) = u(1) * v(1);
      break;

    }

    return A;
  }

  //
  // R^N bun operator, just for Jay
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T>
  bun(Vector<T> const & u, Vector<T> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^N tensor product
  // \param u vector
  // \param v vector
  // \return \f$ u \otimes v \f$
  //
  template<typename T>
  inline
  Tensor<T>
  tensor(Vector<T> const & u, Vector<T> const & v)
  {
    return dyad(u, v);
  }

  //
  // R^N diagonal tensor from vector
  // \param v vector
  // \return A = diag(v)
  //
  template<typename T>
  Tensor<T>
  diag(Vector<T> const & v)
  {
    const Index
    N = v.get_dimension();

    Tensor<T> A = zero<T>(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        A(i, i) = v(i);
      }
      break;

    case 3:
      A(0,0) = v(0);
      A(1,1) = v(1);
      A(2,2) = v(2);
      break;

    case 2:
      A(0,0) = v(0);
      A(1,1) = v(1);
      break;

    }

    return A;
  }

  //
  // R^N diagonal of tensor in a vector
  // \param A tensor
  // \return v = diag(A)
  //
  template<typename T>
  Vector<T>
  diag(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Vector<T> v(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        v(i) = A(i, i);
      }
      break;

    case 3:
      v(0) = A(0,0);
      v(1) = A(1,1);
      v(2) = A(2,2);
      break;

    case 2:
      v(0) = A(0,0);
      v(1) = A(1,1);
      break;

    }

    return v;
  }

  //
  // R^N zero 2nd-order tensor
  // All components are zero
  //
  template<typename T>
  inline
  const Tensor<T>
  zero(const Index N)
  {
    return Tensor<T>(N, T(0.0));
  }

  //
  // R^N 2nd-order identity tensor
  //
  template<typename T>
  inline
  const Tensor<T>
  identity(const Index N)
  {
    Tensor<T> A(N, T(0.0));

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        A(i, i) = 1.0;
      }
      break;

    case 3:
      A(0,0) = 1.0;
      A(1,1) = 1.0;
      A(2,2) = 1.0;
      break;

    case 2:
      A(0,0) = 1.0;
      A(1,1) = 1.0;
      break;

    }

    return A;
  }

  //
  // R^N 2nd-order identity tensor, à la Matlab
  //
  template<typename T>
  inline
  const Tensor<T>
  eye(const Index N)
  {
    return identity<T>(N);
  }

  //
  // R^N 2nd-order tensor transpose
  //
  template<typename T>
  inline
  Tensor<T>
  transpose(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T> B(N);

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          B(i, j) = A(j, i);
        }
      }
      break;

    case 3:
      B(0,0) = A(0,0);
      B(0,1) = A(1,0);
      B(0,2) = A(2,0);

      B(1,0) = A(0,1);
      B(1,1) = A(1,1);
      B(1,2) = A(2,1);

      B(2,0) = A(0,2);
      B(2,1) = A(1,2);
      B(2,2) = A(2,2);
      break;

    case 2:
      B(0,0) = A(0,0);
      B(0,1) = A(1,0);

      B(1,0) = A(0,1);
      B(1,1) = A(1,1);
      break;

    }

    return B;
  }

  //
  // R^N 4th-order tensor transpose
  // per Holzapfel 1.157
  //
  template<typename T>
  inline
  Tensor4<T>
  transpose(Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor4<T> B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            B(i,j,k,l)  = A(k,l,i,j);
          }
        }
      }
    }

    return B;
  }

  //
  // R^N symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A + A^T) \f$
  //
  template<typename T>
  inline Tensor<T> symm(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T> B(N);

    switch (N) {

    default:
      B = 0.5 * (A + transpose(A));
      break;

    case 3:
      {
        T const & s00 = A(0,0);
        T const & s11 = A(1,1);
        T const & s22 = A(2,2);

        T const & s01 = 0.5 * (A(0,1) + A(1,0));
        T const & s02 = 0.5 * (A(0,2) + A(2,0));
        T const & s12 = 0.5 * (A(1,2) + A(2,1));

        B(0,0) = s00;
        B(0,1) = s01;
        B(0,2) = s02;

        B(1,0) = s01;
        B(1,1) = s11;
        B(1,2) = s12;

        B(2,0) = s02;
        B(2,1) = s12;
        B(2,2) = s22;
      }
      break;

    case 2:
      {
        T const & s00 = A(0,0);
        T const & s11 = A(1,1);

        T const & s01 = 0.5 * (A(0,1) + A(1,0));

        B(0,0) = s00;
        B(0,1) = s01;

        B(1,0) = s01;
        B(1,1) = s11;
      }
      break;

    }

    return B;
  }

  //
  // R^N skew symmetric part of 2nd-order tensor
  // \return \f$ \frac{1}{2}(A - A^T) \f$
  //
  template<typename T>
  inline
  Tensor<T>
  skew(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T> B(N);

    switch (N) {

    default:
      B = 0.5 * (A - transpose(A));
      break;

    case 3:
      {
        T const & s01 = 0.5*(A(0,1)-A(1,0));
        T const & s02 = 0.5*(A(0,2)-A(2,0));
        T const & s12 = 0.5*(A(1,2)-A(2,1));

        B(0,0) = 0.0;
        B(0,1) = s01;
        B(0,2) = s02;

        B(1,0) = -s01;
        B(1,1) = 0.0;
        B(1,2) = s12;

        B(2,0) = -s02;
        B(2,1) = -s12;
        B(2,2) = 0.0;
      }
      break;

    case 2:
      {
        T const & s01 = 0.5*(A(0,1)-A(1,0));

        B(0,0) = 0.0;
        B(0,1) = s01;

        B(1,0) = -s01;
        B(1,1) = 0.0;
      }
      break;

    }

    return B;
  }

  //
  // R^N skew symmetric 2nd-order tensor from vector, undefined
  // \param u vector
  //
  template<typename T>
  inline
  Tensor<T>
  skew(Vector<T> const & u)
  {
    const Index
    N = u.get_dimension();

    Tensor<T> A(N);

    switch (N) {

    default:
      std::cerr << "ERROR: Skew from vector undefined for R^" << N << std::endl;
      exit(1);
      break;

    case 3:
      A(0,0) = 0.0;
      A(0,1) = -u(2);
      A(0,2) = u(1);

      A(1,0) = u(2);
      A(1,1) = 0.0;
      A(1,2) = -u(0);

      A(2,0) = -u(1);
      A(2,1) = u(0);
      A(2,2) = 0.0;
      break;

    }

    return A;
  }

  //
  // R^N volumetric part of 2nd-order tensor
  // \return \f$ \frac{1}{N} \mathrm{tr}\:(A) I \f$
  //
  template<typename T>
  inline
  Tensor<T>
  vol(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    const T theta = (1.0/T(N)) * trace(A);

    return theta * eye<T>(N);
  }

  //
  // R^N deviatoric part of 2nd-order tensor
  // \return \f$ A - vol(A) \f$
  //
  template<typename T>
  inline
  Tensor<T>
  dev(Tensor<T> const & A)
  {
    return A - vol(A);
  }

  //
  // Swap row. Echange rows i and j in place
  // \param A tensor
  // \param i index
  // \param j index
  //
  template<typename T>
  void
  swap_row(Tensor<T> & A, Index i, Index j)
  {
    const Index
    N = A.get_dimension();

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
  template<typename T>
  void
  swap_col(Tensor<T> & A, Index i, Index j)
  {
    const Index
    N = A.get_dimension();

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
  template<typename T>
  inline
  T
  det(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    T s = 0.0;

    switch (N) {

    default:
      {
        int sign = 1;
        for (Index i = 0; i < N; ++i) {
          const T d = det(subtensor(A, i, 1));
          s += sign * d * A(i, 1);
          sign *= -1;
        }
      }
      break;

    case 3:
      s =
      -A(0,2)*A(1,1)*A(2,0) + A(0,1)*A(1,2)*A(2,0) +
       A(0,2)*A(1,0)*A(2,1) - A(0,0)*A(1,2)*A(2,1) -
       A(0,1)*A(1,0)*A(2,2) + A(0,0)*A(1,1)*A(2,2);
      break;

    case 2:
      s = A(0,0) * A(1,1) - A(1,0) * A(0,1);
      break;

    }

    return s;
  }

  //
  // R^N trace
  // \param A tensor
  // \return \f$ A:I \f$
  //
  template<typename T>
  inline
  T
  trace(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    T s = 0.0;

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        s += A(i,i);
      }
      break;

    case 3:
      s = A(0,0) + A(1,1) + A(2,2);
      break;

    case 2:
      s = A(0,0) + A(1,1);
      break;

    }

    return s;
  }

  //
  // R^N first invariant, trace
  // \param A tensor
  // \return \f$ I_A = A:I \f$
  //
  template<typename T>
  inline
  T
  I1(Tensor<T> const & A)
  {
    return trace(A);
  }

  //
  // R^N second invariant
  // \param A tensor
  // \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
  //
  template<typename T>
  inline
  T
  I2(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    T s = 0.0;
    const T trA = trace(A);

    switch (N) {

    default:
      s = 0.5 * (trA * trA - trace(A * A));
      break;

    case 3:
      s = 0.5 * (trA*trA - A(0,0)*A(0,0) - A(1,1)*A(1,1) - A(2,2)*A(2,2)) -
          A(0,1)*A(1,0) - A(0,2)*A(2,0) - A(1,2)*A(2,1);
      break;

    case 2:
      s =  0.5 * (trA * trA - trace(A * A));
      break;

    }

    return s;
  }

  //
  // R^N third invariant
  // \param A tensor
  // \return \f$ III_A = \det A \f$
  //
  template<typename T>
  inline
  T
  I3(Tensor<T> const & A)
  {
    return det(A);
  }

  //
  // Dimension
  // get dimension
  //
  template<typename T>
  inline
  Index
  Tensor3<T>::get_dimension() const
  {
    return e.size();
  }

  //
  // R^N Indexing for constant 3rd order tensor
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  const T &
  Tensor3<T>::operator()(
      const Index i,
      const Index j,
      const Index k) const
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    return e[i][j][k];
  }

  //
  // R^N 3rd-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  T &
  Tensor3<T>::operator()(const Index i, const Index j, const Index k)
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    return e[i][j][k];
  }

  //
  // Dimension
  // \return dimension
  //
  template<typename T>
  inline
  Index
  Tensor4<T>::get_dimension() const
  {
    return e.size();
  }

  //
  // R^N indexing for constant 4th order tensor
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  const T &
  Tensor4<T>::operator()(
      const Index i, const Index j, const Index k, const Index l) const
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    assert(l < get_dimension());
    return e[i][j][k][l];
  }

  //
  // R^N 4th-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  T &
  Tensor4<T>::operator()(
      const Index i, const Index j, const Index k, const Index l)
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    assert(l < get_dimension());
    return e[i][j][k][l];
  }

} // namespace LCM

#endif // LCM_Tensor_i_cc
