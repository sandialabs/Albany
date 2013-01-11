//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor3_t_cc)
#define tensor_Tensor3_t_cc

namespace LCM {

  //
  // set dimension
  //
  //
  template<typename T>
  void
  Tensor3<T>::set_dimension(Index const N)
  {
    if (N == dimension) return;

    if (e != NULL) {
      delete [] e;
    }

    Index const
    number_components = N * N * N;

    e = new T[number_components];

    dimension = N;

    return;
  }

  //
  // 3rd-order tensor default constructor
  //
  template<typename T>
  Tensor3<T>::Tensor3() :
    dimension(0),
    e(NULL)
  {
    return;
  }

  //
  // 3rd-order tensor constructor with NaNs
  //
  template<typename T>
  Tensor3<T>::Tensor3(Index const N) :
    dimension(0),
    e(NULL)
  {
    set_dimension(N);

    Index const
    number_components = N * N * N;

    for (Index i = 0; i < number_components; ++i) {
      e[i] = not_a_number<T>();
    }

    return;
  }

  //
  // R^N 3rd-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T>
  Tensor3<T>::Tensor3(Index const N, T const & s) :
    dimension(0),
    e(NULL)
  {
    set_dimension(N);

    Index const
    number_components = N * N * N;

    for (Index i = 0; i < number_components; ++i) {
      e[i] = s;
    }

    return;
  }

  //
  // R^N copy constructor
  // 3rd-order tensor constructor from 3rd-order tensor
  // \param A from which components are copied
  //
  template<typename T>
  Tensor3<T>::Tensor3(Tensor3<T> const & A) :
    dimension(0),
    e(NULL)
  {
    Index const
    N = A.get_dimension();

    set_dimension(N);

    Index const
    number_components = N * N * N;

    for (Index i = 0; i < number_components; ++i) {
      e[i] = A.e[i];
    }

    return;
  }

  //
  // R^N 3rd-order tensor simple destructor
  //
  template<typename T>
  Tensor3<T>::~Tensor3()
  {
    if (e != NULL) {
      delete [] e;
    }
    return;
  }

  //
  // R^N 3rd-order tensor copy assignment
  //
  template<typename T>
  Tensor3<T> &
  Tensor3<T>::operator=(Tensor3<T> const & A)
  {
    if (this != &A) return *this;

    Index const
    N = A.get_dimension();

    set_dimension(N);

    Index const
    number_components = N * N * N;

    for (Index i = 0; i < number_components; ++i) {
      e[i] = A.e[i];
    }

    return *this;
  }

  //
  // 3rd-order tensor increment
  // \param A added to this tensor
  //
  template<typename T>
  Tensor3<T> &
  Tensor3<T>::operator+=(Tensor3<T> const & A)
  {
    Index const
    N = get_dimension();

    assert(A.get_dimension() == N);

    Index const
    number_components = N * N * N;

    for (Index i = 0; i < number_components; ++i) {
      e[i] += A.e[i];
    }

    return *this;
  }

  //
  // 3rd-order tensor decrement
  // \param A substracted from this tensor
  //
  template<typename T>
  Tensor3<T> &
  Tensor3<T>::operator-=(Tensor3<T> const & A)
  {
    Index const
    N = get_dimension();

    assert(A.get_dimension() == N);

    Index const
    number_components = N * N * N;

    for (Index i = 0; i < number_components; ++i) {
      e[i] -= A.e[i];
    }

    return *this;
  }

  //
  // R^N fill 3rd-order tensor with zeros
  //
  template<typename T>
  void
  Tensor3<T>::clear()
  {
    Index const
    N = get_dimension();

    Index const
    number_components = N * N * N;

    for (Index i = 0; i < number_components; ++i) {
      e[i] = 0.0;;
    }

    return;
  }

  //
  // 3rd-order tensor addition
  // \param A 3rd-order tensor
  // \param B 3rd-order tensor
  // \return \f$ A + B \f$
  //
  template<typename T>
  Tensor3<T>
  operator+(Tensor3<T> const & A, Tensor3<T> const & B)
  {
    Index const
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor3<T>
    S(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          S(i,j,k) = A(i,j,k) + B(i,j,k);
        }
      }
    }

    return S;
  }

  //
  // 3rd-order tensor substraction
  // \param A 3rd-order tensor
  // \param B 3rd-order tensor
  // \return \f$ A - B \f$
  //
  template<typename T>
  Tensor3<T>
  operator-(Tensor3<T> const & A, Tensor3<T> const & B)
  {
    Index const
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor3<T>
    S(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          S(i,j,k) = A(i,j,k) - B(i,j,k);
        }
      }
    }

    return S;
  }

  //
  // 3rd-order tensor minus
  // \return \f$ -A \f$
  //
  template<typename T>
  Tensor3<T>
  operator-(Tensor3<T> const & A)
  {
    Index const
    N = A.get_dimension();

    Tensor3<T>
    S(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          S(i,j,k) = - A(i,j,k);
        }
      }
    }

    return S;
  }

  //
  // 3rd-order tensor equality
  // Tested by components
  //
  template<typename T>
  inline bool
  operator==(Tensor3<T> const & A, Tensor3<T> const & B)
  {
    Index const
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          if (A(i,j,k) != B(i,j,k)) {
            return false;
          }
        }
      }
    }

    return true;
  }

  //
  // 3rd-order tensor inequality
  // Tested by components
  //
  template<typename T>
  inline bool
  operator!=(Tensor3<T> const & A, Tensor3<T> const & B)
  {
    return !(A==B);
  }

  //
  // Scalar 3rd-order tensor product
  // \param s scalar
  // \param A 3rd-order tensor
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  Tensor3<T>
  operator*(S const & s, Tensor3<T> const & A)
  {
    Index const
    N = A.get_dimension();

    Tensor3<T>
    B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          B(i,j,k) = s * A(i,j,k);
        }
      }
    }

    return B;
  }

  //
  // 3rd-order tensor scalar product
  // \param A 3rd-order tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  Tensor3<T>
  operator*(Tensor3<T> const & A, S const & s)
  {
    return s * A;
  }

  //
  // 3rd-order tensor vector product
  // \param A 3rd-order tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  Tensor<T>
  dot(Tensor3<T> const & A, Vector<T> const & u)
  {
    Index const
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor<T>
    B(N);

    for (Index j = 0; j < N; ++j) {
      for (Index k = 0; k < N; ++k) {
        T s = 0.0;
        for (Index i = 0; i < N; ++i) {
          s += A(i,j,k) * u(i);
        }
        B(j,k) = s;
      }
    }

    return B;
  }

  //
  // vector 3rd-order tensor product
  // \param A 3rd-order tensor
  // \param u vector
  // \return \f$ u A \f$
  //
  template<typename T>
  Tensor<T>
  dot(Vector<T> const & u, Tensor3<T> const & A)
  {
    Index const
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor<T>
    B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        T s = 0.0;
        for (Index k = 0; k < N; ++k) {
          s += A(i,j,k) * u(k);
        }
        B(i,j) = s;
      }
    }

    return B;
  }


  //
  // 3rd-order tensor vector product2 (contract 2nd index)
  // \param A 3rd-order tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T>
  Tensor<T>
  dot2(Tensor3<T> const & A, Vector<T> const & u)
  {
    Index const
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor<T>
    B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index k = 0; k < N; ++k) {
        T s = 0.0;
        for (Index j = 0; j < N; ++j) {
          s += A(i,j,k) * u(j);
        }
        B(i,k) = s;
      }
    }

    return B;
  }

  //
  // vector 3rd-order tensor product2 (contract 2nd index)
  // \param A 3rd-order tensor
  // \param u vector
  // \return \f$ u A \f$
  //
  template<typename T>
  Tensor<T>
  dot2(Vector<T> const & u, Tensor3<T> const & A)
  {
    return dot2(A, u);
  }

  //
  // 3rd-order tensor input
  // \param A 3rd-order tensor
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor3<T> & A)
  {
    Index const
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          is >> A(i,j,k);
        }
      }
    }

    return is;
  }

  //
  // 3rd-order tensor output
  // \param A 3rd-order tensor
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<T> const & A)
  {
    Index const
    N = A.get_dimension();

    if (N == 0) {
      return os;
    }

    for (Index i = 0; i < N; ++i) {

      for (Index j = 0; j < N; ++j) {

        os << std::scientific << A(i,j,0);

        for (Index k = 1; k < N; ++k) {
          os << std::scientific << "," << A(i,j,k);
        }

        os << std::endl;

      }

      os << std::endl;
      os << std::endl;

    }

    return os;
  }

} // namespace LCM

#endif // tensor_Tensor3_t_cc
