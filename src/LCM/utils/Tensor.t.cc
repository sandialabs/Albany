///
/// \file Tensor.t.cc
/// First cut of LCM small tensor utilities. Templates.
/// \author Alejandro Mota
/// \author Jake Ostien
///
#if !defined(LCM_Tensor_t_cc)
#define LCM_Tensor_t_cc

namespace LCM {

  //
  // set dimension
  //
  //
  template<typename T>
  void
  Tensor3<T>::set_dimension(const Index N)
  {

    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize(N);
      }
    }

    return;
  }

  //
  // 3rd-order tensor default constructor
  //
  template<typename T>
  Tensor3<T>::Tensor3()
  {
    return;
  }

  //
  // 3rd-order tensor constructor with NaNs
  //
  template<typename T>
  Tensor3<T>::Tensor3(const Index N)
  {
    set_dimension(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = not_a_number<T>();
        }
      }
    }

    return;
  }

  //
  // R^N 3rd-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T>
  Tensor3<T>::Tensor3(const Index N, T const & s)
  {
    set_dimension(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = s;
        }
      }
    }

    return;
  }

  //
  // R^N copy constructor
  // 3rd-order tensor constructor from 3rd-order tensor
  // \param A from which components are copied
  //
  template<typename T>
  Tensor3<T>::Tensor3(Tensor3<T> const & A)
  {
    const Index
    N = A.get_dimension();

    set_dimension(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = A.e[i][j][k];
        }
      }
    }

    return;
  }

  //
  // R^N 3rd-order tensor simple destructor
  //
  template<typename T>
  Tensor3<T>::~Tensor3()
  {
    return;
  }

  //
  // R^N 3rd-order tensor copy assignment
  //
  template<typename T>
  Tensor3<T> &
  Tensor3<T>::operator=(Tensor3<T> const & A)
  {
    if (this != &A) {

      const Index
      N = A.get_dimension();

      set_dimension(N);

      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          for (Index k = 0; k < N; ++k) {
            e[i][j][k] = A.e[i][j][k];
          }
        }
      }

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
    const Index
    N = get_dimension();

    assert(A.get_dimension() == N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] += A.e[i][j][k];
        }
      }
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
    const Index
    N = get_dimension();

    assert(A.get_dimension() == N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] -= A.e[i][j][k];
        }
      }
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
    const Index
    N = get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = 0.0;
        }
      }
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
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor3<T> S(N);

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
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor3<T> S(N);

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
    const Index
    N = A.get_dimension();

    Tensor3<T> S(N);

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
    const Index
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
    const Index
    N = A.get_dimension();

    Tensor3<T> B(N);

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
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor<T> B(N);

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
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor<T> B(N);

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
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor<T> B(N);

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
  // set dimension
  //
  //
  template<typename T>
  void
  Tensor4<T>::set_dimension(const Index N)
  {

    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize(N);
        for (Index k = 0; k < N; ++k) {
          e[i][j][k].resize(N);
        }
      }
    }

    return;
  }

  //
  // R^N 4th-order tensor default constructor
  //
  template<typename T>
  Tensor4<T>::Tensor4()
  {
    return;
  }

  //
  // R^N 4th-order tensor constructor with NaNs
  //
  template<typename T>
  Tensor4<T>::Tensor4(const Index N)
  {
    set_dimension(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = not_a_number<T>();
          }
        }
      }
    }

    return;
  }

  //
  // R^N 4th-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T>
  Tensor4<T>::Tensor4(const Index N, T const & s)
  {
    set_dimension(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = s;
          }
        }
      }
    }

    return;
  }

  //
  // R^N copy constructor
  // 4th-order tensor constructor with 4th-order tensor
  // \param A from which components are copied
  //
  template<typename T>
  Tensor4<T>::Tensor4(Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    set_dimension(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = A.e[i][j][k][l];
          }
        }
      }
    }

    return;
  }

  //
  // R^N 4th-order tensor simple destructor
  //
  template<typename T>
  Tensor4<T>::~Tensor4()
  {
    return;
  }

  //
  // R^N 4th-order tensor copy assignment
  //
  template<typename T>
  Tensor4<T> &
  Tensor4<T>::operator=(Tensor4<T> const & A)
  {
    if (this != &A) {
      const Index
      N = A.get_dimension();

      set_dimension(N);

      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          for (Index k = 0; k < N; ++k) {
            for (Index l = 0; l < N; ++l) {
              e[i][j][k][l] = A.e[i][j][k][l];
            }
          }
        }
      }
    }

    return *this;
  }

  //
  // 4th-order tensor increment
  // \param A added to this tensor
  //
  template<typename T>
  Tensor4<T> &
  Tensor4<T>::operator+=(Tensor4<T> const & A)
  {
    const Index
    N = get_dimension();

    assert(A.get_dimension() == N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] += A.e[i][j][k][l];
          }
        }
      }
    }

    return *this;
  }

  //
  // 4th-order tensor decrement
  // \param A substracted from this tensor
  //
  template<typename T>
  Tensor4<T> &
  Tensor4<T>::operator-=(Tensor4<T> const & A)
  {
    const Index
    N = get_dimension();

    assert(A.get_dimension() == N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] -= A.e[i][j][k][l];
          }
        }
      }
    }

    return *this;
  }

  //
  // R^N fill 4th-order tensor with zeros
  //
  template<typename T>
  void
  Tensor4<T>::clear()
  {
    const Index
    N = get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = 0.0;
          }
        }
      }
    }

    return;
  }

  //
  // 4th-order tensor addition
  // \param A 4th-order tensor
  // \param B 4th-order tensor
  // \return \f$ A + B \f$
  //
  template<typename T>
  Tensor4<T>
  operator+(Tensor4<T> const & A, Tensor4<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor4<T> S(N);


    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            S(i,j,k,l) = A(i,j,k,l) + B(i,j,k,l);
          }
        }
      }
    }

    return S;
  }

  //
  // 4th-order tensor substraction
  // \param A 4th-order tensor
  // \param B 4th-order tensor
  // \return \f$ A - B \f$
  //
  template<typename T>
  Tensor4<T>
  operator-(Tensor4<T> const & A, Tensor4<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor4<T> S(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            S(i,j,k,l) = A(i,j,k,l) - B(i,j,k,l);
          }
        }
      }
    }

    return S;
  }

  //
  // 4th-order tensor minus
  // \return \f$ -A \f$
  //
  template<typename T>
  Tensor4<T>
  operator-(Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor4<T> S(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            S(i,j,k,l) = - A(i,j,k,l);
          }
        }
      }
    }

    return S;
  }

  //
  // 4th-order equality
  // Tested by components
  //
  template<typename T>
  inline bool
  operator==(Tensor4<T> const & A, Tensor4<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            if (A(i,j,k,l) != B(i,j,k,l)) {
              return false;
            }
          }
        }
      }
    }

    return true;
  }

  //
  // 4th-order inequality
  // Tested by components
  //
  template<typename T>
  inline bool
  operator!=(Tensor4<T> const & A, Tensor4<T> const & B)
  {
    return !(A==B);
  }

  //
  // Scalar 4th-order tensor product
  // \param s scalar
  // \param A 4th-order tensor
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  Tensor4<T>
  operator*(S const & s, Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor4<T> B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            B(i,j,k,l) = s * A(i,j,k,l);
          }
        }
      }
    }

    return B;
  }

  //
  // 4th-order tensor scalar product
  // \param A 4th-order tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename T, typename S>
  Tensor4<T>
  operator*(Tensor4<T> const & A, S const & s)
  {
    return s * A;
  }

  //
  // R^N 2nd-order tensor inverse
  // Gauss-Jordan elimination. Warning: full pivoting
  // for small tensors. Use Teuchos LAPACK interface for
  // more efficient and robust techniques.
  // \param A nonsingular tensor
  // \return \f$ A^{-1} \f$
  //
  template<typename T>
  Tensor<T>
  inverse(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T> S = A;
    Tensor<T> B = identity<T>(N);

    typedef std::set<Index> IndexSet;
    typedef std::set<Index>::const_iterator IndexIter;

    IndexSet intact_rows;
    IndexSet intact_cols;

    for (Index k = 0; k < N; ++k) {
      intact_rows.insert(k);
      intact_cols.insert(k);
    }

    // Gauss-Jordan elimination with full pivoting
    for (Index k = 0; k < N; ++k) {

      // Determine full pivot
      T pivot = 0.0;

      IndexIter pivot_row_iter = intact_rows.begin();
      IndexIter pivot_col_iter = intact_cols.begin();

      for (IndexIter rows_iter = intact_rows.begin();
          rows_iter != intact_rows.end(); ++rows_iter) {

        for (IndexIter cols_iter = intact_cols.begin();
            cols_iter != intact_cols.end(); ++cols_iter) {

          const Index row = *rows_iter;
          const Index col = *cols_iter;
          const T s = fabs(S(row, col));

          if (s > pivot) {

            pivot_row_iter = rows_iter;
            pivot_col_iter = cols_iter;

            pivot = s;

          }

        }

      }

      const Index pivot_row = *pivot_row_iter;
      const Index pivot_col = *pivot_col_iter;

      // Gauss-Jordan elimination
      const T t = S(pivot_row, pivot_col);

      if (t == 0.0) {
        std::cerr << "ERROR: Inverse of singular tensor." << std::endl;
        exit(1);
      }

      for (Index j = 0; j < N; ++j) {
        S(pivot_row, j) /= t;
        B(pivot_row, j) /= t;
      }

      for (Index i = 0; i < N; ++i) {
        if (i == pivot_row) continue;

        const T c = S(i, pivot_col);

        for (Index j = 0; j < N; ++j) {
          S(i, j) -= c * S(pivot_row, j);
          B(i, j) -= c * B(pivot_row, j);
        }
      }

      // Eliminate current row and col from intact rows and cols
      intact_rows.erase(pivot_row_iter);
      intact_cols.erase(pivot_col_iter);

    }

    Tensor<T>
    X = transpose(S) * B;

    return X;
  }

  //
  // R^N Subtensor
  // \param A tensor
  // \param i index
  // \param j index
  // \return Subtensor with i-row and j-col deleted.
  //
  template<typename T>
  Tensor<T>
  subtensor(Tensor<T> const & A, Index i, Index j)
  {
    const Index
    N = A.get_dimension();

    assert(i < N);
    assert(j < N);

    Tensor<T> B(N - 1);

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
  // R^N exponential map by Taylor series, radius of convergence is infinity
  // \param A tensor
  // \return \f$ \exp A \f$
  //
  template<typename T>
  Tensor<T>
  exp(Tensor<T> const & A)
  {
    const Index
    max_iter = 128;

    const T
    tol = machine_epsilon<T>();

    const Index
    N = A.get_dimension();

    Tensor<T>
    term = identity<T>(N);

    // Relative error taken wrt to the first term, which is I and norm = 1
    T
    relative_error = 1.0;

    Tensor<T>
    B = term;

    Index
    k = 0;

    while (relative_error > tol && k < max_iter) {
      term = T(1.0 / (k + 1.0)) * term * A;
      B = B + term;
      relative_error = norm_1(term);
      ++k;
    }

    return B;
  }

  //
  // R^N logarithmic map by Taylor series, converges for \f$ |A-I| < 1 \f$
  // \param A tensor
  // \return \f$ \log A \f$
  //
  template<typename T>
  Tensor<T>
  log(Tensor<T> const & A)
  {
    // Check whether skew-symmetric holds

    const Index
    max_iter = 128;

    const T
    tol = machine_epsilon<T>();

    const T
    norm_arg = norm_1(A);

    const Index
    N = A.get_dimension();

    const Tensor<T>
    Am1 = A - identity<T>(N);

    Tensor<T>
    term = Am1;

    T
    norm_term = norm_1(term);

    T
    relative_error = norm_term / norm_arg;

    Tensor<T>
    B = term;

    Index
    k = 1;

    while (relative_error > tol && k <= max_iter) {
      term = T(- (k / (k + 1.0))) * term * Am1;
      B = B + term;
      norm_term = norm_1(term);
      relative_error = norm_term / norm_arg;
      ++k;
    }

    return B;
  }

  //
  // R^N logarithmic map of a rotation. Not implemented yet.
  // \param R with \f$ R \in SO(N) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(N) \f$
  //
  template<typename T>
  Tensor<T>
  log_rotation(Tensor<T> const & R)
  {
    const Index
    N = R.get_dimension();

    //firewalls, make sure R \in SO(N)
    assert(norm(R*transpose(R) - eye<T>(N)) < 100.0 * machine_epsilon<T>());
    assert(fabs(det(R) - 1.0) < 100.0 * machine_epsilon<T>());

    // acos requires input between -1 and +1
    T
    cosine = 0.5*(trace(R) - 1.0);

    if (cosine < -1.0) {
      cosine = -1.0;
    } else if(cosine > 1.0) {
      cosine = 1.0;
    }

    T
    theta = acos(cosine);

    Tensor<T>
    r(N);

    switch (N) {

    default:
      std::cerr << "Logarithm of SO(N) N != 2,3 not implemented." << std::endl;
      exit(1);
      break;

    case 3:
      if (theta == 0) {
        r = zero<T>(3);
      } else if (fabs(cosine + 1.0) < 10.0*machine_epsilon<T>())  {
        // Rotation angle is PI.
        r = log_rotation_pi(R);
      } else {
        r = T(theta/(2.0*sin(theta)))*(R - transpose(R));
      }
      break;

    case 2:
      r(0,0) = 0.0;
      r(0,1) = -theta;
      r(1,0) = theta;
      r(1,1) = 0.0;
      break;

    }

    return r;
  }

  // R^N Logarithmic map of a 180-degree rotation. Not implemented.
  // \param R with \f$ R \in SO(N) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(N) \f$
  //
  template<typename T>
  Tensor<T>
  log_rotation_pi(Tensor<T> const & R)
  {
    const Index
    N = R.get_dimension();

    T
    cosine = 0.5*(trace(R) - 1.0);
    // set firewall to make sure the rotation is indeed 180 degrees
    assert(fabs(cosine + 1.0) < 10.0 * machine_epsilon<T>());

    Tensor<T>
    r(N);

    switch (N) {

    default:
      std::cerr << "Logarithm of SO(N) N != 2,3 not implemented." << std::endl;
      exit(1);
      break;

    case 3:
      {
        // obtain U from R = LU
        r = gaussian_elimination((R - identity<T>(3)));

        // backward substitution (for rotation exp(R) only)
        const T tol = 10.0 * machine_epsilon<T>();

        Vector<T> normal(3);

        if (fabs(r(2,2)) < tol){
          normal(2) = 1.0;
        } else {
          normal(2) = 0.0;
        }

        if (fabs(r(1,1)) < tol){
          normal(1) = 1.0;
        } else {
          normal(1) = -normal(2)*r(1,2)/r(1,1);
        }

        if (fabs(r(0,0)) < tol){
          normal(0) = 1.0;
        } else {
          normal(0) = -normal(1)*r(0,1) - normal(2)*r(0,2)/r(0,0);
        }

        normal = normal / norm(normal);

        r.clear();
        r(0,1) = -normal(2);
        r(0,2) =  normal(1);
        r(1,0) =  normal(2);
        r(1,2) = -normal(0);
        r(2,0) = -normal(1);
        r(2,1) =  normal(0);

        const T pi = acos(-1.0);

        r = pi * r;
      }
      break;

    case 2:
      {
        T theta = acos(-1.0);

        if (R(0,0) > 0.0) {
          theta = - theta;
        }

        r(0,0) = 0.0;
        r(0,1) = -theta;
        r(1,0) = theta;
        r(1,1) = 0.0;
      }
      break;

    }

    return r;
  }

  // Gaussian Elimination with partial pivot
  // \param matrix \f$ A \f$
  // \return \f$ U \f$ where \f$ A = LU \f$
  //
  template<typename T>
  Tensor<T>
  gaussian_elimination(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T>
    U = A;

    const T
    tol = 10.0 * machine_epsilon<T>();

    Index i = 0;
    Index j = 0;
    Index i_max = 0;

    while ((i <  N) && (j < N)) {
      // find pivot in column j, starting in row i
      i_max = i;
      for (Index k = i + 1; k < N; ++k) {
        if (fabs(U(k,j) > fabs(U(i_max,j)))) {
          i_max = k;
        }
      }

      // Check if A(i_max,j) equal to or very close to 0
      if (fabs(U(i_max,j)) > tol){
        // swap rows i and i_max and divide each entry in row i
        // by U(i,j)
        for (Index k = 0; k < N; ++k) {
          std::swap(U(i,k), U(i_max,k));
        }

        for (Index k = 0; k < N; ++k) {
          U(i,k) = U(i,k) / U(i,j);
        }

        for (Index l = i + 1; l < N; ++l) {
          for (Index k = 0; k < N; ++k) {
            U(l,k) = U(l,k) - U(l,i) * U(i,k) / U(i,i);
          }
        }
        ++i;
      }
      ++j;
    }

    return U;
  }

  // Apply Givens-Jacobi rotation on the left in place.
  // \param c and s for a rotation G in form [c, s; -s, c]
  // \param A
  //
  template<typename T>
  void
  givens_left(T const & c, T const & s, Index i, Index k, Tensor<T> & A)
  {
    const Index
    N = A.get_dimension();

    for (Index j = 0; j < N; ++j) {
      T const t1 = A(i,j);
      T const t2 = A(k,j);
      A(i,j) = c * t1 - s * t2;
      A(k,j) = s * t1 + c * t2;
    }
    return;
  }

  // Apply Givens-Jacobi rotation on the right in place.
  // \param A
  /// \param c and s for a rotation G in form [c, s; -s, c]
  //
  template<typename T>
  void
  givens_right(T const & c, T const & s, Index i, Index k, Tensor<T> & A)
  {
    const Index
    N = A.get_dimension();

    for (Index j = 0; j < N; ++j) {
      T const t1 = A(j,i);
      T const t2 = A(j,k);
      A(j,i) = c * t1 - s * t2;
      A(j,k) = s * t1 + c * t2;
    }
    return;
  }

  //
  // R^N exponential map of a skew-symmetric tensor. Not implemented.
  // \param r \f$ r \in so(N) \f$
  // \return \f$ R = \exp R \f$ with \f$ R \in SO(N) \f$
  //
  template<typename T>
  Tensor<T>
  exp_skew_symmetric(Tensor<T> const & r)
  {
    // Check whether skew-symmetry holds
    assert(norm(r+transpose(r)) < machine_epsilon<T>());

    const Index
    N = r.get_dimension();

    Tensor<T>
    R = identity<T>(N);

    T
    theta = 0.0;

    switch (N) {

    default:
      std::cerr << "Exp of so(N) N != 2,3 not implemented." << std::endl;
      exit(1);
      break;

    case 3:
      theta = sqrt(r(2,1)*r(2,1)+r(0,2)*r(0,2)+r(1,0)*r(1,0));

      //Check whether norm == 0. If so, return identity.
      if (theta >= machine_epsilon<T>()) {
        R += sin(theta)/theta*r + (1.0-cos(theta))/(theta*theta)*r*r;
      }
      break;

    case 2:
      theta = r(1,0);

      T
      c = cos(theta);

      T
      s = sin(theta);

      R(0,0) = c;
      R(0,1) = -s;
      R(1,0) = s;
      R(1,1) = c;

      break;

    }

    return R;
  }

  //
  // R^N off-diagonal norm. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ \sqrt(\sum_i \sum_{j, j\neq i} a_{ij}^2) \f$
  //
  template<typename T>
  T
  norm_off_diagonal(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    T s = 0.0;

    switch (N) {

    default:
      for (Index i = 0; i < N; ++i) {
        for (Index j = 0; j < N; ++j) {
          if (i != j) s += A(i,j)*A(i,j);
        }
      }
      break;

    case 3:
      s = A(0,1)*A(0,1) + A(0,2)*A(0,2) + A(1,2)*A(1,2) +
          A(1,0)*A(1,0) + A(2,0)*A(2,0) + A(2,1)*A(2,1);
      break;

    case 2:
      s = A(0,1)*A(0,1) + A(1,0)*A(1,0);
      break;

    }

    return sqrt(s);
  }

  //
  // R^N arg max abs. Useful for inverse and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ (p,q) = arg max_{i,j} |a_{ij}| \f$
  //
  template<typename T>
  std::pair<Index, Index>
  arg_max_abs(Tensor<T> const & A)
  {
    Index p = 0;
    Index q = 0;

    T s = fabs(A(p,q));

    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        if (fabs(A(i,j)) > s) {
          p = i;
          q = j;
          s = fabs(A(i,j));
        }
      }
    }

    return std::make_pair(p,q);
  }

  //
  // R^N arg max off-diagonal. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ (p,q) = arg max_{i \neq j} |a_{ij}| \f$
  //
  template<typename T>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T> const & A)
  {
    Index p = 0;
    Index q = 1;

    T s = fabs(A(p,q));

    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        if (i != j && fabs(A(i,j)) > s) {
          p = i;
          q = j;
          s = fabs(A(i,j));
        }
      }
    }

    return std::make_pair(p,q);
  }

  namespace {

    //
    // Singular value decomposition (SVD) for 2x2
    // bidiagonal matrix. Used for general 2x2 SVD.
    // Adapted from LAPAPCK's DLASV2, Netlib's dlasv2.c
    // and LBNL computational crystalography toolbox
    // \param f, g, h where A = [f, g; 0, h]
    // \return \f$ A = USV^T\f$
    //
    template<typename T>
    boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
    svd_bidiagonal(T f, T g, T h)
    {
      T fa = std::abs(f);
      T ga = std::abs(g);
      T ha = std::abs(h);

      T s0 = 0.0;
      T s1 = 0.0;

      T cu = 1.0;
      T su = 0.0;
      T cv = 1.0;
      T sv = 0.0;

      bool swap_diag = (ha > fa);

      if (swap_diag) {
        std::swap(fa, ha);
        std::swap(f, h);
      }

      if (ga == 0.0) {
        s1 = ha;
        s0 = fa;
      } else if (ga > fa && fa / ga < machine_epsilon<T>()) {
        // case of very large ga
        s0 = ga;
        s1 = ha > 1.0 ?
            T(fa / (ga / ha)) :
            T((fa / ga) * ha);
        cu = 1.0;
        su = h / g;
        cv = f / g;
        sv = 1.0;
      } else {
        // normal case
        T d = fa - ha;
        T l = d != fa ?
            T(d / fa) :
            T(1.0); // l \in [0,1]
        T m = g / f; // m \in (-1/macheps, 1/macheps)
        T t = 2.0 - l; // t \in [1,2]
        T mm = m * m;
        T tt = t * t;
        T s = sqrt(tt + mm); // s \in [1,1 + 1/macheps]
        T r = l != 0.0 ?
            T(sqrt(l * l + mm)) :
            T(fabs(m)); // r \in [0,1 + 1/macheps]
        T a = 0.5 * (s + r); // a \in [1,1 + |m|]
        s1 = ha / a;
        s0 = fa * a;

        // Compute singular vectors
        T tau; // second assignment to T in DLASV2
        if (mm != 0.0) {
          tau = (m / (s + t) + m / (r + l)) * (1.0 + a);
        } else {
          // note that m is very tiny
          tau = l == 0.0 ?
              T(copysign(T(2.0), f) * copysign(T(1.0), g)) :
              T(g / copysign(d, f) + m / t);
        }
        T lv = sqrt(tau * tau + 4.0); // second assignment to L in DLASV2
        cv = 2.0 / lv;
        sv = tau / lv;
        cu = (cv + sv * m) / a;
        su = (h / f) * sv / a;
      }

      // Fix signs of singular values in accordance to sign of singular vectors
      s0 = copysign(s0, f);
      s1 = copysign(s1, h);

      if (swap_diag) {
        std::swap(cu, sv);
        std::swap(su, cv);
      }

      Tensor<T> U(cu, -su, su, cu);

      Tensor<T> S(s0, 0.0, 0.0, s1);

      Tensor<T> V(cv, -sv, sv, cv);

      return boost::make_tuple(U, S, V);
    }

    //
    // R^2 singular value decomposition (SVD)
    // \param A tensor
    // \return \f$ A = USV^T\f$
    //
    template<typename T>
    boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
    svd_2x2(Tensor<T> const & A)
    {
      assert(A.get_dimension() == 2);

      // First compute a givens rotation to eliminate 1,0 entry in tensor
      T c = 1.0;
      T s = 0.0;
      boost::tie(c, s) = givens(A(0,0), A(1,0));

      Tensor<T>
      R(c, -s, s, c);

      Tensor<T>
      B = R * A;

      // B is bidiagonal. Use specialized algorithm to compute its SVD
      Tensor<T>
      X(2), S(2), V(2);

      boost::tie(X, S, V) = svd_bidiagonal(B(0,0), B(0,1), B(1,1));

      // Complete general 2x2 SVD with givens rotation calculated above
      Tensor<T>
      U = transpose(R) * X;

      return boost::make_tuple(U, S, V);
    }

    //
    // R^N singular value decomposition (SVD)
    // \param A tensor
    // \return \f$ A = USV^T\f$
    //
    template<typename T>
    boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
    svd_NxN(Tensor<T> const & A)
    {
      Tensor<T>
      S = A;

      const Index
      N = A.get_dimension();

      Tensor<T>
      U = identity<T>(N);

      Tensor<T>
      V = identity<T>(N);

      T
      off = norm_off_diagonal(S);

      const T
      tol = machine_epsilon<T>() * norm(A);

      const Index
      max_iter = 128;

      Index
      num_iter = 0;

      while (off > tol && num_iter < max_iter) {

        // Find largest off-diagonal entry
        Index
        p = 0;

        Index
        q = 0;

        boost::tie(p,q) = arg_max_off_diagonal(S);

        if (p > q) {
          std::swap(p, q);
        }

        // Obtain left and right Givens rotations by using 2x2 SVD
        Tensor <T>
        Spq(S(p,p), S(p,q), S(q,p), S(q,q));

        Tensor <T>
        L(2), D(2), R(2);

        boost::tie(L, D, R) = svd_2x2(Spq);

        T const &
        cl = L(0,0);

        T const &
        sl = L(0,1);

        T const &
        cr = R(0,0);

        T const &
        sr = (sgn(R(0,1)) == sgn(R(1,0))) ? T(-R(0,1)) : T(R(0,1));

        // Apply both Givens rotations to matrices
        // that are converging to singular values and singular vectors
        givens_left(cl, sl, p, q, S);
        givens_right(cr, sr, p, q, S);

        givens_right(cl, sl, p, q, U);
        givens_left(cr, sr, p, q, V);

        off = norm_off_diagonal(S);
        num_iter++;
      }

      if (num_iter == max_iter) {
        std::cerr << "WARNING: SVD iteration did not converge." << std::endl;
      }

      return boost::make_tuple(U, diag(diag(S)), transpose(V));
    }

  } // anonymous namespace

  //
  // R^N singular value decomposition (SVD)
  // \param A tensor
  // \return \f$ A = USV^T\f$
  //
  template<typename T>
  boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
  svd(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T>
    U(N), S(N), V(N);

    switch (N) {

    default:
      boost::tie(U, S, V) = svd_NxN(A);
      break;

    case 2:
      boost::tie(U, S, V) = svd_2x2(A);
      break;

    }

    return boost::make_tuple(U, S, V);
  }

  //
  // Project to O(N) (Orthogonal Group) using a Newton-type algorithm.
  // See Higham's Functions of Matrices p210 [2008]
  // \param A tensor (often a deformation-gradient-like tensor)
  // \return \f$ R = \argmin_Q \|A - Q\|\f$
  // This algorithm projects a given tensor in GL(N) to O(N).
  // The rotation/reflection obtained through this projection is
  // the orthogonal component of the real polar decomposition
  //
  template<typename T>
  Tensor<T>
  polar_rotation(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    bool
    scale = true;

    const T
    tol_scale = 0.01;

    const T
    tol_conv = sqrt(N) * machine_epsilon<T>();

    Tensor<T>
    X = A;

    T
    gamma = 2.0;

    const Index
    max_iter = 128;

    Index
    num_iter = 0;

    while (num_iter < max_iter) {

      Tensor<T>
      Y = inverse(X);

      T
      mu = 1.0;

      if (scale == true) {
        mu = (norm_1(Y) * norm_infinity(Y)) / (norm_1(X) * norm_infinity(X));
        mu = sqrt(sqrt(mu));
      }

      Tensor<T>
      Z = 0.5 * (mu * X + transpose(Y) / mu);

      Tensor<T>
      D = Z - X;

      T
      delta = norm(D) / norm(Z);

      if (scale == true && delta < tol_scale) {
        scale = false;
      }

      bool
      end_iter =
          norm(D) <= sqrt(tol_conv) ||
          (delta > 0.5 * gamma && scale == false);

      X = Z;
      gamma = delta;

      if (end_iter == true) {
        break;
      }

      num_iter++;

    }

    if (num_iter == max_iter) {
      std::cerr << "WARNING: Polar iteration did not converge." << std::endl;
    }

    return X;
  }

  //
  // R^N Left polar decomposition
  // \param A tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = A \f$ with \f$ R \in SO(N) \f$ and \f$ V \in SPD(N) \f$
  //
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_left(Tensor<T> const & A)
  {
    Tensor<T>
    R = polar_rotation(A);

    Tensor<T>
    V = symm(A * transpose(R));

    return std::make_pair(V, R);
  }

  //
  // R^N Right polar decomposition
  // \param A tensor (often a deformation-gradient-like tensor)
  // \return \f$ RU = A \f$ with \f$ R \in SO(N) \f$ and \f$ U \in SPD(N) \f$
  //
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_right(Tensor<T> const & A)
  {
    Tensor<T>
    R = polar_rotation(A);

    Tensor<T>
    U = symm(transpose(R) * A);

    return std::make_pair(R, U);
  }

  //
  // R^3 left polar decomposition with eigenvalue decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD(3)
  //
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_left_eig(Tensor<T> const & F)
  {
    assert(F.get_dimension() == 3);

    // set up return tensors
    Tensor<T>
    R(3);

    Tensor<T>
    V(3);

    // temporary tensor used to compute R
    Tensor<T>
    Vinv(3);

    // compute spd tensor
    Tensor<T>
    b = F * transpose(F);

    // get eigenvalues/eigenvectors
    Tensor<T>
    eVal(3);

    Tensor<T>
    eVec(3);

    boost::tie(eVec, eVal) = eig_spd(b);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<T>
    x = zero<T>(3);

    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));

    Tensor<T>
    xi = zero<T>(3);

    xi(0,0) = 1.0 / x(0,0);
    xi(1,1) = 1.0 / x(1,1);
    xi(2,2) = 1.0 / x(2,2);

    // compute V, Vinv, and R
    V    = eVec * x * transpose(eVec);
    Vinv = eVec * xi * transpose(eVec);
    R    = Vinv * F;

    return std::make_pair(V, R);
  }

  //
  // R^3 right polar decomposition with eigenvalue decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ RU = F \f$ with \f$ R \in SO(3) \f$ and U SPD(3)
  //
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  polar_right_eig(Tensor<T> const & F)
  {
    assert(F.get_dimension() == 3);

    Tensor<T>
    R(3);

    Tensor<T>
    U(3);

    // temporary tensor used to compute R
    Tensor<T>
    Uinv(3);

    // compute spd tensor
    Tensor<T>
    C = transpose(F) * F;

    // get eigenvalues/eigenvectors
    Tensor<T>
    eVal(3);

    Tensor<T>
    eVec(3);

    boost::tie(eVec, eVal) = eig_spd(C);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<T>
    x = zero<T>(3);

    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));

    Tensor<T>
    xi = zero<T>(3);

    xi(0,0) = 1.0 / x(0,0);
    xi(1,1) = 1.0 / x(1,1);
    xi(2,2) = 1.0 / x(2,2);

    // compute U, Uinv, and R
    U    = eVec * x * transpose(eVec);
    Uinv = eVec * xi * transpose(eVec);
    R    = F * Uinv;

    return std::make_pair(R, U);
  }

  //
  // R^N left polar decomposition with matrix logarithm for V
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(N) \f$ and V SPD(N), and log V
  //
  template<typename T>
  boost::tuple<Tensor<T>, Tensor<T>, Tensor<T> >
  polar_left_logV(Tensor<T> const & F)
  {
    const Index
    N = F.get_dimension();

    Tensor<T>
    X(N), S(N), Y(N);

    boost::tie(X, S, Y) = svd(F);

    Tensor<T>
    R = X * transpose(Y);

    Tensor<T>
    V = X * S * transpose(X);

    Tensor<T>
    s = S;

    for (Index i = 0; i < N; ++i) {
      s(i,i) = std::log(s(i,i));
    }

    Tensor<T>
    v = X * s * transpose(X);

    return boost::make_tuple(V, R, v);
  }

  //
  // R^N logarithmic map using BCH expansion (3 terms)
  // \param x tensor
  // \param y tensor
  // \return Baker-Campbell-Hausdorff series up to 3 terms
  //
  template<typename T>
  Tensor<T>
  bch(Tensor<T> const & x, Tensor<T> const & y)
  {
    return
        // first order term
        x + y
        +
        // second order term
        T(0.5)*(x*y - y*x)
        +
        // third order term
        T(1.0/12.0) *
          (x*x*y - T(2.0)*x*y*x + x*y*y + y*x*x - T(2.0)*y*x*y + y*y*x);
  }

  //
  // Symmetric Schur algorithm for R^2.
  // \param \f$ A = [f, g; g, h] \in S(2) \f$
  // \return \f$ c, s \rightarrow [c, -s; s, c]\f diagonalizes A$
  //
  template<typename T>
  std::pair<T, T>
  schur_sym(const T f, const T g, const T h)
  {
    T c = 1.0;
    T s = 0.0;

    if (g != 0.0) {
      T t = (h - f) / (2.0 * g);

      if (t >= 0.0) {
        t = 1.0 / (sqrt(1.0 + t * t) + t);
      } else {
        t = -1.0 / (sqrt(1.0 + t * t) - t);
      }
      c = 1.0 / sqrt(1.0 + t * t);
      s = t * c;
    }

    return std::make_pair(c, s);
  }

  //
  // Givens rotation. [c, -s; s, c] [a; b] = [r; 0]
  // \param a, b
  // \return c, s
  //
  template<typename T>
  std::pair<T, T>
  givens(T const & a, T const & b)
  {
    T c = 1.0;
    T s = 0.0;

    if (b != 0.0) {
      if (fabs(b) > fabs(a)) {
        const T t = - a / b;
        s = 1.0 / sqrt(1.0 + t * t);
        c = t * s;
      } else {
        const T t = - b / a;
        c = 1.0 / sqrt(1.0 + t * t);
        s = t * c;
      }
    }

    return std::make_pair(c, s);
  }

  namespace {

    //
    // R^N eigenvalue decomposition for symmetric 2nd-order tensor
    // \param A tensor
    // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
    //
    template<typename T>
    std::pair<Tensor<T>, Tensor<T> >
    eig_sym_NxN(Tensor<T> const & A)
    {
      Tensor<T>
      D = symm(A);

      const Index
      N = A.get_dimension();

      Tensor<T>
      V = identity<T>(N);

      T
      off = norm_off_diagonal(D);

      const T
      tol = machine_epsilon<T>() * norm(A);

      const Index
      max_iter = 128;

      Index
      num_iter = 0;

      while (off > tol && num_iter < max_iter) {

        // Find largest off-diagonal entry
        Index
        p = 0;

        Index
        q = 0;

        boost::tie(p,q) = arg_max_off_diagonal(D);
        if (p > q) {
          std::swap(p,q);
        }

        // Obtain Givens rotations by using 2x2 symmetric Schur algorithm
        T const &
        f = D(p,p);

        T const &
        g = D(p,q);

        T const &
        h = D(q,q);

        T
        c, s;

        boost::tie(c, s) = schur_sym(f, g, h);

        // Apply Givens rotation to matrices
        // that are converging to eigenvalues and eigenvectors
        givens_left(c, s, p, q, D);
        givens_right(c, s, p, q, D);

        givens_right(c, s, p, q, V);

        off = norm_off_diagonal(D);
        num_iter++;
      }

      if (num_iter == max_iter) {
        std::cerr << "WARNING: EIG iteration did not converge." << std::endl;
      }

      return std::make_pair(V, diag(diag(D)));
    }

    //
    // R^2 eigenvalue decomposition for symmetric 2nd-order tensor
    // \param A tensor
    // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
    //
    template<typename T>
    std::pair<Tensor<T>, Tensor<T> >
    eig_sym_2x2(Tensor<T> const & A)
    {
      assert(A.get_dimension() == 2);

      const T f = A(0,0);
      const T g = 0.5 * (A(0,1) + A(1,0));
      const T h = A(1,1);

      //
      // Eigenvalues, based on LAPACK's dlae2
      //
      const T sum = f + h;
      const T dif = fabs(f - h);
      const T g2 = fabs(g + g);

      T fhmax = f;
      T fhmin = h;

      const bool swap_diag = fabs(h) > fabs(f);

      if (swap_diag == true) {
        std::swap(fhmax, fhmin);
      }

      T r = 0.0;
      if (dif > g2) {
        const T t = g2 / dif;
        r = dif * sqrt(1.0 + t * t);
      } else if (dif < g2) {
        const T t = dif / g2;
        r = g2 * sqrt(1.0 + t * t);
      } else {
        // dif == g2, including zero
        r = g2 * sqrt(2.0);
      }

      T s0 = 0.0;
      T s1 = 0.0;

      if (sum != 0.0) {
        s0 = 0.5 * (sum + copysign(r, sum));
        // Order of execution important.
        // To get fully accurate smaller eigenvalue,
        // next line needs to be executed in higher precision.
        s1 = (fhmax / s0) * fhmin - (g / s0) * g;
      } else {
        // s0 == s1, including zero
        s0 = 0.5 * r;
        s1 = -0.5 * r;
      }

      Tensor<T>
      D(s0, 0.0, 0.0, s1);

      //
      // Eigenvectors
      //
      T
      c, s;

      boost::tie(c, s) = schur_sym(f, g, h);

      Tensor<T>
      V(c, -s, s, c);

      if (swap_diag == true) {
        // swap eigenvectors if eigenvalues were swapped
        std::swap(V(0,0), V(0,1));
        std::swap(V(1,0), V(1,1));
      }

      return std::make_pair(V, D);
    }

  } // anonymous namespace

  //
  // R^N eigenvalue decomposition for symmetric 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  eig_sym(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    Tensor<T>
    V(N), D(N);

    switch (N) {

    default:
      boost::tie(V, D) = eig_sym_NxN(A);
      break;

    case 2:
      boost::tie(V, D) = eig_sym_2x2(A);
      break;

    }

    return std::make_pair(V, D);
  }

  //
  // R^N eigenvalue decomposition for SPD 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  eig_spd(Tensor<T> const & A)
  {
    return eig_sym(A);
  }

  //
  // R^3 eigenvalue decomposition for SPD 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T>
  std::pair<Tensor<T>, Tensor<T> >
  eig_spd_cos(Tensor<T> const & A)
  {
    assert(A.get_dimension() == 3);

    // This algorithm comes from the journal article
    // Scherzinger and Dohrmann, CMAME 197 (2008) 4007-4015

    // this algorithm will return the eigenvalues in D
    // and the eigenvectors in V
    Tensor<T>
    D = zero<T>(3);

    Tensor<T>
    V = zero<T>(3);

    // not sure if this is necessary...
    T
    pi = acos(-1);

    // convenience operators
    const Tensor<T>
    I = identity<T>(3);

    int
    ii[3][2] = { { 1, 2 }, { 2, 0 }, { 0, 1 } };

    Tensor<T>
    rm = zero<T>(3);

    // scale the matrix to reduce the characteristic equation
    T
    trA = (1.0/3.0) * I1(A);

    Tensor<T>
    Ap(A - trA*I);

    // compute other invariants
    T
    J2 = I2(Ap);

    T
    J3 = det(Ap);

    // deal with volumetric tensors
    if (-J2 <= 1.e-30)
    {
      D(0,0) = trA;
      D(1,1) = trA;
      D(2,2) = trA;

      V(0,0) = 1.0;
      V(1,0) = 0.0;
      V(2,0) = 0.0;

      V(0,1) = 0.0;
      V(1,1) = 1.0;
      V(2,1) = 0.0;

      V(0,2) = 0.0;
      V(1,2) = 0.0;
      V(2,2) = 1.0;
    }
    else
    {
      // first things first, find the most dominant e-value
      // Need to solve cos(3 theta)=rhs for theta
      T
      t1 = 3.0 / -J2;

      T
      rhs = (J3 / 2.0) * T(sqrt(t1 * t1 * t1));

      T
      theta = pi / 2.0 * (1.0 - (rhs < 0 ? -1.0 : 1.0));

      if (fabs(rhs) <= 1.0) theta = acos(rhs);

      T
      thetad3 = theta / 3.0;

      if (thetad3 > pi / 6.0) thetad3 += 2.0 * pi / 3.0;

      // most dominant e-value
      D(2,2) = 2.0 * cos(thetad3) * sqrt(-J2 / 3.0);

      // now reduce the system
      Tensor<T>
      R = Ap - D(2,2) * I;

      // QR factorization with column pivoting
      Vector<T> a(3);
      a(0) = R(0,0)*R(0,0) + R(1,0)*R(1,0) + R(2,0)*R(2,0);
      a(1) = R(0,1)*R(0,1) + R(1,1)*R(1,1) + R(2,1)*R(2,1);
      a(2) = R(0,2)*R(0,2) + R(1,2)*R(1,2) + R(2,2)*R(2,2);

      // find the most dominant column
      int k = 0;
      T max = a(0);
      if (a(1) > max)
      {
        k = 1;
        max = a(1);
      }
      if (a(2) > max)
      {
        k = 2;
      }

      // normalize the most dominant column to get s1
      a(k) = sqrt(a(k));
      for (int i(0); i < 3; ++i)
        R(i,k) /= a(k);

      // dot products of dominant column with other two columns
      T d0 = 0.0;
      T d1 = 0.0;
      for (int i(0); i < 3; ++i)
      {
        d0 += R(i,k) * R(i,ii[k][0]);
        d1 += R(i,k) * R(i,ii[k][1]);
      }

      // projection
      for (int i(0); i < 3; ++i)
      {
        R(i,ii[k][0]) -= d0 * R(i,k);
        R(i,ii[k][1]) -= d1 * R(i,k);
      }

      // now finding next most dominant column
      a.clear();
      for (int i(0); i < 3; ++i)
      {
        a(0) += R(i,ii[k][0]) * R(i,ii[k][0]);
        a(1) += R(i,ii[k][1]) * R(i,ii[k][1]);
      }

      int p = 0;
      if (fabs(a(1)) > fabs(a(0))) p = 1;

      // normalize next most dominant column to get s2
      a(p) = sqrt(a(p));
      int k2 = ii[k][p];

      for (int i(0); i < 3; ++i)
        R(i,k2) /= a(p);

      // set first eigenvector as cross product of s1 and s2
      V(0,2) = R(1,k) * R(2,k2) - R(2,k) * R(1,k2);
      V(1,2) = R(2,k) * R(0,k2) - R(0,k) * R(2,k2);
      V(2,2) = R(0,k) * R(1,k2) - R(1,k) * R(0,k2);

      // normalize
      T
      mag = sqrt(V(0,2) * V(0,2) + V(1,2) * V(1,2) + V(2,2) * V(2,2));

      V(0,2) /= mag;
      V(1,2) /= mag;
      V(2,2) /= mag;

      // now for the other two eigenvalues, extract vectors
      Vector<T>
      rk(3, R(0,k), R(1,k), R(2,k));

      Vector<T>
      rk2(3, R(0,k2), R(1,k2), R(2,k2));

      // compute projections
      Vector<T>
      ak = Ap * rk;

      Vector<T>
      ak2 = Ap * rk2;

      // set up reduced remainder matrix
      rm(0,0) = dot(rk,ak);
      rm(0,1) = dot(rk,ak2);
      rm(1,1) = dot(rk2,ak2);

      // compute eigenvalues 2 and 3
      T
      b = 0.5 * (rm(0,0) - rm(1,1));

      T
      fac = (b < 0 ? -1.0 : 1.0);

      T
      arg = b * b + rm(0,1) * rm(0,1);

      if (arg == 0)
        D(0,0) = rm(1,1) + b;
      else
        D(0,0) = rm(1,1) + b - fac * sqrt(b * b + rm(0,1) * rm(0,1));

      D(1,1) = rm(0,0) + rm(1,1) - D(0,0);

      // update reduced remainder matrix
      rm(0,0) -= D(0,0);
      rm(1,0) = rm(0,1);
      rm(1,1) -= D(0,0);

      // again, find most dominant column
      a.clear();
      a(0) = rm(0,0) * rm(0,0) + rm(0,1) * rm(0,1);
      a(1) = rm(0,1) * rm(0,1) + rm(1,1) * rm(1,1);

      int k3 = 0;
      if (a(1) > a(0)) k3 = 1;
      if (a(k3) == 0.0)
      {
        rm(0,k3) = 1.0;
        rm(1,k3) = 0.0;
      }

      // set 2nd eigenvector via cross product
      V(0,0) = rm(0,k3) * rk2(0) - rm(1,k3) * rk(0);
      V(1,0) = rm(0,k3) * rk2(1) - rm(1,k3) * rk(1);
      V(2,0) = rm(0,k3) * rk2(2) - rm(1,k3) * rk(2);

      // normalize
      mag = sqrt(V(0,0) * V(0,0) + V(1,0) * V(1,0) + V(2,0) * V(2,0));
      V(0,0) /= mag;
      V(1,0) /= mag;
      V(2,0) /= mag;

      // set last eigenvector as cross product of other two
      V(0,1) = V(1,0) * V(2,2) - V(2,0) * V(1,2);
      V(1,1) = V(2,0) * V(0,2) - V(0,0) * V(2,2);
      V(2,1) = V(0,0) * V(1,2) - V(1,0) * V(0,2);

      // normalize
      mag = sqrt(V(0,1) * V(0,1) + V(1,1) * V(1,1) + V(2,1) * V(2,1));
      V(0,1) /= mag;
      V(1,1) /= mag;
      V(2,1) /= mag;

      // add back in the offset
      for (int i(0); i < 3; ++i)
        D(i,i) += trA;
    }

    return std::make_pair(V, D);
  }

  //
  // 4th-order identity I1
  // \return \f$ \delta_{ik} \delta_{jl} \f$ such that \f$ A = I_1 A \f$
  //
  template<typename T>
  const Tensor4<T>
  identity_1(const Index N)
  {
    Tensor4<T> I(N, T(0.0));

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            if (i == k && j == l) {
              I(i,j,k,l) = 1.0;
            }
          }
        }
      }
    }

    return I;
  }

  //
  // 4th-order identity I2
  // \return \f$ \delta_{il} \delta_{jk} \f$ such that \f$ A^T = I_2 A \f$
  //
  template<typename T>
  const Tensor4<T>
  identity_2(const Index N)
  {
    Tensor4<T> I(N, T(0.0));

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            if (i == l && j == k) {
              I(i,j,k,l) = 1.0;
            }
          }
        }
      }
    }

    return I;
  }

  //
  // 4th-order identity I3
  // \return \f$ \delta_{ij} \delta_{kl} \f$ such that \f$ I_A I = I_3 A \f$
  //
  template<typename T>
  const Tensor4<T>
  identity_3(const Index N)
  {
    Tensor4<T> I(N, T(0.0));

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            if (i == j && k == l) {
              I(i,j,k,l) = 1.0;
            }
          }
        }
      }
    }

    return I;
  }

  //
  // 4th-order tensor vector dot product
  // \param A 4th-order tensor
  // \param u vector
  // \return 3rd-order tensor \f$ A dot u \f$ as \f$ B_{ijk}=A_{ijkl}u_{l} \f$
  //
  template<typename T>
  Tensor3<T>
  dot(Tensor4<T> const & A, Vector<T> const & u)
  {
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor3<T> B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          T s = 0.0;
          for (Index l = 0; l < N; ++l) {
            s += A(i,j,k,l) * u(l);
          }
          B(i,j,k) = s;
        }
      }
    }

    return B;
  }

  //
  // vector 4th-order tensor dot product
  // \param A 4th-order tensor
  // \param u vector
  // \return 3rd-order tensor \f$ u dot A \f$ as \f$ B_{jkl}=u_{i}A_{ijkl} \f$
  //
  template<typename T>
  Tensor3<T>
  dot(Vector<T> const & u, Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor3<T> B(N);

    for (Index j = 0; j < N; ++j) {
      for (Index k = 0; k < N; ++k) {
        for (Index l = 0; l < N; ++l) {
          T s = 0.0;
          for (Index i = 0; i < N; ++i) {
            s += u(i) * A(i,j,k,l);
          }
          B(j,k,l) = s;
        }
      }
    }

    return B;
  }

  //
  // 4th-order tensor vector dot2 product
  // \param A 4th-order tensor
  // \param u vector
  // \return 3rd-order tensor \f$ A dot2 u \f$ as \f$ B_{ijl}=A_{ijkl}u_{k} \f$
  //
  template<typename T>
  Tensor3<T>
  dot2(Tensor4<T> const & A, Vector<T> const & u)
  {
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor3<T> B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index l = 0; l < N; ++l) {
          T s = 0.0;
          for (Index k = 0; k < N; ++k) {
            s += A(i,j,k,l) * u(k);
          }
          B(i,j,l) = s;
        }
      }
    }

    return B;
  }

  //
  // vector 4th-order tensor dot2 product
  // \param A 4th-order tensor
  // \param u vector
  // \return 3rd-order tensor \f$ u dot2 A \f$ as \f$ B_{ikl}=u_{j}A_{ijkl} \f$
  //
  template<typename T>
  Tensor3<T>
  dot2(Vector<T> const & u, Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    assert(u.get_dimension() == N);

    Tensor3<T> B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index k = 0; k < N; ++k) {
        for (Index l = 0; l < N; ++l) {
          T s = 0.0;
          for (Index j = 0; j < N; ++j) {
            s += u(j) * A(i,j,k,l);
          }
          B(i,k,l) = s;
        }
      }
    }

    return B;
  }

  //
  // 4th-order tensor 2nd-order tensor double dot product
  // \param A 4th-order tensor
  // \param B 2nd-order tensor
  // \return 2nd-order tensor \f$ A:B \f$ as \f$ C_{ij}=A_{ijkl}B_{kl} \f$
  //
  template<typename T>
  Tensor<T>
  dotdot(Tensor4<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor<T> C(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        T s = 0.0;
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            s += A(i,j,k,l) * B(k,l);
          }
        }
        C(i,j) = s;
      }
    }

    return C;
  }

  //
  // 2nd-order tensor 4th-order tensor double dot product
  // \param B 2nd-order tensor
  // \param A 4th-order tensor
  // \return 2nd-order tensor \f$ B:A \f$ as \f$ C_{kl}=A_{ijkl}B_{ij} \f$
  //
  template<typename T>
  Tensor<T>
  dotdot(Tensor<T> const & B, Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor<T> C(N);

    for (Index k = 0; k < N; ++k) {
      for (Index l = 0; l < N; ++l) {
        T s = 0.0;
        for (Index i = 0; i < N; ++i) {
          for (Index j = 0; j < N; ++j) {
            s += A(i,j,k,l) * B(i,j);
          }
        }
        C(k,l) = s;
      }
    }

    return C;
  }

  // Tensor4 Tensor4 double dot product
  /// \param A Tensor4
  /// \param B Tensor4
  /// \return a Tensor4 \f$ C_{ijkl} = A_{ijmn} : B){mnkl} \f$
  template<typename T>
  Tensor4<T>
  dotdot(Tensor4<T> const & A, Tensor4<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor4<T> C(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            T s = 0.0;
            for (Index m = 0; m < N; ++m) {
              for (Index n = 0; n < N; ++n) {
                s += A(i,j,m,n) * B(m,n,k,l);
              }
            }
            C(i,j,k,l) = s;
          }
        }
      }
    }

    return C;
  }

  //
  // 2nd-order tensor 2nd-order tensor tensor product
  // \param A 2nd-order tensor
  // \param B 2nd-order tensor
  // \return \f$ A \otimes B \f$
  //
  template<typename T>
  Tensor4<T>
  tensor(Tensor<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor4<T> C(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            C(i,j,k,l) = A(i,j) * B(k,l);
          }
        }
      }
    }

    return C;
  }

  //
  // odot operator useful for \f$ \frac{\partial A^{-1}}{\partial A} \f$
  // see Holzapfel eqn 6.165
  // \param A 2nd-order tensor
  // \param B 2nd-order tensor
  // \return \f$ A \odot B \f$ which is
  // \f$ C_{ijkl} = \frac{1}{2}(A_{ik} B_{jl} + A_{il} B_{jk}) \f$
  //
  template<typename T>
  Tensor4<T>
  odot(Tensor<T> const & A, Tensor<T> const & B)
  {
    const Index
    N = A.get_dimension();

    assert(B.get_dimension() == N);

    Tensor4<T> C(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            C(i,j,k,l) = 0.5 * (A(i,k) * B(j,l) + A(i,l) * B(j,k));
          }
        }
      }
    }

    return C;
  }

  //
  // R^N vector input
  // \param u vector
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Vector<T> & u)
  {
    const Index
    N = u.get_dimension();

    for (Index i = 0; i < N; ++i) {
      is >> u(i);
    }

    return is;
  }

  //
  // R^N vector output
  // \param u vector
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Vector<T> const & u)
  {
    const Index
    N = u.get_dimension();

    for (Index i = 0; i < N; ++i) {
      os << std::scientific << " " << u(i);
    }

    os << std::endl;
    os << std::endl;

    return os;
  }

  //
  // R^N tensor input
  // \param A tensor
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor<T> & A)
  {

    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        is >> A(i,j);
      }
    }

    return is;
  }

  //
  // R^N tensor output
  // \param A tensor
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        os << std::scientific << " " << A(i,j);
      }
      os << std::endl;
    }

    os << std::endl;
    os << std::endl;

    return os;
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
    const Index
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
    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          os << std::scientific << " ";
          os << A(i,j,k);
        }
        os << std::endl;
      }
      os << std::endl;
      os << std::endl;
    }

    return os;
  }

  //
  // 4th-order input
  // \param A 4th-order tensor
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor4<T> & A)
  {
    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            is >> A(i,j,k,l);
          }
        }
      }
    }

    return is;
  }

  //
  // 4th-order output
  // \param A 4th-order tensor
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<T> const & A)
  {
    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            os << std::scientific << " ";
            os << A(i,j,k,l);
          }
          os << std::endl;
        }
        os << std::endl;
        os << std::endl;
      }
      os << std::endl;
      os << std::endl;
      os << std::endl;
    }

    return os;
  }

} // namespace LCM

#endif // LCM_Tensor_t_cc
