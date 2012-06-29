///
/// \file Tensor.t.cc
/// First cut of LCM small tensor utilities. Templates.
/// \author Alejandro Mota
/// \author Jake Ostien
///
#if !defined(LCM_Tensor_t_cc)
#define LCM_Tensor_t_cc

#include <boost/tuple/tuple.hpp>
#include <Sacado_MathFunctions.hpp>

namespace LCM {

  //
  // R^N 3rd-order tensor constructor with NaNs
  //
  template<typename T, Index N>
  Tensor3<T, N>::Tensor3()
  {
    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize(N);
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = std::numeric_limits<T>::quiet_NaN();
        }
      }
    }

    return;
  }

  //
  // R^3 3rd-order tensor constructor with NaNs
  //
  template<typename T>
  Tensor3<T, 3>::Tensor3()
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          e[i][j][k] = std::numeric_limits<T>::quiet_NaN();
        }
      }
    }

    return;
  }

  //
  // R^2 3rd-order tensor constructor with NaNs
  //
  template<typename T>
  Tensor3<T, 2>::Tensor3()
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
          e[i][j][k] = std::numeric_limits<T>::quiet_NaN();
        }
      }
    }

    return;
  }

  //
  // R^N 3rd-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T, Index N>
  Tensor3<T, N>::Tensor3(T const & s)
  {
    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize(N);
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = s;
        }
      }
    }

    return;
  }

  //
  // R^3 3rd-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T>
  Tensor3<T, 3>::Tensor3(T const & s)
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          e[i][j][k] = s;
        }
      }
    }

    return;
  }

  //
  // R^2 3rd-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T>
  Tensor3<T, 2>::Tensor3(T const & s)
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
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
  template<typename T, Index N>
  Tensor3<T, N>::Tensor3(Tensor3<T, N> const & A)
  {
    e.resize(N);

    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize(N);
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = A.e[i][j][k];
        }
      }
    }

    return;
  }

  //
  // R^3 copy constructor
  // 3rd-order tensor constructor from 3rd-order tensor
  // \param A from which components are copied
  //
  template<typename T>
  Tensor3<T, 3>::Tensor3(Tensor3<T, 3> const & A)
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          e[i][j][k] = A.e[i][j][k];
        }
      }
    }

    return;
  }

  //
  // R^2 copy constructor
  // 3rd-order tensor constructor from 3rd-order tensor
  // \param A from which components are copied
  //
  template<typename T>
  Tensor3<T, 2>::Tensor3(Tensor3<T, 2> const & A)
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
          e[i][j][k] = A.e[i][j][k];
        }
      }
    }

    return;
  }

  //
  // R^N 3rd-order tensor simple destructor
  //
  template<typename T, Index N>
  Tensor3<T, N>::~Tensor3()
  {
    return;
  }

  //
  // R^3 3rd-order tensor simple destructor
  //
  template<typename T>
  Tensor3<T, 3>::~Tensor3()
  {
    return;
  }

  //
  // R^2 3rd-order tensor simple destructor
  //
  template<typename T>
  Tensor3<T, 2>::~Tensor3()
  {
    return;
  }

  //
  // R^N 3rd-order tensor copy assignment
  //
  template<typename T, Index N>
  Tensor3<T, N> &
  Tensor3<T, N>::operator=(Tensor3<T, N> const & A)
  {
    if (this != &A) {
      e.resize(N);
      for (Index i = 0; i < N; ++i) {
        e[i].resize(N);
        for (Index j = 0; j < N; ++j) {
          e[i][j].resize(N);
          for (Index k = 0; k < N; ++k) {
            e[i][j][k] = A.e[i][j][k];
          }
        }
      }
    }

    return *this;
  }

  //
  // R^3 3rd-order tensor copy assignment
  //
  template<typename T>
  Tensor3<T, 3> &
  Tensor3<T, 3>::operator=(Tensor3<T, 3> const & A)
  {
    if (this != &A) {
      for (Index i = 0; i < 3; ++i) {
        for (Index j = 0; j < 3; ++j) {
          for (Index k = 0; k < 3; ++k) {
            e[i][j][k] = A.e[i][j][k];
          }
        }
      }
    }

    return *this;
  }

  //
  // R^2 3rd-order tensor copy assignment
  //
  template<typename T>
  Tensor3<T, 2> &
  Tensor3<T, 2>::operator=(Tensor3<T, 2> const & A)
  {
    if (this != &A) {
      for (Index i = 0; i < 2; ++i) {
        for (Index j = 0; j < 2; ++j) {
          for (Index k = 0; k < 2; ++k) {
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
  template<typename T, Index N>
  Tensor3<T, N> &
  Tensor3<T, N>::operator+=(Tensor3<T, N> const & A)
  {
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
  template<typename T, Index N>
  Tensor3<T, N> &
  Tensor3<T, N>::operator-=(Tensor3<T, N> const & A)
  {
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
  template<typename T, Index N>
  void
  Tensor3<T, N>::clear()
  {
    e.resize(N);
    for (Index i = 0; i < N; ++i) {
      e[i].resize(N);
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize(N);
        for (Index k = 0; k < N; ++k) {
          e[i][j][k] = 0.0;
        }
      }
    }

    return;
  }

  //
  // R^3 fill 3rd-order tensor with zeros
  //
  template<typename T>
  void
  Tensor3<T, 3>::clear()
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          e[i][j][k] = 0.0;
        }
      }
    }

    return;
  }

  //
  // R^2 fill 3rd-order tensor with zeros
  //
  template<typename T>
  void
  Tensor3<T, 2>::clear()
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
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
  template<typename T, Index N>
  Tensor3<T, N>
  operator+(Tensor3<T, N> const & A, Tensor3<T, N> const & B)
  {
    Tensor3<T, N> S;

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
  template<typename T, Index N>
  Tensor3<T, N>
  operator-(Tensor3<T, N> const & A, Tensor3<T, N> const & B)
  {
    Tensor3<T, N> S;

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
  template<typename T, Index N>
  Tensor3<T, N>
  operator-(Tensor3<T, N> const & A)
  {
    Tensor3<T, N> S;

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
  template<typename T, Index N>
  inline bool
  operator==(Tensor3<T, N> const & A, Tensor3<T, N> const & B)
  {
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
  template<typename T, Index N>
  inline bool
  operator!=(Tensor3<T, N> const & A, Tensor3<T, N> const & B)
  {
    return !(A==B);
  }

  //
  // Scalar 3rd-order tensor product
  // \param s scalar
  // \param A 3rd-order tensor
  // \return \f$ s A \f$
  //
  template<typename T, Index N, typename S>
  Tensor3<T, N>
  operator*(S const & s, Tensor3<T, N> const & A)
  {
    Tensor3<T, N> B;

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
  template<typename T, Index N, typename S>
  Tensor3<T, N>
  operator*(Tensor3<T, N> const & A, S const & s)
  {
    return s * A;
  }

  //
  // 3rd-order tensor vector product
  // \param A 3rd-order tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename T, Index N>
  Tensor<T, N>
  dot(Tensor3<T, N> const & A, Vector<T, N> const & u)
  {
    Tensor<T, N> B;

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
  template<typename T, Index N>
  Tensor<T, N>
  dot(Vector<T, N> const & u, Tensor3<T, N> const & A)
  {
    Tensor<T, N> B;

    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        T s = 0.0;
        for (Index k = 0; k < 3; ++k) {
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
  template<typename T, Index N>
  Tensor<T, N>
  dot2(Tensor3<T, N> const & A, Vector<T, N> const & u)
  {
    Tensor<T, N> B;

    for (Index i = 0; i < 3; ++i) {
      for (Index k = 0; k < 3; ++k) {
        B(i,k) = 0.0;
        T s = 0.0;
        for (Index j = 0; j < 3; ++j) {
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
  template<typename T, Index N>
  Tensor<T, N>
  dot2(Vector<T, N> const & u, Tensor3<T, N> const & A)
  {
    return dot2(A, u);
  }


  //
  // R^N 4th-order tensor constructor with NaNs
  //
  template<typename T, Index N>
  Tensor4<T, N>::Tensor4()
  {
    e.resize();
    for (Index i = 0; i < N; ++i) {
      e[i].resize();
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize();
        for (Index k = 0; k < N; ++k) {
          e[i][j][k].resize();
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = std::numeric_limits<T>::quiet_NaN();
          }
        }
      }
    }

    return;
  }

  //
  // R^3 4th-order tensor constructor with NaNs
  //
  template<typename T>
  Tensor4<T, 3>::Tensor4()
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          for (Index l = 0; l < 3; ++l) {
            e[i][j][k][l] = std::numeric_limits<T>::quiet_NaN();
          }
        }
      }
    }

    return;
  }

  //
  // R^2 4th-order tensor constructor with NaNs
  //
  template<typename T>
  Tensor4<T, 2>::Tensor4()
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
          for (Index l = 0; l < 2; ++l) {
            e[i][j][k][l] = std::numeric_limits<T>::quiet_NaN();
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
  template<typename T, Index N>
  Tensor4<T, N>::Tensor4(T const & s)
  {
    e.resize();
    for (Index i = 0; i < N; ++i) {
      e[i].resize();
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize();
        for (Index k = 0; k < N; ++k) {
          e[i][j][k].resize();
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = s;
          }
        }
      }
    }

    return;
  }

  //
  // R^3 4th-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T>
  Tensor4<T, 3>::Tensor4(T const & s)
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          for (Index l = 0; l < 3; ++l) {
            e[i][j][k][l] = s;
          }
        }
      }
    }

    return;
  }

  //
  // R^2 4th-order tensor constructor with a scalar
  // \param s all components set to this scalar
  //
  template<typename T>
  Tensor4<T, 2>::Tensor4(T const & s)
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
          for (Index l = 0; l < 2; ++l) {
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
  template<typename T, Index N>
  Tensor4<T, N>::Tensor4(Tensor4<T, N> const & A)
  {
    e.resize();
    for (Index i = 0; i < N; ++i) {
      e[i].resize();
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize();
        for (Index k = 0; k < N; ++k) {
          e[i][j][k].resize();
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = A.e[i][j][k][l];
          }
        }
      }
    }

    return;
  }

  //
  // R^3 copy constructor
  // 4th-order tensor constructor with 4th-order tensor
  // \param A from which components are copied
  //
  template<typename T>
  Tensor4<T, 3>::Tensor4(Tensor4<T, 3> const & A)
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          for (Index l = 0; l < 3; ++l) {
            e[i][j][k][l] = A.e[i][j][k][l];
          }
        }
      }
    }

    return;
  }

  //
  // R^2 copy constructor
  // 4th-order tensor constructor with 4th-order tensor
  // \param A from which components are copied
  //
  template<typename T>
  Tensor4<T, 2>::Tensor4(Tensor4<T, 2> const & A)
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
          for (Index l = 0; l < 2; ++l) {
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
  template<typename T, Index N>
  Tensor4<T, N>::~Tensor4()
  {
    return;
  }

  //
  // R^3 4th-order tensor simple destructor
  //
  template<typename T>
  Tensor4<T, 3>::~Tensor4()
  {
    return;
  }

  //
  // R^2 4th-order tensor simple destructor
  //
  template<typename T>
  Tensor4<T, 2>::~Tensor4()
  {
    return;
  }

  //
  // R^N 4th-order tensor copy assignment
  //
  template<typename T, Index N>
  Tensor4<T, N> &
  Tensor4<T, N>::operator=(Tensor4<T, N> const & A)
  {
    if (this != &A) {
      e.resize();
      for (Index i = 0; i < N; ++i) {
        e[i].resize();
        for (Index j = 0; j < N; ++j) {
          e[i][j].resize();
          for (Index k = 0; k < N; ++k) {
            e[i][j][k].resize();
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
  // R^3 4th-order tensor copy assignment
  //
  template<typename T>
  Tensor4<T, 3> &
  Tensor4<T, 3>::operator=(Tensor4<T, 3> const & A)
  {
    if (this != &A) {
      for (Index i = 0; i < 3; ++i) {
        for (Index j = 0; j < 3; ++j) {
          for (Index k = 0; k < 3; ++k) {
            for (Index l = 0; l < 3; ++l) {
              e[i][j][k][l] = A.e[i][j][k][l];
            }
          }
        }
      }
    }

    return *this;
  }

  //
  // R^2 4th-order tensor copy assignment
  //
  template<typename T>
  Tensor4<T, 2> &
  Tensor4<T, 2>::operator=(Tensor4<T, 2> const & A)
  {
    if (this != &A) {
      for (Index i = 0; i < 2; ++i) {
        for (Index j = 0; j < 2; ++j) {
          for (Index k = 0; k < 2; ++k) {
            for (Index l = 0; l < 2; ++l) {
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
  template<typename T, Index N>
  Tensor4<T, N> &
  Tensor4<T, N>::operator+=(Tensor4<T, N> const & A)
  {
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
  template<typename T, Index N>
  Tensor4<T, N> &
  Tensor4<T, N>::operator-=(Tensor4<T, N> const & A)
  {
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
  template<typename T, Index N>
  void
  Tensor4<T, N>::clear()
  {
    e.resize();
    for (Index i = 0; i < N; ++i) {
      e[i].resize();
      for (Index j = 0; j < N; ++j) {
        e[i][j].resize();
        for (Index k = 0; k < N; ++k) {
          e[i][j][k].resize();
          for (Index l = 0; l < N; ++l) {
            e[i][j][k][l] = 0.0;
          }
        }
      }
    }

    return;
  }

  //
  // R^3 fill 4th-order tensor with zeros
  //
  template<typename T>
  void
  Tensor4<T, 3>::clear()
  {
    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 3; ++k) {
          for (Index l = 0; l < 3; ++l) {
            e[i][j][k][l] = 0.0;
          }
        }
      }
    }

    return;
  }

  //
  // R^2 fill 4th-order tensor with zeros
  //
  template<typename T>
  void
  Tensor4<T, 2>::clear()
  {
    for (Index i = 0; i < 2; ++i) {
      for (Index j = 0; j < 2; ++j) {
        for (Index k = 0; k < 2; ++k) {
          for (Index l = 0; l < 2; ++l) {
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
  template<typename T, Index N>
  Tensor4<T, N>
  operator+(Tensor4<T, N> const & A, Tensor4<T, N> const & B)
  {
    Tensor4<T, N> S;

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
  template<typename T, Index N>
  Tensor4<T, N>
  operator-(Tensor4<T, N> const & A, Tensor4<T, N> const & B)
  {
    Tensor4<T, N> S;

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
  template<typename T, Index N>
  Tensor4<T, N>
  operator-(Tensor4<T, N> const & A)
  {
    Tensor4<T, N> S;

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
  template<typename T, Index N>
  inline bool
  operator==(Tensor4<T, N> const & A, Tensor4<T, N> const & B)
  {
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
  template<typename T, Index N>
  inline bool
  operator!=(Tensor4<T, N> const & A, Tensor4<T, N> const & B)
  {
    return !(A==B);
  }

  //
  // Scalar 4th-order tensor product
  // \param s scalar
  // \param A 4th-order tensor
  // \return \f$ s A \f$
  //
  template<typename T, Index N, typename S>
  Tensor4<T, N>
  operator*(S const & s, Tensor4<T, N> const & A)
  {
    Tensor4<T, N> B;

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
  template<typename T, Index N, typename S>
  Tensor4<T, N>
  operator*(Tensor4<T, N> const & A, S const & s)
  {
    return s * A;
  }

  //
  // R^N exponential map by Taylor series, radius of convergence is infinity
  // \param A tensor
  // \return \f$ \exp A \f$
  //
  template<typename T, Index N>
  Tensor<T, N>
  exp(Tensor<T, N> const & A)
  {
    const Index
    max_iter = 128;

    const T
    tol = std::numeric_limits<T>::epsilon();

    Tensor<T, N>
    term = identity<T, N>();

    // Relative error taken wrt to the first term, which is I and norm = 1
    T
    relative_error = 1.0;

    Tensor<T, N>
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
  template<typename T, Index N>
  Tensor<T, N>
  log(Tensor<T, N> const & A)
  {
    // Check whether skew-symmetric holds

    const Index
    max_iter = 128;

    const T
    tol = std::numeric_limits<T>::epsilon();

    const T
    norm_arg = norm_1(A);

    const Tensor<T, N>
    Am1 = A - identity<T, N>();

    Tensor<T, N>
    term = Am1;

    T
    norm_term = norm_1(term);

    T
    relative_error = norm_term / norm_arg;

    Tensor<T, N>
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
  template<typename T, Index N>
  Tensor<T, N>
  log_rotation(Tensor<T, N> const & R)
  {
    //firewalls, make sure R \in SO(N)
    assert(norm(R*transpose(R) - eye<T, N>())
        < 100.0 * std::numeric_limits<T>::epsilon());
    assert(fabs(det(R) - 1.0)
        < 100.0 * std::numeric_limits<T>::epsilon());

    std::cerr << "Logarithm of SO(N) N != 2,3 not implemented." << std::endl;
    exit(1);

    Tensor<T, N>
    r;
    return r;
  }

  //
  // R^3 logarithmic map of a rotation
  // \param R with \f$ R \in SO(3) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  //
  template<typename T>
  Tensor<T, 3>
  log_rotation(Tensor<T, 3> const & R)
  {
    //firewalls, make sure R \in SO(3)
    assert(norm(R*transpose(R) - eye<T, 3>())
        < 100.0 * std::numeric_limits<T>::epsilon());
    assert(fabs(det(R) - 1.0)
        < 100.0 * std::numeric_limits<T>::epsilon());

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

    Tensor<T, 3>
    r;

    if (theta == 0) {
      r = zero<T, 3>();
    } else if (fabs(cosine + 1.0) <
        10.0*std::numeric_limits<T>::epsilon())  {
      // Rotation angle is PI.
      r = log_rotation_pi(R);
    } else {
      r = T(theta/(2.0*sin(theta)))*(R - transpose(R));
    }

    return r;
  }

  //
  // R^2 logarithmic map of a rotation
  // \param R with \f$ R \in SO(2) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(2) \f$
  //
  template<typename T>
  Tensor<T, 2>
  log_rotation(Tensor<T, 2> const & R)
  {
    //firewalls, make sure R \in SO(2)
    assert(norm(R*transpose(R) - eye<T, 2>())
        < 100.0 * std::numeric_limits<T>::epsilon());
    assert(fabs(det(R) - 1.0)
        < 100.0 * std::numeric_limits<T>::epsilon());

    // acos requires input between -1 and +1
    T
    cosine = R(0,0);

    if (cosine < -1.0) {
      cosine = -1.0;
    } else if(cosine > 1.0) {
      cosine = 1.0;
    }

    T
    theta = acos(cosine);

    Tensor<T, 2>
    r(0.0, -1.0, 1.0, 0.0);

    r *= theta;

    return r;
  }

  // R^N Logarithmic map of a 180-degree rotation. Not implemented.
  // \param R with \f$ R \in SO(N) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(N) \f$
  //
  template<typename T, Index N>
  Tensor<T, N>
  log_rotation_pi(Tensor<T, N> const & R)
  {
    std::cerr << "Logarithm of SO(N) N != 2,3 not implemented." << std::endl;
    exit(1);

    Tensor<T, N>
    r;

    return r;
  }

  // R^3 logarithmic map of a 180-degree rotation
  // \param R with \f$ R \in SO(3) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  //
  template<typename T>
  Tensor<T, 3>
  log_rotation_pi(Tensor<T, 3> const & R)
  {
    // set firewall to make sure the rotation is indeed 180 degrees
    assert(fabs(0.5 * (trace(R) - 1.0) + 1.0)
        < std::numeric_limits<T>::epsilon());

    // obtain U from R = LU
    Tensor<T, 3>
    r = gaussian_elimination((R - identity<T, 3>()));

    // backward substitution (for rotation exp(R) only)
    const T
    tol = 10.0*std::numeric_limits<T>::epsilon();

    Vector<T, 3>
    normal;

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

    const T
    pi = acos(-1.0);
    r = pi * r;

    return r;
  }

  // R^2 Logarithmic map of a 180-degree rotation
  // \param R with \f$ R \in SO(2) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(2) \f$
  //
  template<typename T>
  Tensor<T, 2>
  log_rotation_pi(Tensor<T, 2> const & R)
  {
    // set firewall to make sure the rotation is indeed 180 degrees
    assert(fabs(R(0,0) - 1.0) < std::numeric_limits<T>::epsilon());

    const T
    theta = acos(-1.0);

    if (R(0,0) > 0.0) {
      theta = - theta;
    }

    // obtain U from R = LU
    Tensor<T, 2>
    r(0.0, -1.0, 1.0, 0.0);

    r *= theta;

    return r;
  }

  // Gaussian Elimination with partial pivot
  // \param matrix \f$ A \f$
  // \return \f$ U \f$ where \f$ A = LU \f$
  //
  template<typename T, Index N>
  Tensor<T, N>
  gaussian_elimination(Tensor<T, N> const & A)
  {
    Tensor<T, N>
    U = A;

    const T
    tol = 10.0 * std::numeric_limits<T>::epsilon();

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
  template<typename T, Index N>
  void
  givens_left(T const & c, T const & s, Index i, Index k, Tensor<T, N> & A)
  {
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
  template<typename T, Index N>
  void
  givens_right(T const & c, T const & s, Index i, Index k, Tensor<T, N> & A)
  {
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
  template<typename T, Index N>
  Tensor<T, N>
  exp_skew_symmetric(Tensor<T, N> const & r)
  {
    std::cerr << "Exponential of so(N) N != 2,3 not implemented." << std::endl;
    exit(1);

    Tensor<T, N>
    R;

    return R;
  }

  //
  // R^3 exponential map of a skew-symmetric tensor
  // \param r \f$ r \in so(3) \f$
  // \return \f$ R = \exp R \f$ with \f$ R \in SO(3) \f$
  //
  template<typename T>
  Tensor<T, 3>
  exp_skew_symmetric(Tensor<T, 3> const & r)
  {
    // Check whether skew-symmetry holds
    assert(norm(r+transpose(r)) < std::numeric_limits<T>::epsilon());

    T
    theta = sqrt(r(2,1)*r(2,1)+r(0,2)*r(0,2)+r(1,0)*r(1,0));

    Tensor<T, 3>
    R = identity<T, 3>();

    //Check whether norm == 0. If so, return identity.
    if (theta >= std::numeric_limits<T>::epsilon()) {
      R += sin(theta)/theta*r + (1.0-cos(theta))/(theta*theta)*r*r;
    }

    return R;
  }

  //
  // R^2 exponential map of a skew-symmetric tensor
  // \param r \f$ r \in so(3) \f$
  // \return \f$ R = \exp R \f$ with \f$ R \in SO(3) \f$
  //
  template<typename T>
  Tensor<T, 2>
  exp_skew_symmetric(Tensor<T, 2> const & r)
  {
    // Check whether skew-symmetry holds
    assert(norm(r+transpose(r)) < std::numeric_limits<T>::epsilon());

    T
    theta = r(1,0);

    T
    c = cos(theta);

    T
    s = sin(theta);

    Tensor<T, 2>
    R(c, -s, s, c);

    return R;
  }

  //
  // R^N off-diagonal norm. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ \sqrt(\sum_i \sum_{j, j\neq i} a_{ij}^2) \f$
  //
  template<typename T, Index N>
  T
  norm_off_diagonal(Tensor<T, N> const & A)
  {
    T s = 0.0;
    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        if (i != j) s += A(i,j)*A(i,j);
      }
    }
    return sqrt(s);
  }

  //
  // R^3 off-diagonal norm. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ \sqrt(\sum_i \sum_{j, j\neq i} a_{ij}^2) \f$
  //
  template<typename T>
  T
  norm_off_diagonal(Tensor<T, 3> const & A)
  {
    return sqrt(
        A(0,1)*A(0,1) + A(0,2)*A(0,2) + A(1,2)*A(1,2) +
        A(1,0)*A(1,0) + A(2,0)*A(2,0) + A(2,1)*A(2,1));
  }

  //
  // R^2 off-diagonal norm. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ \sqrt(\sum_i \sum_{j, j\neq i} a_{ij}^2) \f$
  //
  template<typename T>
  T
  norm_off_diagonal(Tensor<T, 2> const & A)
  {
    return sqrt(A(0,1)*A(0,1) + A(1,0)*A(1,0));
  }

  //
  // R^N arg max off-diagonal. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ (p,q) = arg max_{i \neq j} |a_{ij}| \f$
  //
  template<typename T, Index N>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T, N> const & A)
  {
    Index p = 0;
    Index q = 1;

    T s = fabs(A(p,q));

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

  //
  // R^3 arg max off-diagonal. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ (p,q) = arg max_{i \neq j} |a_{ij}| \f$
  //
  template<typename T>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T, 3> const & A)
  {
    Index p = 0;
    Index q = 1;

    T s = fabs(A(p,q));

    for (Index i = 0; i < 3; ++i) {
      for (Index j = 0; j < 3; ++j) {
        if (i != j && fabs(A(i,j)) > s) {
          p = i;
          q = j;
          s = fabs(A(i,j));
        }
      }
    }

    return std::make_pair(p,q);
  }

  //
  // R^2 arg max off-diagonal. Useful for SVD and other algorithms
  // that rely on Jacobi-type procedures.
  // \param A
  // \return \f$ (p,q) = arg max_{i \neq j} |a_{ij}| \f$
  //
  template<typename T>
  std::pair<Index, Index>
  arg_max_off_diagonal(Tensor<T, 2> const & A)
  {
    Index p = 0;
    Index q = 1;

    if (fabs(A(1,0)) > fabs(A(0,1))) {
      p = 1;
      q = 0;
    }

    return std::make_pair(p,q);
  }

  //
  // Singular value decomposition (SVD) for 2x2
  // bidiagonal matrix. Used for general 2x2 SVD.
  // Adapted from LAPAPCK's DLASV2, Netlib's dlasv2.c
  // and LBNL computational crystalography toolbox
  // \param f, g, h where A = [f, g; 0, h]
  // \return \f$ A = USV^T\f$
  //
  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
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
    } else if (ga > fa && fa / ga < std::numeric_limits<T>::epsilon()) {
      // case of very large ga
      s0 = ga;
      s1 = ha > 1.0 ?
          fa / (ga / ha) :
          (fa / ga) * ha;
      cu = 1.0;
      su = h / g;
      cv = f / g;
      sv = 1.0;
    } else {
      // normal case
      T d = fa - ha;
      T l = d != fa ?
          d / fa :
          1.0; // l \in [0,1]
      T m = g / f; // m \in (-1/macheps, 1/macheps)
      T t = 2.0 - l; // t \in [1,2]
      T mm = m * m;
      T tt = t * t;
      T s = sqrt(tt + mm); // s \in [1,1 + 1/macheps]
      T r = l != 0.0 ?
          sqrt(l * l + mm) :
          fabs(m); // r \in [0,1 + 1/macheps]
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
            copysign(T(2.0), f) * copysign(T(1.0), g) :
            g / copysign(d, f) + m / t;
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

    Tensor<T, 2> U(cu, -su, su, cu);

    Tensor<T, 2> S(s0, 0.0, 0.0, s1);

    Tensor<T, 2> V(cv, -sv, sv, cv);

    return boost::make_tuple(U, S, V);
  }

  //
  // R^N singular value decomposition (SVD)
  // \param A tensor
  // \return \f$ A = USV^T\f$
  //
  template<typename T, Index N>
  boost::tuple<Tensor<T, N>, Tensor<T, N>, Tensor<T, N> >
  svd(Tensor<T, N> const & A)
  {
    Tensor<T, N>
    S = A;

    Tensor<T, N>
    U = identity<T, N>();

    Tensor<T, N>
    V = identity<T, N>();

    T
    off = norm_off_diagonal(S);

    const T
    tol = std::numeric_limits<T>::epsilon() * norm(A);

    const Index
    max_iter = 1000;

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
      Tensor <T, 2>
      Spq(S(p,p), S(p,q), S(q,p), S(q,q));

      Tensor <T, 2>
      L, D, R;

      boost::tie(L, D, R) = svd(Spq);

      T const &
      cl = L(0,0);

      T const &
      sl = L(0,1);

      T const &
      cr = R(0,0);

      T const &
      sr = (sgn(R(0,1)) == sgn(R(1,0))) ? -R(0,1) : R(0,1);

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

  //
  // R^2 singular value decomposition (SVD)
  // \param A tensor
  // \return \f$ A = USV^T\f$
  //
  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
  svd2(Tensor<T, 2> const & A)
  {
    // Preliminaries
    const T
    Ju = A(0,0)*A(0,0) + A(0,1)*A(0,1);

    const T
    Lu = A(0,0)*A(1,0) + A(0,1)*A(1,1);

    const T
    Ku = A(1,0)*A(1,0) + A(1,1)*A(1,1);

    const T
    Jv = A(0,0)*A(0,0) + A(1,0)*A(1,0);

    const T
    Lv = A(0,0)*A(0,1) + A(1,0)*A(1,1);

    const T
    Kv = A(0,1)*A(0,1) + A(1,1)*A(1,1);

    // Form left singular vectors
    T cu, su;
    boost::tie(cu, su) = half_angle(Ju - Ku, 2 * Lu);

    Tensor<T, 2>
    U(cu, -su, su, cu);

    // Form right singular vectors
    T cv, sv;
    boost::tie(cv, sv) = half_angle(Jv - Kv, 2 * Lv);

    Tensor<T, 2>
    V(cv, -sv, sv, cv);

    // Compute singular values
   const Tensor<T, 2>
    X = transpose(U) * A * V;

    const T
    s0 = X(0,0);

    const T
    s1 = X(1,1);

    if (s0 < 0.0) {
      V(0,0) = -V(0,0);
      V(1,0) = -V(1,0);
    }

    if (s1 < 0.0) {
      V(0,1) = -V(0,1);
      V(1,1) = -V(1,1);
    }

    Tensor<T, 2>
    S(fabs(s0), 0.0, 0.0, fabs(s1));

    return boost::make_tuple(U, S, V);
  }

  //
  // R^2 singular value decomposition (SVD)
  // \param A tensor
  // \return \f$ A = USV^T\f$
  //
  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
  svd(Tensor<T, 2> const & A)
  {
    // First compute a givens rotation to eliminate 1,0 entry in tensor
    T c = 1.0;
    T s = 0.0;
    boost::tie(c, s) = givens(A(0,0), A(1,0));

    Tensor<T, 2>
    R(c, -s, s, c);

    Tensor<T, 2>
    B = R * A;

    // B is bidiagonal. Use specialized algorithm to compute its SVD
    Tensor<T, 2>
    X, S, V;

    boost::tie(X, S, V) = svd_bidiagonal(B(0,0), B(0,1), B(1,1));

    // Complete general 2x2 SVD with givens rotation calculated above
    Tensor<T, 2>
    U = transpose(R) * X;

    return boost::make_tuple(U, S, V);
  }

  //
  // R^N Left polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(N) \f$ and V SPD(N)
  //
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N > >
  polar_left(Tensor<T, N> const & F)
  {
    Tensor<T, N>
    X, S, Y;

    boost::tie(X, S, Y) = svd(F);

    Tensor<T, N>
    R = X * transpose(Y);

    Tensor<T, N>
    V = X * S * transpose(X);

    return std::make_pair(V, R);
  }

  //
  // R^3 left polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD(3)
  //
  template<typename T>
  std::pair<Tensor<T, 3>, Tensor<T, 3> >
  polar_left(Tensor<T, 3> const & F)
  {
    // set up return tensors
    Tensor<T, 3>
    R;

    Tensor<T, 3>
    V;

    // temporary tensor used to compute R
    Tensor<T, 3>
    Vinv;

    // compute spd tensor
    Tensor<T, 3>
    b = F * transpose(F);

    // get eigenvalues/eigenvectors
    Tensor<T, 3>
    eVal;

    Tensor<T, 3>
    eVec;

    boost::tie(eVec, eVal) = eig_spd(b);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<T, 3>
    x = zero<T, 3>();

    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));

    Tensor<T, 3>
    xi = zero<T, 3>();

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
  // R^2 left polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(2) \f$ and V SPD(2)
  //
  template<typename T>
  std::pair<Tensor<T, 2>, Tensor<T, 2 > >
  polar_left(Tensor<T, 2> const & F)
  {
    Tensor<T, 2>
    X, S, Y;

    boost::tie(X, S, Y) = svd(F);

    Tensor<T, 2>
    R = X * transpose(Y);

    Tensor<T, 2>
    V = X * S * transpose(X);

    return std::make_pair(V, R);
  }

  //
  // R^N right polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ RU = F \f$ with \f$ R \in SO(N) \f$ and U SPD(N)
  //
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N> >
  polar_right(Tensor<T, N> const & F)
  {
    Tensor<T, N>
    X, S, Y;

    boost::tie(X, S, Y) = svd(F);

    Tensor<T, N>
    R = X * transpose(Y);

    Tensor<T, N>
    U = Y * S * transpose(Y);

    return std::make_pair(R, U);
  }

  //
  // R^3 right polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ RU = F \f$ with \f$ R \in SO(3) \f$ and U SPD(3)
  //
  template<typename T>
  std::pair<Tensor<T, 3>, Tensor<T, 3> >
  polar_right(Tensor<T, 3> const & F)
  {
    Tensor<T, 3>
    R;

    Tensor<T, 3>
    U;

    // temporary tensor used to compute R
    Tensor<T, 3>
    Uinv;

    // compute spd tensor
    Tensor<T, 3>
    C = transpose(F) * F;

    // get eigenvalues/eigenvectors
    Tensor<T, 3>
    eVal;

    Tensor<T, 3>
    eVec;

    boost::tie(eVec, eVal) = eig_spd(C);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<T, 3>
    x = zero<T, 3>();

    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));

    Tensor<T, 3>
    xi = zero<T, 3>();

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
  // R^2 right polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ RU = F \f$ with \f$ R \in SO(2) \f$ and U SPD(2)
  //
  template<typename T>
  std::pair<Tensor<T, 2>,Tensor<T, 2> >
  polar_right(Tensor<T, 2> const & F)
  {
    Tensor<T, 2>
    X, S, Y;

    boost::tie(X, S, Y) = svd(F);

    Tensor<T, 2>
    R = X * transpose(Y);

    Tensor<T, 2>
    U = Y * S * transpose(Y);

    return std::make_pair(R, U);
  }

  //
  // R^N left polar decomposition with matrix logarithm for V
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(N) \f$ and V SPD(N), and log V
  //
  template<typename T, Index N>
  boost::tuple<Tensor<T, N>, Tensor<T, N>, Tensor<T, N> >
  polar_left_logV(Tensor<T, N> const & F)
  {
    Tensor<T, N>
    X, S, Y;

    boost::tie(X, S, Y) = svd(F);

    Tensor<T, N>
    R = X * transpose(Y);

    Tensor<T, N>
    V = X * S * transpose(X);

    Tensor<T, N>
    s = S;

    for (Index i = 0; i < N; ++i) {
      s(i,i) = log(s(i,i));
    }

    Tensor<T, N>
    v = X * s * transpose(X);

    return boost::make_tuple(V, R, v);
  }

  //
  // R^3 left polar decomposition with matrix logarithm for V
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD(3), and log V
  //
  template<typename T>
  boost::tuple<Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3> >
  polar_left_logV(Tensor<T, 3> const & F)
  {
    // set up return tensors
    Tensor<T, 3>
    R;

    Tensor<T, 3>
    V;

    //v = log(V)
    Tensor<T, 3>
    v;

    // temporary tensor used to compute R
    Tensor<T, 3>
    Vinv;

    // compute spd tensor
    Tensor<T, 3>
    b = F * transpose(F);

    // get eigenvalues/eigenvectors
    Tensor<T, 3>
    eVal;

    Tensor<T, 3>
    eVec;

    boost::tie(eVec, eVal) = eig_spd(b);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<T, 3>
    x = zero<T, 3>();

    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));

    Tensor<T, 3>
    xi = zero<T, 3>();

    xi(0,0) = 1.0/x(0,0);
    xi(1,1) = 1.0/x(1,1);
    xi(2,2) = 1.0/x(2,2);

    Tensor<T, 3>
    lnx = zero<T, 3>();

    lnx(0,0) = std::log(x(0,0));
    lnx(1,1) = std::log(x(1,1));
    lnx(2,2) = std::log(x(2,2));

    // compute V, Vinv, log(V)=v, and R
    V    = eVec * x * transpose(eVec);
    Vinv = eVec * xi * transpose(eVec);
    v    = eVec * lnx * transpose(eVec);
    R    = Vinv * F;

    return boost::make_tuple(V, R, v);
  }

  //
  // R^2 left polar decomposition with matrix logarithm for V
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(2) \f$ and V SPD(2), and log V
  //
  template<typename T>
  boost::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2> >
  polar_left_logV(Tensor<T, 2> const & F)
  {
    Tensor<T, 2>
    X, S, Y;

    boost::tie(X, S, Y) = svd(F);

    Tensor<T, 2>
    R = X * transpose(Y);

    Tensor<T, 2>
    V = X * S * transpose(X);

    Tensor<T, 2>
    s = S;

    s(0,0) = log(s(0,0));
    s(1,1) = log(s(1,1));

    Tensor<T, 2>
    v = X * s * transpose(X);

    return boost::make_tuple(V, R, v);
  }

  //
  // R^N logarithmic map using BCH expansion (3 terms)
  // \param x tensor
  // \param y tensor
  // \return Baker-Campbell-Hausdorff series up to 3 terms
  //
  template<typename T, Index N>
  Tensor<T, N>
  bch(Tensor<T, N> const & x, Tensor<T, N> const & y)
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
  // \param \f$ A \in S(2) \f$
  // \return \f$ c, s \rightarrow [c, -s; s, c]\f diagonalizes A$
  //
  template<typename T>
  std::pair<T, T>
  schur_sym(Tensor<T, 2> const & A)
  {
    T c = 1.0;
    T s = 0.0;

    if (A(0,1) != 0.0) {
      T t = (A(0,0) - A(1,1)) / (2.0 * A(0,1));

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

  //
  // R^N eigenvalue decomposition for symmetric 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N> >
  eig_sym(Tensor<T, N> const & A)
  {
    Tensor<T, N>
    D = A;

    Tensor<T, N>
    V = identity<T, N>();

    T
    off = norm_off_diagonal(D);

    const T
    tol = 100.0 * std::numeric_limits<T>::epsilon() * norm(A);

    const Index
    max_iter = 1000;

    Index
    num_iter = 0;

    while (off > tol || num_iter < max_iter) {

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
      Tensor <T, 2>
      Apq(A(p,p), A(p,q), A(q,p), A(q,q));

      T
      c, s;

      boost::tie(c, s) = schur_sym(Apq);

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

    return boost::make_tuple(V, D);
  }

  //
  // R^2 eigenvalue decomposition for symmetric 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T>
  std::pair<Tensor<T, 2>, Tensor<T, 2> >
  eig_sym(Tensor<T, 2> const & A)
  {
    T
    c, s;

    boost::tie(c, s) = schur_sym(A);

    Tensor<T, 2>
    V(c, -s, s, c);

    Tensor<T, 2>
    D = transpose(V) * A * V;

    return std::make_pair(V, D);
  }

  //
  // R^N eigenvalue decomposition for SPD 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T, Index N>
  std::pair<Tensor<T, N>, Tensor<T, N> >
  eig_spd(Tensor<T, N> const & A)
  {
    return eig_sym(A);
  }

  //
  // R^3 eigenvalue decomposition for SPD 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T>
  std::pair<Tensor<T, 3>, Tensor<T, 3> >
  eig_spd(Tensor<T, 3> const & A)
  {
    // This algorithm comes from the journal article
    // Scherzinger and Dohrmann, CMAME 197 (2008) 4007-4015

    // this algorithm will return the eigenvalues in D
    // and the eigenvectors in V
    Tensor<T, 3>
    D = zero<T, 3>();

    Tensor<T, 3>
    V = zero<T, 3>();

    // not sure if this is necessary...
    T
    pi = acos(-1);

    // convenience operators
    const Tensor<T, 3>
    I = identity<T, 3>();

    int
    ii[3][2] = { { 1, 2 }, { 2, 0 }, { 0, 1 } };

    Tensor<T, 3>
    rm = zero<T, 3>();

    // scale the matrix to reduce the characteristic equation
    T
    trA = (1.0/3.0) * I1(A);

    Tensor<T, 3>
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
      Tensor<T, 3>
      R = Ap - D(2,2) * I;

      // QR factorization with column pivoting
      Vector<T, 3> a;
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
      Vector<T, 3>
      rk(R(0,k), R(1,k), R(2,k));

      Vector<T, 3>
      rk2(R(0,k2), R(1,k2), R(2,k2));

      // compute projections
      Vector<T, 3>
      ak  = Ap * rk;

      Vector<T, 3>
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
  // R^2 eigenvalue decomposition for SPD 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename T>
  std::pair<Tensor<T, 2>,Tensor<T, 2> >
  eig_spd(Tensor<T, 2> const & A)
  {
    return eig_sym(A);
  }

  //
  // 4th-order identity I1
  // \return \f$ \delta_{ik} \delta_{jl} \f$ such that \f$ A = I_1 A \f$
  //
  template<typename T, Index N>
  const Tensor4<T, N>
  identity_1()
  {
    Tensor4<T, N> I;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            I(i,j,k,l) = (i == k && j == l) ? 1.0 : 0.0;
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
  template<typename T, Index N>
  const Tensor4<T, N>
  identity_2()
  {
    Tensor4<T, N> I;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            I(i,j,k,l) = (i == l && j == k) ? 1.0 : 0.0;
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
  template<typename T, Index N>
  const Tensor4<T, N>
  identity_3()
  {
    Tensor4<T, N> I;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            I(i,j,k,l) = (i == j && k == l) ? 1.0 : 0.0;
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
  template<typename T, Index N>
  Tensor3<T, N>
  dot(Tensor4<T, N> const & A, Vector<T, N> const & u)
  {
    Tensor3<T, N> B;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          B(i,j,k) = 0.0;
          for (Index l = 0; l < N; ++l) {
            B(i,j,k) = A(i,j,k,l) * u(l);
          }
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
  template<typename T, Index N>
  Tensor3<T, N>
  dot(Vector<T, N> const & u, Tensor4<T, N> const & A)
  {
    Tensor3<T, N> B;

    for (Index j = 0; j < N; ++j) {
      for (Index k = 0; k < N; ++k) {
        for (Index l = 0; l < N; ++l) {
          B(j,k,l) = 0.0;
          for (Index i = 0; i < N; ++i) {
            B(j,k,l) = u(i) * A(i,j,k,l);
          }
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
  template<typename T, Index N>
  Tensor3<T, N>
  dot2(Tensor4<T, N> const & A, Vector<T, N> const & u)
  {
    Tensor3<T, N> B;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index l = 0; l < N; ++l) {
          B(i,j,l) = 0.0;
          for (Index k = 0; k < N; ++k) {
            B(i,j,l) = A(i,j,k,l) * u(k);
          }
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
  template<typename T, Index N>
  Tensor3<T, N>
  dot2(Vector<T, N> const & u, Tensor4<T, N> const & A)
  {
    Tensor3<T, N> B;

    for (Index i = 0; i < N; ++i) {
      for (Index k = 0; k < N; ++k) {
        for (Index l = 0; l < N; ++l) {
          B(i,k,l) = 0.0;
          for (Index j = 0; j < N; ++j) {
            B(i,k,l) = u(j) * A(i,j,k,l);
          }
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
  template<typename T, Index N>
  Tensor<T, N>
  dotdot(Tensor4<T, N> const & A, Tensor<T, N> const & B)
  {
    Tensor<T, N> C;

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        C(i,j) = 0.0;
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            C(i,j) += A(i,j,k,l) * B(k,l);
          }
        }
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
  template<typename T, Index N>
  Tensor<T, N>
  dotdot(Tensor<T, N> const & B, Tensor4<T, N> const & A)
  {
    Tensor<T, N> C;

    for (Index k = 0; k < N; ++k) {
      for (Index l = 0; l < N; ++l) {
        C(k,l) = 0.0;
        for (Index i = 0; i < N; ++i) {
          for (Index j = 0; j < N; ++j) {
            C(k,l) += A(i,j,k,l) * B(i,j);
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
  template<typename T, Index N>
  Tensor4<T, N>
  tensor(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    Tensor4<T, N> C;

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
  template<typename T, Index N>
  Tensor4<T, N>
  odot(Tensor<T, N> const & A, Tensor<T, N> const & B)
  {
    Tensor4<T, N> C;

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
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Vector<T, N> & u)
  {
    for (Index i = 0; i < N; ++i) {
      is >> u(i);
    }

    return is;
  }

  //
  // R^3 vector input
  // \param u vector
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Vector<T, 3> & u)
  {
    is >> u(0);
    is >> u(1);
    is >> u(2);

    return is;
  }

  //
  // R^2 vector input
  // \param u vector
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Vector<T, 2> & u)
  {
    is >> u(0);
    is >> u(1);

    return is;
  }

  //
  // R^N vector output
  // \param u vector
  // \param os output stream
  // \return os output stream
  //
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Vector<T, N> const & u)
  {
    for (Index i = 0; i < N; ++i) {
      os << std::scientific << " " << u(i);
    }

    os << std::endl;
    os << std::endl;

    return os;
  }

  //
  // R^3 vector output
  // \param u vector
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Vector<T, 3> const & u)
  {
    os << std::scientific << " " << u(0);
    os << std::scientific << " " << u(1);
    os << std::scientific << " " << u(2);

    os << std::endl;
    os << std::endl;

    return os;
  }

  //
  // R^2 vector output
  // \param u vector
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Vector<T, 2> const & u)
  {
    os << std::scientific << " " << u(0);
    os << std::scientific << " " << u(1);

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
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Tensor<T, N> & A)
  {

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        is >> A(i,j);
      }
    }

    return is;
  }

  //
  // R^3 tensor input
  // \param A tensor
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor<T, 3> & A)
  {
    is >> A(0,0);
    is >> A(0,1);
    is >> A(0,2);

    is >> A(1,0);
    is >> A(1,1);
    is >> A(1,2);

    is >> A(2,0);
    is >> A(2,1);
    is >> A(2,2);

    return is;
  }

  //
  // R^2 tensor input
  // \param A tensor
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor<T, 2> & A)
  {
    is >> A(0,0);
    is >> A(0,1);

    is >> A(1,0);
    is >> A(1,1);

    return is;
  }

  //
  // R^N tensor output
  // \param A tensor
  // \param os output stream
  // \return os output stream
  //
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T, N> const & A)
  {
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
  // R^3 tensor output
  // \param A tensor
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T, 3> const & A)
  {

    os << std::scientific << " " << A(0,0);
    os << std::scientific << " " << A(0,1);
    os << std::scientific << " " << A(0,2);

    os << std::endl;

    os << std::scientific << " " << A(1,0);
    os << std::scientific << " " << A(1,1);
    os << std::scientific << " " << A(1,2);

    os << std::endl;

    os << std::scientific << " " << A(2,0);
    os << std::scientific << " " << A(2,1);
    os << std::scientific << " " << A(2,2);

    os << std::endl;
    os << std::endl;

    return os;
  }

  //
  // R^2 tensor output
  // \param A tensor
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T, 2> const & A)
  {

    os << std::scientific << " " << A(0,0);
    os << std::scientific << " " << A(0,1);

    os << std::endl;

    os << std::scientific << " " << A(1,0);
    os << std::scientific << " " << A(1,1);

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
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Tensor3<T, N> & A)
  {
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
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<T, N> const & A)
  {
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
  template<typename T, Index N>
  std::istream &
  operator>>(std::istream & is, Tensor4<T, N> & A)
  {
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
  template<typename T, Index N>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<T, N> const & A)
  {
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
