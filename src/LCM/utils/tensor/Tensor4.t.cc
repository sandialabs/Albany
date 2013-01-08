//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor4_t_cc)
#define tensor_Tensor4_t_cc

namespace LCM {

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
  // \param A Tensor4
  // \param B Tensor4
  // \return a Tensor4 \f$ C_{ijkl} = A_{ijmn} : B){mnkl} \f$
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

    if (N == 0) {
      return os;
    }

    for (Index i = 0; i < N; ++i) {

      for (Index j = 0; j < N; ++j) {

        for (Index k = 0; k < N; ++k) {

          os << std::scientific << "," << A(i,j,k,0);

          for (Index l = 0; l < N; ++l) {

            os << std::scientific << "," << A(i,j,k,l);
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

#endif // tensor_Tensor4_t_cc
