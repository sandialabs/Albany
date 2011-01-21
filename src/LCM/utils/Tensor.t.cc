//
// First cut of LCM small tensor utilities.
//
#if !defined(LCM_Tensor_t_cc)

namespace LCM {

  //
  //
  //
  template<typename ScalarT>
  std::istream &
  operator<<(std::istream & is, Vector<ScalarT> & u)
  {
    is >> u(0);
    is >> u(1);
    is >> u(2);

    return is;
  }

  //
  //
  //
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Vector<ScalarT> const & u)
  {
    os << std::scientific << " " << u(0);
    os << std::scientific << " " << u(1);
    os << std::scientific << " " << u(2);

    os << std::endl;
    os << std::endl;

    return os;
  }

  //
  //
  //
  template<typename ScalarT>
  std::istream &
  operator<<(std::istream & is, Tensor<ScalarT> & A)
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
  //
  //
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor<ScalarT> const & A)
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
  //
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor4<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    Tensor<ScalarT> C;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        C(i,j) = 0.0;
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            C(i,j) += A(i,j,k,l) * B(k,l);
          }
        }
      }
    }

    return C;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor<ScalarT> const & B, Tensor4<ScalarT> const & A)
  {
    Tensor<ScalarT> C;

    for (Index k = 0; k < MaxDim; ++k) {
      for (Index l = 0; l < MaxDim; ++l) {
        C(k,l) = 0.0;
        for (Index i = 0; i < MaxDim; ++i) {
          for (Index j = 0; j < MaxDim; ++j) {
            C(k,l) += A(i,j,k,l) * B(i,j);
          }
        }
      }
    }

    return C;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT>
  tensor(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    Tensor4<ScalarT> C;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            C(i,j,k,l) = A(i,j) * B(k,l);
          }
        }
      }
    }

    return C;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT>
  odot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B)
  {
    Tensor4<ScalarT> C;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            C(i,j,k,l) = 0.5 * (A(i,k) * B(j,l) + A(i,l) * B(j,k));
          }
        }
      }
    }

    return C;
  }

  //
  // Third order tensor construction/destruction and utilities.
  //
  template<typename ScalarT>
  Tensor3<ScalarT>::Tensor3()
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          e[i][j][k] = std::numeric_limits<ScalarT>::quiet_NaN();
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT>::Tensor3(const ScalarT s)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          e[i][j][k] = s;
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT>::Tensor3(Tensor3<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          e[i][j][k] = A.e[i][j][k];
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT>::~Tensor3()
  {
    return;
  }

  //
  //
  //
  template<typename ScalarT>
  void
  Tensor3<ScalarT>::clear()
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          e[i][j][k] = 0.0;
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT> &
  Tensor3<ScalarT>::operator=(Tensor3<ScalarT> const & A)
  {
    if (this != &A) {
      for (Index i = 0; i < MaxDim; ++i) {
        for (Index j = 0; j < MaxDim; ++j) {
          for (Index k = 0; k < MaxDim; ++k) {
            e[i][j][k] = A.e[i][j][k];
          }
        }
      }
    }

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT> &
  Tensor3<ScalarT>::operator+=(Tensor3<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          e[i][j][k] += A.e[i][j][k];
        }
      }
    }

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT> &
  Tensor3<ScalarT>::operator-=(Tensor3<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          e[i][j][k] -= A.e[i][j][k];
        }
      }
    }

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator+(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B)
  {
    Tensor3<ScalarT> S;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          S(i,j,k) = A(i,j,k) + B(i,j,k);
        }
      }
    }

    return S;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator-(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B)
  {
    Tensor3<ScalarT> S;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          S(i,j,k) = A(i,j,k) - B(i,j,k);
        }
      }
    }

    return S;
  }

  //
  //
  //
  template<typename ScalarT>
  inline bool
  operator==(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          if (A(i,j,k) != B(i,j,k)) {
            return false;
          }
        }
      }
    }

    return true;
  }

  //
  //
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B)
  {
    return !(A==B);
  }

  //
  //
  //
  template<typename ScalarT>
  std::istream &
  operator<<(std::istream & is, Tensor3<ScalarT> & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          is >> A(i,j,k);
        }
      }
    }

    return is;
  }

  //
  //
  //
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
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
  // Fourth order tensor construction/destruction and utilities.
  //
  template<typename ScalarT>
  Tensor4<ScalarT>::Tensor4()
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            e[i][j][k][l] = std::numeric_limits<ScalarT>::quiet_NaN();
          }
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT>::Tensor4(const ScalarT s)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            e[i][j][k][l] = s;
          }
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT>::Tensor4(Tensor4<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            e[i][j][k][l] = A.e[i][j][k][l];
          }
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT>::~Tensor4()
  {
    return;
  }

  //
  //
  //
  template<typename ScalarT>
  void
  Tensor4<ScalarT>::clear()
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            e[i][j][k][l] = 0.0;
          }
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT> &
  Tensor4<ScalarT>::operator=(Tensor4<ScalarT> const & A)
  {
    if (this != &A) {
      for (Index i = 0; i < MaxDim; ++i) {
        for (Index j = 0; j < MaxDim; ++j) {
          for (Index k = 0; k < MaxDim; ++k) {
            for (Index l = 0; l < MaxDim; ++l) {
              e[i][j][k][l] = A.e[i][j][k][l];
            }
          }
        }
      }
    }

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT> &
  Tensor4<ScalarT>::operator+=(Tensor4<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            e[i][j][k][l] += A.e[i][j][k][l];
          }
        }
      }
    }

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT> &
  Tensor4<ScalarT>::operator-=(Tensor4<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            e[i][j][k][l] -= A.e[i][j][k][l];
          }
        }
      }
    }

    return *this;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator+(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B)
  {
    Tensor4<ScalarT> S;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            S(i,j,k,l) = A(i,j,k,l) + B(i,j,k,l);
          }
        }
      }
    }

    return S;
  }

  //
  //
  //
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator-(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B)
  {
    Tensor4<ScalarT> S;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            S(i,j,k,l) = A(i,j,k,l) - B(i,j,k,l);
          }
        }
      }
    }

    return S;
  }

  //
  //
  //
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_1()
  {
    Tensor4<ScalarT> I;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            I(i,j,k,l) = (i == k && j == l) ? 1.0 : 0.0;
          }
        }
      }
    }

    return I;
  }

  //
  //
  //
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_2()
  {
    Tensor4<ScalarT> I;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            I(i,j,k,l) = (i == l && j == k) ? 1.0 : 0.0;
          }
        }
      }
    }

    return I;
  }

  //
  //
  //
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_3()
  {
    Tensor4<ScalarT> I;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            I(i,j,k,l) = (i == j && k == l) ? 1.0 : 0.0;
          }
        }
      }
    }

    return I;
  }

  //
  //
  //
  template<typename ScalarT>
  inline bool
  operator==(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
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
  //
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B)
  {
    return !(A==B);
  }

  //
  //
  //
  template<typename ScalarT>
  std::istream &
  operator<<(std::istream & is, Tensor4<ScalarT> & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
          is >> A(i,j,k,l);
          }
        }
      }
    }

    return is;
  }

  //
  //
  //
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<ScalarT> const & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
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
