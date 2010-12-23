//
// First cut of LCM small tensor utilities.
//
#include "Tensor.h"

namespace LCM {

  //
  //
  //
  template<typename Scalar>
  std::istream &
  operator<<(std::istream & is, Vector<Scalar> & u)
  {
    is >> u(0);
    is >> u(1);
    is >> u(2);

    return is;
  }

  //
  //
  //
  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Vector<Scalar> & u)
  {
    os << std::scientific << std::setw(24) << std::setprecision(16) << u(0);
    os << std::scientific << std::setw(24) << std::setprecision(16) << u(1);
    os << std::scientific << std::setw(24) << std::setprecision(16) << u(2);

    os << std::endl;
    os << std::endl;

    return os;
  }

  //
  //
  //
  template<typename Scalar>
  std::istream &
  operator<<(std::istream & is, Tensor<Scalar> & A)
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
  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Tensor<Scalar> & A)
  {
    os << std::scientific << std::setw(24) << std::setprecision(16) << A(0,0);
    os << std::scientific << std::setw(24) << std::setprecision(16) << A(0,1);
    os << std::scientific << std::setw(24) << std::setprecision(16) << A(0,2);

    os << std::endl;

    os << std::scientific << std::setw(24) << std::setprecision(16) << A(1,0);
    os << std::scientific << std::setw(24) << std::setprecision(16) << A(1,1);
    os << std::scientific << std::setw(24) << std::setprecision(16) << A(1,2);

    os << std::endl;

    os << std::scientific << std::setw(24) << std::setprecision(16) << A(2,0);
    os << std::scientific << std::setw(24) << std::setprecision(16) << A(2,1);
    os << std::scientific << std::setw(24) << std::setprecision(16) << A(2,2);

    os << std::endl;
    os << std::endl;

    return os;
  }

  //
  //
  //
  template<typename Scalar>
  Tensor<Scalar>
  dotdot(const Tensor4<Scalar> & A, const Tensor<Scalar> & B)
  {
    Tensor<Scalar> C;

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
  template<typename Scalar>
  Tensor<Scalar>
  dotdot(const Tensor<Scalar> & B, const Tensor4<Scalar> & A)
  {
    Tensor<Scalar> C;

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
  template<typename Scalar>
  Tensor4<Scalar>
  tensor(const Tensor<Scalar> & A, const Tensor<Scalar> & B)
  {
    Tensor4<Scalar> C;

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
  template<typename Scalar>
  Tensor4<Scalar>
  odot(const Tensor<Scalar> & A, const Tensor<Scalar> & B)
  {
    Tensor4<Scalar> C;

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
  template<typename Scalar>
  Tensor3<Scalar>::Tensor3()
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          e[i][j][k] = std::numeric_limits<Scalar>::quiet_NaN();
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename Scalar>
  Tensor3<Scalar>::Tensor3(const Scalar s)
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
  template<typename Scalar>
  Tensor3<Scalar>::Tensor3(const Tensor3<Scalar> & A)
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
  template<typename Scalar>
  Tensor3<Scalar>::~Tensor3()
  {
    return;
  }

  //
  //
  //
  template<typename Scalar>
  void
  Tensor3<Scalar>::clear()
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
  template<typename Scalar>
  Tensor3<Scalar> &
  Tensor3<Scalar>::operator=(const Tensor3<Scalar> & A)
  {
    if (*this != A) {
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
  template<typename Scalar>
  Tensor3<Scalar> &
  Tensor3<Scalar>::operator+=(const Tensor3<Scalar> & A)
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
  template<typename Scalar>
  Tensor3<Scalar> &
  Tensor3<Scalar>::operator-=(const Tensor3<Scalar> & A)
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
  template<typename Scalar>
  Tensor3<Scalar>
  operator+(const Tensor3<Scalar> & A, const Tensor3<Scalar> & B)
  {
    Tensor3<Scalar> S;

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
  template<typename Scalar>
  Tensor3<Scalar>
  operator-(const Tensor3<Scalar> & A, const Tensor3<Scalar> & B)
  {
    Tensor3<Scalar> S;

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
  // Fourth order tensor construction/destruction and utilities.
  //
  template<typename Scalar>
  Tensor4<Scalar>::Tensor4()
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            e[i][j][k][l] = std::numeric_limits<Scalar>::quiet_NaN();
          }
        }
      }
    }

    return;
  }

  //
  //
  //
  template<typename Scalar>
  Tensor4<Scalar>::Tensor4(const Scalar s)
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
  template<typename Scalar>
  Tensor4<Scalar>::Tensor4(const Tensor4<Scalar> & A)
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
  template<typename Scalar>
  Tensor4<Scalar>::~Tensor4()
  {
    return;
  }

  //
  //
  //
  template<typename Scalar>
  void
  Tensor4<Scalar>::clear()
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
  template<typename Scalar>
  Tensor4<Scalar> &
  Tensor4<Scalar>::operator=(const Tensor4<Scalar> & A)
  {
    if (*this != A) {
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
  template<typename Scalar>
  Tensor4<Scalar> &
  Tensor4<Scalar>::operator+=(const Tensor4<Scalar> & A)
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
  template<typename Scalar>
  Tensor4<Scalar> &
  Tensor4<Scalar>::operator-=(const Tensor4<Scalar> & A)
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
  template<typename Scalar>
  Tensor4<Scalar>
  operator+(const Tensor4<Scalar> & A, const Tensor4<Scalar> & B)
  {
    Tensor4<Scalar> S;

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
  template<typename Scalar>
  Tensor4<Scalar>
  operator-(const Tensor4<Scalar> & A, const Tensor4<Scalar> & B)
  {
    Tensor4<Scalar> S;

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
  template<typename Scalar>
  const Tensor4<Scalar>
  identity_1()
  {
    Tensor4<Scalar> I;

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
  template<typename Scalar>
  const Tensor4<Scalar>
  identity_2()
  {
    Tensor4<Scalar> I;

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
  template<typename Scalar>
  const Tensor4<Scalar>
  identity_3()
  {
    Tensor4<Scalar> I;

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
  template<typename Scalar>
  std::istream &
  operator<<(std::istream & is, Tensor3<Scalar> & A)
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
  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Tensor3<Scalar> & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          os << std::scientific << std::setw(24) << std::setprecision(16);
          os << A(i,j,k);
        }
        os << std::endl;
      }
      os << std::endl;
      os << std::endl;
    }

    return os;
  }

  template<typename Scalar>
  std::istream &
  operator<<(std::istream & is, Tensor4<Scalar> & A)
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
  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Tensor4<Scalar> & A)
  {
    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            os << std::scientific << std::setw(24) << std::setprecision(16);
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
