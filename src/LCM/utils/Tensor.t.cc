#include <boost/tuple/tuple.hpp>

//
// First cut of LCM small tensor utilities.
//
#if !defined(LCM_Tensor_t_cc)

namespace LCM {

  //
  // Vector input
  //
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Vector<ScalarT> & u)
  {
    is >> u(0);
    is >> u(1);
    is >> u(2);

    return is;
  }

  //
  // Vector output
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
  // Tensor input
  //
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor<ScalarT> & A)
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
  // Tensor output
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
  // 4th-order tensor 2nd-order tensor double dot product
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
  // 2nd-order tensor 4th-order tensor double dot product
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
  // 2nd-order tensor 2nd-order tensor tensor product
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
  // odot operator useful for dA(-1)/dA, see Holzapfel eqn 6.165
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
  // 3rd-order tensor constructor with NaNs
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
  // 3rd-order tensor constructor with a scalar
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
  // 3rd-order tensor constructor from 3rd-order tensor
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
  // 3rd-order tensor destructor
  //
  template<typename ScalarT>
  Tensor3<ScalarT>::~Tensor3()
  {
    return;
  }

  //
  // Fill with zeros
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
  // 3rd-order tensor copy assignment
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
  // 3rd-order tensor increment
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
  // 3rd-order tensor decrement
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
  // 3rd-order tensor addition
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
  // 3rd-order tensor substraction
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
  // 3rd-order tensor inequality
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
  // 3rd-order tensor inequality
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B)
  {
    return !(A==B);
  }

  //
  // 3rd-order tensor input
  //
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor3<ScalarT> & A)
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
  // 3rd-order tensor output
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
  // 4th-order tensor constructor with NaNs
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
  // 4th-order tensor constructor with a scalar
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
  // 4th-order tensor constructor with 4th-order tensor
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
  // 4th-order tensor destructor
  //
  template<typename ScalarT>
  Tensor4<ScalarT>::~Tensor4()
  {
    return;
  }

  //
  // Fill with zeros
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
  // 4th-order tensor copy assignment
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
  // 4th-order tensor increment
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
  // 4th-order tensor decrement
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
  // 4th-order tensor addition
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
  // 4th-order tensor substraction
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
  // 4th-order identity delta_ik delta_jl, A = I_1 A
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
  // 4th-order identity delta_il delta_jk, A^T = I_2 A
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
  // 4th-order identity delta_ij delta_kl, trA I = I_3 A
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
  // 4th-order equality
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
  // 4th-order inequality
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B)
  {
    return !(A==B);
  }

  //
  // 4th-order input
  //
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor4<ScalarT> & A)
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
  // 4th-order output
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

  //
  // Left polar decomposition
  //
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  polarL(Tensor<ScalarT> const & F)
  {
    // set up return tensors
    Tensor<ScalarT> R;
    Tensor<ScalarT> V;

    // compute spd tensor
    Tensor<ScalarT> b(F*transpose(F));

    // get eigenvalues/eigenvectors
    Tensor<ScalarT> eVal;
    Tensor<ScalarT> eVec;
    
    boost::tie(eVal,eVec) = polarL(b);
    

    return std::make_pair(V,R);
  }

  //
  // Right polar decomposition
  //
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  polarR(Tensor<ScalarT> const & F)
  {
    Tensor<ScalarT> R;
    Tensor<ScalarT> U;

    

    return std::make_pair(R,U);
  }

  //
  // Eigenvalue decomposition for SPD 2nd order tensor
  //
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  eig_spd(Tensor<ScalarT> const & A)
  {
    ScalarT pi = 3.14159265358979323846;

    Tensor<ScalarT> D = zero<ScalarT>();
    Tensor<ScalarT> V = zero<ScalarT>();

    const Tensor<ScalarT> I(identity<ScalarT>());
    const Vector<ScalarT> e0(1.0, 0.0, 0.0);
    const Vector<ScalarT> e1(0.0, 1.0, 0.0);
    const Vector<ScalarT> e2(0.0, 0.0, 1.0);
    const int ii[3][2] = { 1, 2, 2, 0, 0, 1 };

    std::cout << "checking ii: \n";
    std::cout << ii[0][0] << " " << ii[0][1] << "\n";
    std::cout << ii[1][0] << " " << ii[1][1] << "\n";
    std::cout << ii[2][0] << " " << ii[2][1] << "\n";

    ScalarT trA = (1.0/3.0)*I1(A);
    Tensor<ScalarT> Ap(A - trA*I);

    ScalarT J2 = I2(Ap);
    ScalarT J3 = det(Ap);

    std::cout << "J2: " << J2 << std::endl;
    std::cout << "J3: " << J3 << std::endl;

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
      ScalarT rhs = (-J3/2.0)*pow(3.0/-J2,1.5);
      ScalarT theta = pi/2*(1.0 - (rhs < 0 ? -1.0 : 1.0));
      if (abs(rhs) <= 1.0) theta = acos(rhs);
      ScalarT thetad3 = theta/3.0;
      if (thetad3 > pi/6) thetad3 += 2.0*pi/3.0;

      // most dominant e-value
      D(0,0) = 2.0*cos(thetad3)*sqrt(-J2/3.0);

      std::cout << "D(0,0): " << D(0,0) << std::endl;
      std::cout << "rhs   : " << rhs << std::endl;
      std::cout << "theta : " << theta << std::endl;
      std::cout << "thd3  : " << thetad3 << std::endl;
      

      // now find the e-vector associated with the most dominant e-value
      Tensor<ScalarT> R = Ap - D(0,0)*I;

      // QR factorization with column pivoting
      Tensor<ScalarT> RtR = transpose(R)*R;
      
      // find the largest vector
      Vector<ScalarT> a;
      a(0) = e0*RtR*e0;
      a(1) = e1*RtR*e1;
      a(2) = e2*RtR*e2;
      
      int k = 0;
      ScalarT max = a(0);
      if (a(1) > max) 
      {
	k = 1;
	max = a(1);
      }
      if (a(2) > max)
      {
	k = 2;
      }

      // normalize
      a(k) = sqrt(a(k));
      for (int i(0); i < 3; ++i)
	RtR(i,k) /= a(k);
      

      ScalarT d0 = 0.0;
      ScalarT d1 = 0.0;
      // dot products
      for (int i(0); i < 3; ++i)
      {
	d0 += RtR(i,k)*RtR(i,ii[k][0]);
	d1 += RtR(i,k)*RtR(i,ii[k][1]);
      }

      for (int i(0); i < 3; ++i)
      {
	RtR(i,ii[k][0]) -= d0*RtR(i,k);
	RtR(i,ii[k][1]) -= d1*RtR(i,k);
      }

      a(0) = 0.0;
      a(1) = 0.0;
      for (int i(0); i < 3; ++i)
      {
	a(0) += RtR(i,ii[k][0])*RtR(i,ii[k][0]);
	a(1) += RtR(i,ii[k][1])*RtR(i,ii[k][1]);
      }
      
      int p = 0;
      if (abs(a(0)) > abs(a(1))) p = 1;

      // normalize
      a(p) = sqrt(a(p));
      int k2 = ii[k][p];

      for (int i(0); i < 3; ++i)
	RtR(i,k2) /= a(p);

      // set first eigenvector
      V(0,0) = RtR(1,k)*RtR(2,k2) - RtR(2,k)*RtR(1,k2);
      V(1,0) = RtR(2,k)*RtR(0,k2) - RtR(0,k)*RtR(2,k2);
      V(2,0) = RtR(0,k)*RtR(1,k2) - RtR(1,k)*RtR(0,k2);

      ScalarT mag = sqrt(V(0,0)*V(0,0) + V(1,0)*V(1,0) + V(2,0)*V(2,0));
      V(0,0) /= mag;
      V(1,0) /= mag;
      V(2,0) /= mag;

      // now for the other two eigenvalues
      Vector<ScalarT> rk(RtR(0,k), RtR(1,k), RtR(2,k));
      Vector<ScalarT> rk2(RtR(0,k2), RtR(1,k2), RtR(2,k2));

      Vector<ScalarT> ak  = Ap*rk;
      Vector<ScalarT> ak2 = Ap*rk2;

      

      // add back in the offset
      for (int i(0); i < 3; ++i)
	D(i,i) += trA;
    }

    return std::make_pair(V,D);
  }

} // namespace LCM

#endif // LCM_Tensor_t_cc
