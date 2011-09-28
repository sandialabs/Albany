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
  // Exponential map by Taylor series, radius of convergence is infinity
  // \param A tensor
  // \return \f$ \exp A \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  exp(Tensor<ScalarT> const & A)
  {
    const Index
      maxNumIter = 128;

    const ScalarT
      tol = std::numeric_limits<ScalarT>::epsilon();

    Index k = 0;

    Tensor<ScalarT>
      term = identity<ScalarT>();

    // Relative error taken wrt to the first term, which is I and norm = 1
    ScalarT
      relError = 1.0;

    Tensor<ScalarT>
      B = term;

    while (relError > tol && k < maxNumIter) {
      term = ScalarT(1.0 / (k + 1.0)) * term * A;
      B = B + term;
      relError = norm_1(term);
      ++k;
    }

    return B;
  }

  //
  // Logarithmic map by Taylor series, converges for \f$ |A-I| < 1 \f$
  // \param A tensor
  // \return \f$ \log A \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  log(Tensor<ScalarT> const & A)
  {
    // Check whether skew-symmetric holds

    const Index
      maxNumIter = 128;

    const ScalarT
      tol = std::numeric_limits<ScalarT>::epsilon();

    Index k = 1;

    const ScalarT
      normA = norm_1(A);

    const Tensor<ScalarT>
      Am1 = A - identity<ScalarT>();

    Tensor<ScalarT>
      term = Am1;

    ScalarT
      normTerm = norm_1(term);

    ScalarT
      relError = normTerm / normA;

    Tensor<ScalarT>
      B = term;

    while (relError > tol && k < maxNumIter) {
      term = - (k / (k + 1.0)) * term * Am1;
      B = B + term;
      normTerm = norm_1(term);
      relError = normTerm / normA;
      ++k;
    }

    return B;
  }

  //
  // Logarithmic map of a rotation
  // \param R with \f$ R \in SO(3) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  log_rotation(Tensor<ScalarT> const & R)
  {
    //firewalls, make sure R \in SO(3)
    assert(norm(R*transpose(R) - eye<ScalarT>())
	   < 100.0 * std::numeric_limits<ScalarT>::epsilon());
    assert((det(R) - 1.0)
	   < 100.0 * std::numeric_limits<ScalarT>::epsilon());

    // acos requires input between -1 and +1
    ScalarT
      cosine = 0.5*(trace(R) - 1.0);

    if (cosine < -1.0) {
      cosine = -1.0;
    } else if(cosine > 1.0) {
      cosine = 1.0;
    }

    ScalarT
      theta = acos(cosine);

    Tensor<ScalarT> r;

    if (theta == 0) {
      r = zero<ScalarT>();
    } else if (abs(cosine + 1.0) <
	       10.0*std::numeric_limits<ScalarT>::epsilon())  {
      // Rotation angle is PI.
      r = log_rotation_pi(R);
    } else {
      r = ScalarT(theta/(2.0*sin(theta)))*(R - transpose(R));
    }

    return r;
  }

  // Logarithmic map of a 180-degree rotation
  // \param R with \f$ R \in SO(3) \f$
  // \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  log_rotation_pi(Tensor<ScalarT> const & R)
  {
    // set firewall to make sure the rotation is indeed 180 degrees
    assert(abs(0.5 * (trace(R) - 1.0) + 1.0)
	   < std::numeric_limits<ScalarT>::epsilon());

    // obtain U from R = LU
    Tensor<ScalarT>
      r = GaussianElimination((R - identity<ScalarT>()));

    // backward substitution (for rotation exp(R) only)
    const ScalarT tol = 10.0*std::numeric_limits<ScalarT>::epsilon();
    Vector<ScalarT> normal;

    if (abs(r(2,2)) < tol){
      normal(2) = 1.0;
    } else {
      normal(2) = 0.0;
    }

    if (abs(r(1,1)) < tol){
      normal(1) = 1.0;
    } else {
      normal(1) = -normal(2)*r(1,2)/r(1,1);
    }

    if (abs(r(0,0)) < tol){
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

    const ScalarT pi = acos(-1.0);
    r = pi * r;

    return r;
  }

  // Gaussian Elimination with partial pivot
  // \param matrix \f$ A \f$
  // \return \f$ U \f$ where \f$ A = LU \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  GaussianElimination(Tensor<ScalarT> const & A)
  {
    Tensor<ScalarT>
      U = A;

    const ScalarT
      tol = 10.0 * std::numeric_limits<ScalarT>::epsilon();

    Index i = 0;
    Index j = 0;
    Index i_max = 0;

    while ((i <  MaxDim) && (j < MaxDim)) {
      // find pivot in column j, starting in row i
      i_max = i;
      for (Index k = i + 1; k < MaxDim; ++k) {
        if (abs(U(k,j) > abs(U(i_max,j)))) {
          i_max = k;
        }
      }

      // Check if A(i_max,j) equal to or very close to 0
      if (abs(U(i_max,j)) > tol){
        // swap rows i and i_max and divide each entry in row i
        // by U(i,j)
        for (Index k = 0; k < MaxDim; ++k) {
          std::swap(U(i,k), U(i_max,k));
        }

        for (Index k = 0; k < MaxDim; ++k) {
          U(i,k) = U(i,k) / U(i,j);
        }

        for (Index l = i + 1; l < MaxDim; ++l) {
          for (Index k = 0; k < MaxDim; ++k) {
            U(l,k) = U(l,k) - U(l,i) * U(i,k) / U(i,i);
          }
        }

        ++i;
      }

      ++j;

    }

    return U;
  }



  //
  // Exponential map of a skew-symmetric tensor
  // \param r \f$ r \in so(3) \f$
  // \return \f$ R = \exp R \f$ with \f$ R \in SO(3) \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  exp_skew_symmetric(Tensor<ScalarT> const & r)
  {
    Tensor<ScalarT> R;

    // Check whether skew-symmetry holds
    assert(norm(r+transpose(r)) < std::numeric_limits<ScalarT>::epsilon());
    ScalarT normVector = sqrt(r(2,1)*r(2,1)+r(0,2)*r(0,2)+r(1,0)*r(1,0));
    //Check whether norm == 0. If so, return identity.
    if (normVector < std::numeric_limits<ScalarT>::epsilon()) {
      R = identity<ScalarT>();
    } else {
      // compute the norm of the basis vector
      R = identity<ScalarT>() + sin(normVector)/normVector*r+
	(1.0-cos(normVector))/(normVector*normVector)*r*r;
    }



    return R;

  }

  //
  // Left polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD
  //
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  polar_left(Tensor<ScalarT> const & F)
  {
    // set up return tensors
    Tensor<ScalarT> R;
    Tensor<ScalarT> V;

    // temporary tensor used to compute R
    Tensor<ScalarT> Vinv;

    // compute spd tensor
    Tensor<ScalarT> b = F*transpose(F);

    // get eigenvalues/eigenvectors
    Tensor<ScalarT> eVal;
    Tensor<ScalarT> eVec;
    boost::tie(eVec,eVal) = eig_spd(b);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<ScalarT> x = zero<ScalarT>();
    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));
    Tensor<ScalarT> xi = zero<ScalarT>();
    xi(0,0) = 1.0/x(0,0);
    xi(1,1) = 1.0/x(1,1);
    xi(2,2) = 1.0/x(2,2);

    // compute V, Vinv, and R
    V    = eVec*x*transpose(eVec);
    Vinv = eVec*xi*transpose(eVec);
    R    = Vinv*F;

    return std::make_pair(V,R);
  }

  //
  // Right polar decomposition
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ RU = F \f$ with \f$ R \in SO(3) \f$ and U SPD
  //
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  polar_right(Tensor<ScalarT> const & F)
  {
    Tensor<ScalarT> R;
    Tensor<ScalarT> U;

    // temporary tensor used to compute R
    Tensor<ScalarT> Uinv;

    // compute spd tensor
    Tensor<ScalarT> C = transpose(F)*F;

    // get eigenvalues/eigenvectors
    Tensor<ScalarT> eVal;
    Tensor<ScalarT> eVec;
    boost::tie(eVec,eVal) = eig_spd(C);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<ScalarT> x = zero<ScalarT>();
    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));
    Tensor<ScalarT> xi = zero<ScalarT>();
    xi(0,0) = 1.0/x(0,0);
    xi(1,1) = 1.0/x(1,1);
    xi(2,2) = 1.0/x(2,2);

    // compute U, Uinv, and R
    U    = eVec*x*transpose(eVec);
    Uinv = eVec*xi*transpose(eVec);
    R    = F*Uinv;

    return std::make_pair(R,U);
  }

  //
  // Left polar decomposition with matrix logarithm for V
  // \param F tensor (often a deformation-gradient-like tensor)
  // \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD, and log V
  //
  template<typename ScalarT>
  boost::tuple<Tensor<ScalarT>,Tensor<ScalarT>,Tensor<ScalarT> >
  polar_left_logV(Tensor<ScalarT> const & F)
  {
    // set up return tensors
    Tensor<ScalarT> R;
    Tensor<ScalarT> V;
    Tensor<ScalarT> v; //v = log(V)

    // temporary tensor used to compute R
    Tensor<ScalarT> Vinv;

    // compute spd tensor
    Tensor<ScalarT> b = F*transpose(F);

    // get eigenvalues/eigenvectors
    Tensor<ScalarT> eVal;
    Tensor<ScalarT> eVec;
    boost::tie(eVec,eVal) = eig_spd(b);

    // compute sqrt() and inv(sqrt()) of eigenvalues
    Tensor<ScalarT> x = zero<ScalarT>();
    x(0,0) = sqrt(eVal(0,0));
    x(1,1) = sqrt(eVal(1,1));
    x(2,2) = sqrt(eVal(2,2));
    Tensor<ScalarT> xi = zero<ScalarT>();
    xi(0,0) = 1.0/x(0,0);
    xi(1,1) = 1.0/x(1,1);
    xi(2,2) = 1.0/x(2,2);
    Tensor<ScalarT> lnx = zero<ScalarT>();
    lnx(0,0) = std::log(x(0,0));
    lnx(1,1) = std::log(x(1,1));
    lnx(2,2) = std::log(x(2,2));

    // compute V, Vinv, log(V)=v, and R
    V    = eVec*x*transpose(eVec);
    Vinv = eVec*xi*transpose(eVec);
    v    = eVec*lnx*transpose(eVec);
    R    = Vinv*F;

    return boost::make_tuple(V,R,v);
  }

  //
  // Logarithmic map using BCH expansion (3 terms)
  // \param v tensor
  // \param r tensor
  // \return Baker-Campbell-Hausdorff series up to 3 terms
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  bch(Tensor<ScalarT> const & v, Tensor<ScalarT> const & r)
  {
    Tensor<ScalarT> f;

    return f =
      v + r // term 1
      + ScalarT(0.5)*(v*r - r*v) // term 2
      + ScalarT(1.0/12.0)*
      (v*v*r - ScalarT(2.0)*v*r*v +
       v*r*r + r*v*v - ScalarT(2.0)*r*v*r + r*r*v); // term 3
  }

  //
  // Eigenvalue decomposition for SPD 2nd-order tensor
  // \param A tensor
  // \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  //
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  eig_spd(Tensor<ScalarT> const & A)
  {
    // This algorithm comes from the journal article
    // Scherzinger and Dohrmann, CMAME 197 (2008) 4007-4015

    // this algorithm will return the eigenvalues in D
    // and the eigenvectors in V
    Tensor<ScalarT> D = zero<ScalarT>();
    Tensor<ScalarT> V = zero<ScalarT>();

    // not sure if this is necessary...
    ScalarT pi = acos(-1);

    // convenience operators
    const Tensor<ScalarT> I(identity<ScalarT>());
    int ii[3][2] = { { 1, 2 }, { 2, 0 }, { 0, 1 } } ;
    Tensor<ScalarT> rm = zero<ScalarT>();

    // scale the matrix to reduce the characteristic equation
    ScalarT trA = (1.0/3.0)*I1(A);
    Tensor<ScalarT> Ap(A - trA*I);

    // compute other invariants
    ScalarT J2 = I2(Ap);
    ScalarT J3 = det(Ap);

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
      ScalarT t1 = 3.0/-J2;
      ScalarT rhs = (J3/2.0)*ScalarT(std::sqrt(t1*t1*t1));
      ScalarT theta = pi/2.0*(1.0 - (rhs < 0 ? -1.0 : 1.0));
      if (std::abs(rhs) <= 1.0) theta = acos(rhs);
      ScalarT thetad3 = theta/3.0;
      if (thetad3 > pi/6.0) thetad3 += 2.0*pi/3.0;

      // most dominant e-value
      D(2,2) = 2.0*cos(thetad3)*sqrt(-J2/3.0);

      // now reduce the system
      Tensor<ScalarT> R = Ap - D(2,2)*I;

      // QR factorization with column pivoting
      Vector<ScalarT> a;
      a(0) = R(0,0)*R(0,0) + R(1,0)*R(1,0) + R(2,0)*R(2,0);
      a(1) = R(0,1)*R(0,1) + R(1,1)*R(1,1) + R(2,1)*R(2,1);
      a(2) = R(0,2)*R(0,2) + R(1,2)*R(1,2) + R(2,2)*R(2,2);

      // find the most dominant column
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

      // normalize the most dominant column to get s1
      a(k) = sqrt(a(k));
      for (int i(0); i < 3; ++i)
        R(i,k) /= a(k);

      // dot products of dominant column with other two columns
      ScalarT d0 = 0.0;
      ScalarT d1 = 0.0;
      for (int i(0); i < 3; ++i)
      {
        d0 += R(i,k)*R(i,ii[k][0]);
        d1 += R(i,k)*R(i,ii[k][1]);
      }

      // projection
      for (int i(0); i < 3; ++i)
      {
        R(i,ii[k][0]) -= d0*R(i,k);
        R(i,ii[k][1]) -= d1*R(i,k);
      }

      // now finding next most dominant column
      a.clear();
      for (int i(0); i < 3; ++i)
      {
        a(0) += R(i,ii[k][0])*R(i,ii[k][0]);
        a(1) += R(i,ii[k][1])*R(i,ii[k][1]);
      }

      int p = 0;
      if (std::abs(a(1)) > std::abs(a(0))) p = 1;

      // normalize next most dominant column to get s2
      a(p) = sqrt(a(p));
      int k2 = ii[k][p];

      for (int i(0); i < 3; ++i)
        R(i,k2) /= a(p);

      // set first eigenvector as cross product of s1 and s2
      V(0,2) = R(1,k)*R(2,k2) - R(2,k)*R(1,k2);
      V(1,2) = R(2,k)*R(0,k2) - R(0,k)*R(2,k2);
      V(2,2) = R(0,k)*R(1,k2) - R(1,k)*R(0,k2);

      // normalize
      ScalarT mag = std::sqrt(V(0,2)*V(0,2) + V(1,2)*V(1,2) + V(2,2)*V(2,2));
      V(0,2) /= mag;
      V(1,2) /= mag;
      V(2,2) /= mag;

      // now for the other two eigenvalues, extract vectors
      Vector<ScalarT> rk(R(0,k), R(1,k), R(2,k));
      Vector<ScalarT> rk2(R(0,k2), R(1,k2), R(2,k2));

      // compute projections
      Vector<ScalarT> ak  = Ap*rk;
      Vector<ScalarT> ak2 = Ap*rk2;

      // set up reduced remainder matrix
      rm(0,0) = dot(rk,ak);
      rm(0,1) = dot(rk,ak2);
      rm(1,1) = dot(rk2,ak2);
     
      // compute eigenvalues 2 and 3
      ScalarT b = 0.5*(rm(0,0) - rm(1,1));
      ScalarT fac = (b < 0 ? -1.0 : 1.0);
      ScalarT arg = b*b+rm(0,1)*rm(0,1);
      if (arg == 0) 
	D(0,0) = rm(1,1) + b;
      else
	D(0,0) = rm(1,1) + b - fac*std::sqrt(b*b+rm(0,1)*rm(0,1));
      D(1,1) = rm(0,0) + rm(1,1) - D(0,0);

      // update reduced remainder matrix
      rm(0,0) -= D(0,0);
      rm(1,0) = rm(0,1);
      rm(1,1) -= D(0,0);

      // again, find most dominant column
      a.clear();
      a(0) = rm(0,0)*rm(0,0) + rm(0,1)*rm(0,1);
      a(1) = rm(0,1)*rm(0,1) + rm(1,1)*rm(1,1);

      int k3 = 0;
      if ( a(1) > a(0) ) k3 = 1;
      if ( a(k3) == 0.0 )
      {
	rm(0,k3) = 1.0;
	rm(1,k3) = 0.0;
      }

      // set 2nd eigenvector via cross product
      V(0,0) = rm(0,k3)*rk2(0) - rm(1,k3)*rk(0);
      V(1,0) = rm(0,k3)*rk2(1) - rm(1,k3)*rk(1);
      V(2,0) = rm(0,k3)*rk2(2) - rm(1,k3)*rk(2);

      // normalize
      mag = std::sqrt(V(0,0)*V(0,0) + V(1,0)*V(1,0) + V(2,0)*V(2,0));
      V(0,0) /= mag;
      V(1,0) /= mag;
      V(2,0) /= mag;

      // set last eigenvector as cross product of other two
      V(0,1) = V(1,0)*V(2,2) - V(2,0)*V(1,2);
      V(1,1) = V(2,0)*V(0,2) - V(0,0)*V(2,2);
      V(2,1) = V(0,0)*V(1,2) - V(1,0)*V(0,2);

      // normalize
      mag = std::sqrt(V(0,1)*V(0,1) + V(1,1)*V(1,1) + V(2,1)*V(2,1));
      V(0,1) /= mag;
      V(1,1) /= mag;
      V(2,1) /= mag;

      // add back in the offset
      for (int i(0); i < 3; ++i)
        D(i,i) += trA;
    }

    return std::make_pair(V,D);
  }

  //
  // 4th-order tensor 2nd-order tensor double dot product
  // \param A 4th-order tensor
  // \param B 2nd-order tensor
  // \return 2nd-order tensor \f$ A:B \f$ as \f$ C_{ij}=A_{ijkl}B_{kl} \f$
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
  // \param B 2nd-order tensor
  // \param A 4th-order tensor
  // \return 2nd-order tensor \f$ B:A \f$ as \f$ C_{kl}=A_{ijkl}B_{ij} \f$
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
  // Vector input
  // \param u vector
  // \param is input stream
  // \return is input stream
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
  // \param u vector
  // \param os output stream
  // \return os output stream
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
  // \param A tensor
  // \param is input stream
  // \return is input stream
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
  // \param A tensor
  // \param os output stream
  // \return os output stream
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
  // 2nd-order tensor 2nd-order tensor tensor product
  // \param A 2nd-order tensor
  // \param B 2nd-order tensor
  // \return \f$ A \otimes B \f$
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
  // odot operator useful for \f$ \frac{\partial A^{-1}}{\partial A} \f$
  // see Holzapfel eqn 6.165
  // \param A 2nd-order tensor
  // \param B 2nd-order tensor
  // \return \f$ A \odot B \f$ which is
  // \f$ C_{ijkl} = \frac{1}{2}(A_{ik} B_{jl} + A_{il} B_{jk}) \f$
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
  // \param s all components set to this scalar
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
  // Copy constructor
  // 3rd-order tensor constructor from 3rd-order tensor
  // \param A from which components are copied
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
  // 3rd-order tensor simple destructor
  //
  template<typename ScalarT>
  Tensor3<ScalarT>::~Tensor3()
  {
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
  // \param A added to this tensor
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
  // \param A substracted from this tensor
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
  // Fill 3rd-order tensor with zeros
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
  // 3rd-order tensor addition
  // \param A 3rd-order tensor
  // \param B 3rd-order tensor
  // \return \f$ A + B \f$
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
  // \param A 3rd-order tensor
  // \param B 3rd-order tensor
  // \return \f$ A - B \f$
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
  // 3rd-order tensor minus
  // \return \f$ -A \f$
  //
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator-(Tensor3<ScalarT> const & A)
  {
    Tensor3<ScalarT> S;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
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
  // Tested by components
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B)
  {
    return !(A==B);
  }

  //
  // Scalar 3rd-order tensor product
  // \param s scalar
  // \param A 3rd-order tensor
  // \return \f$ s A \f$
  //
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator*(const ScalarT s, Tensor3<ScalarT> const & A)
  {
    Tensor3<ScalarT> B;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
	for (Index k = 0; k < MaxDim; ++k) {
	  B(i,j,k) = s * A(i,j,k);
	}
      }
    }

    return B;
  }

  //
  // 3th-order tensor scalar product
  // \param A 3th-order tensor
  // \param s scalar
  // \return \f$ s A \f$
  //
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator*(Tensor3<ScalarT> const & A, const ScalarT s)
  {
    return s * A;
  }

  //
  // 3th-order tensor vector product
  // \param A 3th-order tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  dot(Tensor3<ScalarT> const & A, Vector<ScalarT> const & u)
  {
    Tensor<ScalarT> B;

    for (Index j = 0; j < MaxDim; ++j) {
      for (Index k = 0; k < MaxDim; ++k) {
	B(j,k) = 0.0;
	for (Index i = 0; i < MaxDim; ++i) {
	  B(j,k) += A(i,j,k) * u(i);
	}
      }
    }

    return B;
  }

  //
  // vector 3th-order tensor product
  // \param A 3th-order tensor
  // \param u vector
  // \return \f$ u A \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  dot(Vector<ScalarT> const & u, Tensor3<ScalarT> const & A)
  {
    Tensor<ScalarT> B;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
	B(i,j) = 0.0;
	for (Index k = 0; k < MaxDim; ++k) {
	  B(i,j) += A(i,j,k) * u(k);
	}
      }
    }

    return B;
  }


  //
  // 3th-order tensor vector product2 (contract 2nd index)
  // \param A 3th-order tensor
  // \param u vector
  // \return \f$ A u \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  dot2(Tensor3<ScalarT> const & A, Vector<ScalarT> const & u)
  {
    Tensor<ScalarT> B;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index k = 0; k < MaxDim; ++k) {
	B(i,k) = 0.0;
	for (Index j = 0; j < MaxDim; ++j) {
	  B(i,k) += A(i,j,k) * u(j);
	}
      }
    }

    return B;
  }

  //
  // vector 3th-order tensor product2 (contract 2nd index)
  // \param A 3th-order tensor
  // \param u vector
  // \return \f$ u A \f$
  //
  template<typename ScalarT>
  Tensor<ScalarT>
  dot2(Vector<ScalarT> const & u, Tensor3<ScalarT> const & A)
  {
    return dot2(A, u);
  }


  //
  // 3rd-order tensor input
  // \param A 3rd-order tensor
  // \param is input stream
  // \return is input stream
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
  // \param A 3rd-order tensor
  // \param os output stream
  // \return os output stream
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
  // \param s all components set to this scalar
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
  // Copy constructor
  // 4th-order tensor constructor with 4th-order tensor
  // \param A from which components are copied
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
  // 4th-order tensor simple destructor
  //
  template<typename ScalarT>
  Tensor4<ScalarT>::~Tensor4()
  {
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
  // \param A added to this tensor
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
  // \param A substracted from this tensor
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
  // Fill 4th-order tensor with zeros
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
  // 4th-order tensor addition
  // \param A 4th-order tensor
  // \param B 4th-order tensor
  // \return \f$ A + B \f$
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
  // \param A 4th-order tensor
  // \param B 4th-order tensor
  // \return \f$ A - B \f$
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
  // 4th-order tensor minus
  // \return \f$ -A \f$
  //
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator-(Tensor4<ScalarT> const & A)
  {
    Tensor4<ScalarT> S;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            S(i,j,k,l) = - A(i,j,k,l);
          }
        }
      }
    }

    return S;
  }

  //
  // 4th-order identity I1
  // \return \f$ \delta_{ik} \delta_{jl} \f$ such that \f$ A = I_1 A \f$
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
  // 4th-order identity I2
  // \return \f$ \delta_{il} \delta_{jk} \f$ such that \f$ A^T = I_2 A \f$
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
  // 4th-order identity I3
  // \return \f$ \delta_{ij} \delta_{kl} \f$ such that \f$ I_A I = I_3 A \f$
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
  // Tested by components
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
  // Tested by components
  //
  template<typename ScalarT>
  inline bool
  operator!=(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B)
  {
    return !(A==B);
  }

  //
  // Scalar 4th-order tensor product
  // \param s scalar
  // \param A 4th-order tensor
  // \return \f$ s A \f$
  //
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator*(const ScalarT s, Tensor4<ScalarT> const & A)
  {
    Tensor4<ScalarT> B;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
          for (Index l = 0; l < MaxDim; ++l) {
            B(i,j,k,l)=s * A(i,j,k,l);
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
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator*(Tensor4<ScalarT> const & A, const ScalarT s)
  {
    return s * A;
  }

  //
  // 4th-order tensor vector dot product
  // \param A 4th-order tensor
  // \param u vector
  // \return 3rd-order tensor \f$ A dot u \f$ as \f$ B_{ijk}=A_{ijkl}u_{l} \f$
  //
  template<typename ScalarT>
  Tensor3<ScalarT>
  dot(Tensor4<ScalarT> const & A, Vector<ScalarT> const & u)
  {
    Tensor3<ScalarT> B;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index k = 0; k < MaxDim; ++k) {
	  B(i,j,k) = 0.0;
          for (Index l = 0; l < MaxDim; ++l) {
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
  template<typename ScalarT>
  Tensor3<ScalarT>
  dot(Vector<ScalarT> const & u, Tensor4<ScalarT> const & A)
  {
    Tensor3<ScalarT> B;

    for (Index j = 0; j < MaxDim; ++j) {
      for (Index k = 0; k < MaxDim; ++k) {
        for (Index l = 0; l < MaxDim; ++l) {
	  B(j,k,l) = 0.0;
          for (Index i = 0; i < MaxDim; ++i) {
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
  template<typename ScalarT>
  Tensor3<ScalarT>
  dot2(Tensor4<ScalarT> const & A, Vector<ScalarT> const & u)
  {
    Tensor3<ScalarT> B;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index j = 0; j < MaxDim; ++j) {
        for (Index l = 0; l < MaxDim; ++l) {
	  B(i,j,l) = 0.0;
          for (Index k = 0; k < MaxDim; ++k) {
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
  template<typename ScalarT>
  Tensor3<ScalarT>
  dot2(Vector<ScalarT> const & u, Tensor4<ScalarT> const & A)
  {
    Tensor3<ScalarT> B;

    for (Index i = 0; i < MaxDim; ++i) {
      for (Index k = 0; k < MaxDim; ++k) {
        for (Index l = 0; l < MaxDim; ++l) {
	  B(i,k,l) = 0.0;
          for (Index j = 0; j < MaxDim; ++j) {
            B(i,k,l) = u(j) * A(i,j,k,l);
          }
        }
      }
    }
    return B;
  }

  //
  // 4th-order input
  // \param A 4th-order tensor
  // \param is input stream
  // \return is input stream
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
  // \param A 4th-order tensor
  // \param os output stream
  // \return os output stream
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
