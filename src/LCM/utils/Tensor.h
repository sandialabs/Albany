//
// First cut of LCM small tensor utilities.
//
#if !defined(LCM_Tensor_h)
#define LCM_Tensor_h

#include <iostream>
#include <boost/tuple/tuple.hpp>

namespace LCM {

  typedef unsigned int Index;
  const Index MaxDim = 3;

  //
  // Vector
  //
  template<typename ScalarT>
  class Vector
  {
  public:

    // Constructors
    Vector();
    Vector(const ScalarT s);
    Vector(const ScalarT s0, const ScalarT s1, const ScalarT s2);
    Vector(Vector const & v);

    // Destructor
    ~Vector();

    // Indexing
    const ScalarT & operator()(const Index i) const;
    ScalarT & operator()(const Index i);

    // Copy assignment
    Vector<ScalarT> & operator=  (Vector<ScalarT> const & v);

    // Increment, decrement
    Vector<ScalarT> & operator+= (Vector<ScalarT> const & v);
    Vector<ScalarT> & operator-= (Vector<ScalarT> const & v);

    // Fill with zeros
    void clear();

  private:
    ScalarT e[MaxDim];
  };

  //
  // Second order tensor
  //
  template<typename ScalarT>
  class Tensor
  {
  public:

    // Constructors
    Tensor();
    Tensor(const ScalarT s);
    Tensor(
        const ScalarT s00, const ScalarT s01, const ScalarT s02,
        const ScalarT s10, const ScalarT s11, const ScalarT s12,
        const ScalarT s20, const ScalarT s21, const ScalarT s22);
    Tensor(const Tensor & A);

    // Destructor
    ~Tensor();

    // Indexing
    const ScalarT & operator()(const Index i, const Index j) const;
    ScalarT & operator()(const Index i, const Index j);

    // Copy assignment
    Tensor<ScalarT> & operator=  (Tensor<ScalarT> const & A);

    // Increment, decrement
    Tensor<ScalarT> & operator+= (Tensor<ScalarT> const & A);
    Tensor<ScalarT> & operator-= (Tensor<ScalarT> const & A);

    // Fill with zeros
    void clear();

  private:
    ScalarT e[MaxDim][MaxDim];
  };

  //
  // Third order tensor
  //
  template<typename ScalarT>
  class Tensor3
  {
  public:

    // Constructors
    Tensor3();
    Tensor3(const ScalarT s);
    Tensor3(const Tensor3 & A);

    // Destructor
    ~Tensor3();

    // Indexing
    const ScalarT & operator()(
        const Index i, const Index j, const Index k) const;

    ScalarT & operator()(const Index i, const Index j, const Index k);

    // Copy assignment
    Tensor3<ScalarT> & operator=  (Tensor3<ScalarT> const & A);

    // Increment, decrement
    Tensor3<ScalarT> & operator+= (Tensor3<ScalarT> const & A);
    Tensor3<ScalarT> & operator-= (Tensor3<ScalarT> const & A);

    // Fill with zeros
    void clear();

  private:
    ScalarT e[MaxDim][MaxDim][MaxDim];
  };

  //
  // Fourth order tensor
  //
  template<typename ScalarT>
  class Tensor4
  {
  public:

    // Constructor
    Tensor4();
    Tensor4(const ScalarT s);
    Tensor4(const Tensor4 & A);

    // Destructor
    ~Tensor4();

    // Indexing
    const ScalarT & operator()(
        const Index i, const Index j, const Index k, const Index l) const;

    ScalarT & operator()(
        const Index i, const Index j,
        const Index k, const Index l);

    // Copy assignment
    Tensor4<ScalarT> & operator=  (Tensor4<ScalarT> const & A);

    // Increment, decrement
    Tensor4<ScalarT> & operator+= (Tensor4<ScalarT> const & A);
    Tensor4<ScalarT> & operator-= (Tensor4<ScalarT> const & A);

    // Fill with zeros
    void clear();

  private:
    ScalarT e[MaxDim][MaxDim][MaxDim][MaxDim];
  };

  //
  // Prototypes for utilities
  //

  // Vector addition
  template<typename ScalarT>
  Vector<ScalarT>
  operator+(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Vector substraction
  template<typename ScalarT>
  Vector<ScalarT>
  operator-(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Vector dot product
  template<typename ScalarT>
  ScalarT
  operator*(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Vector equality
  template<typename ScalarT>
  bool
  operator==(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Vector inequality
  template<typename ScalarT>
  bool
  operator!=(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Scalar vector product
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(const ScalarT s, Vector<ScalarT> const & u);

  // Vector scalar product
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, const ScalarT s);

  // Vector scalar division
  template<typename ScalarT>
  Vector<ScalarT>
  operator/(Vector<ScalarT> const & u, const ScalarT s);

  // Vector dot product
  template<typename ScalarT>
  ScalarT
  dot(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Tensor vector product v = A u
  template<typename ScalarT>
  Vector<ScalarT>
  dot(Tensor<ScalarT> const & A, Vector<ScalarT> const & u);

  // Vector tensor product v = u A
  template<typename ScalarT>
  Vector<ScalarT>
  dot(Vector<ScalarT> const & u, Tensor<ScalarT> const & A);

  // Tensor vector product v = A u
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Tensor<ScalarT> const & A, Vector<ScalarT> const & u);

  // Vector tensor product v = u A
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, Tensor<ScalarT> const & A);

  // Cross product
  template<typename ScalarT>
  Vector<ScalarT>
  cross(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Vector 2-norm
  template<typename ScalarT>
  ScalarT
  norm(Vector<ScalarT> const & u);

  // Vector 1-norm
  template<typename ScalarT>
  ScalarT
  norm_1(Vector<ScalarT> const & u);

  // Vector infinity norm
  template<typename ScalarT>
  ScalarT
  norm_infinity(Vector<ScalarT> const & u);

  // Tensor addition
  template<typename ScalarT>
  Tensor<ScalarT>
  operator+(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // Tensor substraction
  template<typename ScalarT>
  Tensor<ScalarT>
  operator-(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // Scalar tensor product
  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(const ScalarT s, Tensor<ScalarT> const & A);

  // Tensor scalar product
  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, const ScalarT s);

  // Tensor dot product C = A B
  template<typename ScalarT>
  Tensor<ScalarT>
  dot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // Tensor dot product C = A B
  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // Tensor equality
  template<typename ScalarT>
  bool
  operator==(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // Tensor inequality
  template<typename ScalarT>
  bool
  operator!=(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // Tensor double dot product (contraction)
  template<typename ScalarT>
  ScalarT
  dotdot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // 4th-order tensor 2nd-order tensor double dot product
  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor4<ScalarT> const & A, Tensor<ScalarT> const & B);

  // 2nd-order tensor 4th-order tensor double dot product
  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor<ScalarT> const & B, Tensor4<ScalarT> const & A);

  // 2nd-order tensor 2nd-order tensor tensor product
  template<typename ScalarT>
  Tensor4<ScalarT>
  tensor(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // odot operator useful for dA(-1)/dA, see Holzapfel eqn 6.165
  // C_ijkl = 1/2 (A_ik B_jl + A_il B_jk)
  template<typename ScalarT>
  Tensor4<ScalarT>
  odot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  // Tensor Frobenius norm
  template<typename ScalarT>
  ScalarT
  norm(Tensor<ScalarT> const & A);

  // Tensor 1-norm
  template<typename ScalarT>
  ScalarT
  norm_1(Tensor<ScalarT> const & A);

  // Tensor infinity norm
  template<typename ScalarT>
  ScalarT
  norm_infinity(Tensor<ScalarT> const & A);

  // Dyad
  template<typename ScalarT>
  Tensor<ScalarT>
  dyad(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Dyad
  template<typename ScalarT>
  Tensor<ScalarT>
  tensor(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // Dyad, just for Jay
  template<typename ScalarT>
  Tensor<ScalarT>
  bun(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  // 2nd-order identity
  template<typename ScalarT>
  const Tensor<ScalarT>
  identity();

  // 2nd-order identity
  template<typename ScalarT>
  const Tensor<ScalarT>
  eye();

  // 2nd-order zero
  template<typename ScalarT>
  const Tensor<ScalarT>
  zero();

  // Transpose
  template<typename ScalarT>
  Tensor<ScalarT>
  transpose(Tensor<ScalarT> const & A);

  // Symmetric part of 2nd-order tensor
  template<typename ScalarT>
  Tensor<ScalarT>
  symm(Tensor<ScalarT> const & A);

  // Skew symmetric part of 2nd-order tensor
  template<typename ScalarT>
  Tensor<ScalarT>
  skew(Tensor<ScalarT> const & A);

  // Skew symmetric 2nd-order tensor from vector
  template<typename ScalarT>
  Tensor<ScalarT>
  skew(Vector<ScalarT> const & u);

  // Tensor inverse
  template<typename ScalarT>
  Tensor<ScalarT>
  inverse(Tensor<ScalarT> const & A);

  // Determinant
  template<typename ScalarT>
  ScalarT
  det(Tensor<ScalarT> const & A);

  // Trace
  template<typename ScalarT>
  ScalarT
  trace(Tensor<ScalarT> const & A);

  // First invariant, trace
  template<typename ScalarT>
  ScalarT
  I1(Tensor<ScalarT> const & A);

  // Second invariant
  template<typename ScalarT>
  ScalarT
  I2(Tensor<ScalarT> const & A);

  // Third invariant, determinat
  template<typename ScalarT>
  ScalarT
  I3(Tensor<ScalarT> const & A);

  // Left polar decomposition
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  polar_left(Tensor<ScalarT> const & F);

  // Right polar decomposition
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  polar_right(Tensor<ScalarT> const & F);

  // Left polar decomposition with matrix logarithm for V
  template<typename ScalarT>
  boost::tuple<Tensor<ScalarT>,Tensor<ScalarT>,Tensor<ScalarT> >
  polar_left_logV(Tensor<ScalarT> const & F);

  // Eigenvalue decomposition for SPD 2nd order tensor
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>,Tensor<ScalarT> >
  eig_spd(Tensor<ScalarT> const & A);

  // Exponential map using Taylor series
  template<typename ScalarT>
  Tensor<ScalarT>
  exp(Tensor<ScalarT> const & A);

  // Logarithmic map using Taylor series
  template<typename ScalarT>
  Tensor<ScalarT>
  log(Tensor<ScalarT> const & A);

  // Logarithmic map of a rotation
  template<typename ScalarT>
  Tensor<ScalarT>
  log_rotation(Tensor<ScalarT> const & R);

  // Logarithmic map using BCH expansion (3 terms)
  template<typename ScalarT>
  Tensor<ScalarT>
  bch(Tensor<ScalarT> const & v, Tensor<ScalarT> const & r);

  // 4th-order identity delta_ik delta_jl, A = I_1 A
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_1();

  // 4th-order identity delta_il delta_jk, A^T = I_2 A
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_2();

  // 4th-order identity delta_ij delta_kl, trA I = I_3 A
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_3();

  // Vector input
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Vector<ScalarT> & u);

  // Vector output
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Vector<ScalarT> const & u);

  // Tensor input
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor<ScalarT> & A);

  // Tensor output
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor<ScalarT> const & A);

  // 3rd-order tensor input
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor3<ScalarT> & A);

  // 3rd-order tensor output
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<ScalarT> const & A);

  // 4th-order tensor input
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor4<ScalarT> & A);

  // 4th-order tensor output
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<ScalarT> const & A);

  // 3rd-order tensor addition
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator+(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  // 3rd-order tensor substraction
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator-(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  // 3rd-order equality
  template<typename ScalarT>
  bool
  operator==(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  // 3rd-order inequality
  template<typename ScalarT>
  bool
  operator!=(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  // 4th-order tensor addition
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator+(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  // 4th-order tensor substraction
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator-(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  // 4th-order equality
  template<typename ScalarT>
  bool
  operator==(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  // 4th-order inequality
  template<typename ScalarT>
  bool
  operator!=(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

} // namespace LCM

#include "Tensor.i.cc"
#include "Tensor.t.cc"

#endif //LCM_Tensor_h
