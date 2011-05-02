///
/// \file Tensor.h
/// First cut of LCM small tensor utilities. Declarations.
/// \author Alejandro Mota
/// \author Jake Ostien
///
#if !defined(LCM_Tensor_h)
#define LCM_Tensor_h

#include <iostream>
#include <boost/tuple/tuple.hpp>

namespace LCM {

  ///
  /// Indexing type
  ///
  typedef unsigned int Index;

  ///
  /// Maximum spatial dimension
  ///
  const Index MaxDim = 3;

  ///
  /// Vector in R^3
  ///
  template<typename ScalarT>
  class Vector
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Vector();

    ///
    /// Create vector from a scalar
    /// \param s all components are set equal to this value
    ///
    Vector(const ScalarT s);

    ///
    /// Create vector specifying components
    /// \param s0
    /// \param s1
    /// \param s2 are the vector components in the canonical basis
    ///
    Vector(const ScalarT s0, const ScalarT s1, const ScalarT s2);

    ///
    /// Copy constructor
    /// \param v the values of its componets are copied to the new vector
    ///
    Vector(Vector const & v);

    ///
    /// Simple destructor
    ///
    ~Vector();

    ///
    /// Indexing for constant vector
    /// \param i the index
    ///
    const ScalarT &
    operator()(const Index i) const;

    ///
    /// Vector indexing
    /// \param i the index
    ///
    ScalarT &
    operator()(const Index i);

    ///
    /// Copy assignment
    /// \param v the values of its componets are copied to this vector
    ///
    Vector<ScalarT> &
    operator=(Vector<ScalarT> const & v);

    ///
    /// Vector increment
    /// \param v added to currrent vector
    ///
    Vector<ScalarT> &
    operator+=(Vector<ScalarT> const & v);

    ///
    /// Vector decrement
    /// \param v substracted from current vector
    ///
    Vector<ScalarT> &
    operator-=(Vector<ScalarT> const & v);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Vector components
    ///
    ScalarT
    e[MaxDim];

  };

  ///
  /// Second order tensor
  ///
  template<typename ScalarT>
  class Tensor
  {
  public:

    ///
    /// Default constructor that initializes to NaNs
    ///
    Tensor();

    ///
    /// Create tensor from a scalar
    /// \param s all components are set equal to this value
    ///
    Tensor(const ScalarT s);

    ///
    /// Create tensor specifying components
    /// The parameters are the components in the canonical basis
    /// \param s00 ...
    /// \param s22
    ///
    Tensor(
        const ScalarT s00, const ScalarT s01, const ScalarT s02,
        const ScalarT s10, const ScalarT s11, const ScalarT s12,
        const ScalarT s20, const ScalarT s21, const ScalarT s22);

    ///
    /// Copy constructor
    /// \param A the values of its componets are copied to the new tensor
    ///
    Tensor(const Tensor & A);

    ///
    /// Simple destructor
    ///
    ~Tensor();

    ///
    /// Indexing for constant tensor
    /// \param i index
    /// \param j index
    ///
    const ScalarT &
    operator()(const Index i, const Index j) const;

    ///
    /// Tensor indexing
    /// \param i index
    /// \param j index
    ///
    ScalarT &
    operator()(const Index i, const Index j);

    ///
    /// Copy assignment
    /// \param A the values of its componets are copied to this tensor
    ///
    Tensor<ScalarT> &
    operator=(Tensor<ScalarT> const & A);

    ///
    /// Tensor increment
    /// \param A added to current tensor
    ///
    Tensor<ScalarT> &
    operator+=(Tensor<ScalarT> const & A);

    ///
    /// Tensor decrement
    /// \param A substracted from current tensor
    ///
    Tensor<ScalarT> &
    operator-=(Tensor<ScalarT> const & A);

    ///
    /// Fill with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    ScalarT
    e[MaxDim][MaxDim];

  };

  ///
  /// Third order tensor
  ///
  template<typename ScalarT>
  class Tensor3
  {
  public:

    ///
    /// 3rd-order tensor constructor with NaNs
    ///
    Tensor3();

    ///
    /// 3rd-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor3(const ScalarT s);

    ///
    /// Copy constructor
    /// 3rd-order tensor constructor from 3rd-order tensor
    /// \param A from which components are copied
    ///
    Tensor3(const Tensor3 & A);

    ///
    /// 3rd-order tensor simple destructor
    ///
    ~Tensor3();

    ///
    /// Indexing for constant 3rd order tensor
    /// \param i index
    /// \param j index
    /// \param k index
    ///
    const ScalarT &
    operator()(const Index i, const Index j, const Index k) const;

    ///
    /// 3rd-order tensor indexing
    /// \param i index
    /// \param j index
    /// \param k index
    ///
    ScalarT &
    operator()(const Index i, const Index j, const Index k);

    ///
    /// 3rd-order tensor copy assignment
    ///
    Tensor3<ScalarT> &
    operator=(Tensor3<ScalarT> const & A);

    ///
    /// 3rd-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor3<ScalarT> &
    operator+=(Tensor3<ScalarT> const & A);

    ///
    /// 3rd-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor3<ScalarT> &
    operator-=(Tensor3<ScalarT> const & A);

    ///
    /// Fill 3rd-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    ScalarT
    e[MaxDim][MaxDim][MaxDim];

  };

  ///
  /// Fourth order tensor
  ///
  template<typename ScalarT>
  class Tensor4
  {
  public:

    ///
    /// 4th-order tensor constructor with NaNs
    ///
    Tensor4();

    ///
    /// 4th-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor4(const ScalarT s);

    ///
    /// Copy constructor
    /// 4th-order tensor constructor with 4th-order tensor
    /// \param A from which components are copied
    ///
    Tensor4(const Tensor4 & A);

    ///
    /// 4th-order tensor simple destructor
    ///
    ~Tensor4();

    ///
    /// Indexing for constant 4th order tensor
    /// \param i index
    /// \param j index
    /// \param k index
    /// \param l index
    ///
    const ScalarT &
    operator()(
        const Index i,
        const Index j,
        const Index k,
        const Index l) const;

    ///
    /// 4th-order tensor indexing
    /// \param i index
    /// \param j index
    /// \param k index
    /// \param l index
    ///
    ScalarT &
    operator()(
        const Index i,
        const Index j,
        const Index k,
        const Index l);

    ///
    /// 4th-order tensor copy assignment
    ///
    Tensor4<ScalarT> &
    operator=(Tensor4<ScalarT> const & A);

    ///
    /// 4th-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor4<ScalarT> &
    operator+=(Tensor4<ScalarT> const & A);

    ///
    /// 4th-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor4<ScalarT> &
    operator-=(Tensor4<ScalarT> const & A);

    ///
    /// Fill 4th-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor components
    ///
    ScalarT
    e[MaxDim][MaxDim][MaxDim][MaxDim];

  };

  //
  // Prototypes for utilities
  //

  ///
  /// Vector addition
  /// \param u
  /// \param v the operands
  /// \return \f$ u + v \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator+(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Vector substraction
  /// \param u
  /// \param v the operands
  /// \return \f$ u - v \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator-(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Vector minus
  /// \param u
  /// \return \f$ -u \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator-(Vector<ScalarT> const & u);

  ///
  /// Vector dot product
  /// \param u
  /// \param v the operands
  /// \return \f$ u \cdot v \f$
  ///
  template<typename ScalarT>
  ScalarT
  operator*(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Vector equality tested by components
  /// \param u
  /// \param v the operands
  /// \return \f$ u \equiv v \f$
  ///
  template<typename ScalarT>
  bool
  operator==(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Vector inequality tested by components
  /// \param u
  /// \param v the operands
  /// \return \f$ u \neq v \f$
  ///
  template<typename ScalarT>
  bool
  operator!=(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Scalar vector product
  /// \param s scalar factor
  /// \param u vector factor
  /// \return \f$ s u \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(const ScalarT s, Vector<ScalarT> const & u);

  ///
  /// Vector scalar product
  /// \param u vector factor
  /// \param s scalar factor
  /// \return \f$ s u \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, const ScalarT s);

  ///
  /// Vector scalar division
  /// \param u vector
  /// \param s scalar that divides each component of vector
  /// \return \f$ u / s \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator/(Vector<ScalarT> const & u, const ScalarT s);

  ///
  /// Vector dot product
  /// \param u
  /// \param v operands
  /// \return \f$ u \cdot v \f$
  ///
  template<typename ScalarT>
  ScalarT
  dot(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Cross product
  /// \param u
  /// \param v operands
  /// \return \f$ u \times v \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  cross(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Vector 2-norm
  /// \return \f$ \sqrt{u \cdot u} \f$
  ///
  template<typename ScalarT>
  ScalarT
  norm(Vector<ScalarT> const & u);

  ///
  /// Vector 1-norm
  /// \return \f$ |u_0|+|u_1|+|u_2| \f$
  ///
  template<typename ScalarT>
  ScalarT
  norm_1(Vector<ScalarT> const & u);

  ///
  /// Vector infinity-norm
  /// \return \f$ \max(|u_0|,|u_1|,|u_2|) \f$
  ///
  template<typename ScalarT>
  ScalarT
  norm_infinity(Vector<ScalarT> const & u);

  ///
  /// Tensor addition
  /// \return \f$ A + B \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  operator+(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Tensor substraction
  /// \return \f$ A - B \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  operator-(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  operator-(Tensor<ScalarT> const & A);

  ///
  /// Tensor equality
  /// Tested by components
  /// \return \f$ A \equiv B \f$
  ///
  template<typename ScalarT>
  bool
  operator==(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Tensor inequality
  /// Tested by components
  /// \return \f$ A \neq B \f$
  ///
  template<typename ScalarT>
  bool
  operator!=(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Scalar tensor product
  /// \param s scalar
  /// \param A tensor
  /// \return \f$ s A \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(const ScalarT s, Tensor<ScalarT> const & A);

  ///
  /// Tensor scalar product
  /// \param A tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, const ScalarT s);

  ///
  /// Tensor vector product v = A u
  /// \param A tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  dot(Tensor<ScalarT> const & A, Vector<ScalarT> const & u);

  ///
  /// Vector tensor product v = u A
  /// \param A tensor
  /// \param u vector
  /// \return \f$ u A = A^T u \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  dot(Vector<ScalarT> const & u, Tensor<ScalarT> const & A);

  ///
  /// Tensor vector product v = A u
  /// \param A tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Tensor<ScalarT> const & A, Vector<ScalarT> const & u);

  ///
  /// Vector tensor product v = u A
  /// \param A tensor
  /// \param u vector
  /// \return \f$ u A = A^T u \f$
  ///
  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, Tensor<ScalarT> const & A);

  ///
  /// Tensor dot product C = A B
  /// \return \f$ A \cdot B \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Tensor tensor product C = A B
  /// \param A tensor
  /// \param B tensor
  /// \return a tensor \f$ A \cdot B \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  dot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Tensor tensor double dot product (contraction)
  /// \param A tensor
  /// \param B tensor
  /// \return a scalar \f$ A : B \f$
  ///
  template<typename ScalarT>
  ScalarT
  dotdot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Tensor Frobenius norm
  /// \return \f$ \sqrt{A:A} \f$
  ///
  template<typename ScalarT>
  ScalarT
  norm(Tensor<ScalarT> const & A);

  ///
  /// Tensor 1-norm
  /// \return \f$ \max_{j \in {0,1,2}}\Sigma_{i=0}^2 |A_{ij}| \f$
  ///
  template<typename ScalarT>
  ScalarT
  norm_1(Tensor<ScalarT> const & A);

  ///
  /// Tensor infinity-norm
  /// \return \f$ \max_{i \in {0,1,2}}\Sigma_{j=0}^2 |A_{ij}| \f$
  ///
  template<typename ScalarT>
  ScalarT
  norm_infinity(Tensor<ScalarT> const & A);

  ///
  /// Dyad
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  dyad(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Bun operator, just for Jay
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  bun(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Tensor product
  /// \param u vector
  /// \param v vector
  /// \return \f$ u \otimes v \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  tensor(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  ///
  /// Zero 2nd-order tensor
  /// All components are zero
  ///
  template<typename ScalarT>
  const Tensor<ScalarT>
  zero();

  ///
  /// 2nd-order identity tensor
  ///
  template<typename ScalarT>
  const Tensor<ScalarT>
  identity();

  ///
  /// 2nd-order identity tensor, à la Matlab
  ///
  template<typename ScalarT>
  const Tensor<ScalarT>
  eye();

  ///
  /// 2nd-order tensor transpose
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  transpose(Tensor<ScalarT> const & A);

  ///
  /// Symmetric part of 2nd-order tensor
  /// \return \f$ \frac{1}{2}(A + A^T) \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  symm(Tensor<ScalarT> const & A);

  ///
  /// Skew symmetric part of 2nd-order tensor
  /// \return \f$ \frac{1}{2}(A - A^T) \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  skew(Tensor<ScalarT> const & A);

  ///
  /// Skew symmetric 2nd-order tensor from vector
  /// \param u vector
  /// \return \f$ {{0, -u_2, u_1}, {u_2, 0, -u_0}, {-u_1, u+0, 0}} \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  skew(Vector<ScalarT> const & u);

  ///
  /// 2nd-order tensor inverse
  /// \param A nonsingular tensor
  /// \return \f$ A^{-1} \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  inverse(Tensor<ScalarT> const & A);

  ///
  /// Determinant
  /// \param A tensor
  /// \return \f$ \det A \f$
  ///
  template<typename ScalarT>
  ScalarT
  det(Tensor<ScalarT> const & A);

  ///
  /// Trace
  /// \param A tensor
  /// \return \f$ A:I \f$
  ///
  template<typename ScalarT>
  ScalarT
  trace(Tensor<ScalarT> const & A);

  ///
  /// First invariant, trace
  /// \param A tensor
  /// \return \f$ I_A = A:I \f$
  ///
  template<typename ScalarT>
  ScalarT
  I1(Tensor<ScalarT> const & A);

  ///
  /// Second invariant
  /// \param A tensor
  /// \return \f$ II_A = \frac{1}{2}((I_A)^2-I_{A^2}) \f$
  ///
  template<typename ScalarT>
  ScalarT
  I2(Tensor<ScalarT> const & A);

  ///
  /// Third invariant
  /// \param A tensor
  /// \return \f$ III_A = \det A \f$
  ///
  template<typename ScalarT>
  ScalarT
  I3(Tensor<ScalarT> const & A);

  ///
  /// Exponential map by Taylor series, radius of convergence is infinity
  /// \param A tensor
  /// \return \f$ \exp A \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  exp(Tensor<ScalarT> const & A);

  ///
  /// Logarithmic map by Taylor series, converges for \f$ |A-I| < 1 \f$
  /// \param A tensor
  /// \return \f$ \log A \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  log(Tensor<ScalarT> const & A);

  ///
  /// Logarithmic map of a rotation
  /// \param R with \f$ R \in SO(3) \f$
  /// \return \f$ r = \log R \f$ with \f$ r \in so(3) \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  log_rotation(Tensor<ScalarT> const & R);

  ///
  /// Left polar decomposition
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD
  ///
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>, Tensor<ScalarT> >
  polar_left(Tensor<ScalarT> const & F);

  ///
  /// Right polar decomposition
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ RU = F \f$ with \f$ R \in SO(3) \f$ and U SPD
  ///
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>, Tensor<ScalarT> >
  polar_right(Tensor<ScalarT> const & F);

  ///
  /// Left polar decomposition with matrix logarithm for V
  /// \param F tensor (often a deformation-gradient-like tensor)
  /// \return \f$ VR = F \f$ with \f$ R \in SO(3) \f$ and V SPD, and log V
  ///
  template<typename ScalarT>
  boost::tuple<Tensor<ScalarT>, Tensor<ScalarT>, Tensor<ScalarT> >
  polar_left_logV(Tensor<ScalarT> const & F);

  ///
  /// Logarithmic map using BCH expansion (3 terms)
  /// \param v tensor
  /// \param r tensor
  /// \return Baker-Campbell-Hausdorff series up to 3 terms
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  bch(Tensor<ScalarT> const & v, Tensor<ScalarT> const & r);

  ///
  /// Eigenvalue decomposition for SPD 2nd-order tensor
  /// \param A tensor
  /// \return V eigenvectors, D eigenvalues in diagonal Matlab-style
  ///
  template<typename ScalarT>
  std::pair<Tensor<ScalarT>, Tensor<ScalarT> >
  eig_spd(Tensor<ScalarT> const & A);

  ///
  /// 4th-order identity I1
  /// \return \f$ \delta_{ik} \delta_{jl} \f$ such that \f$ A = I_1 A \f$
  ///
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_1();

  ///
  /// 4th-order identity I2
  /// \return \f$ \delta_{il} \delta_{jk} \f$ such that \f$ A^T = I_2 A \f$
  ///
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_2();

  ///
  /// 4th-order identity I3
  /// \return \f$ \delta_{ij} \delta_{kl} \f$ such that \f$ I_A I = I_3 A \f$
  ///
  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_3();

  ///
  /// 4th-order tensor 2nd-order tensor double dot product
  /// \param A 4th-order tensor
  /// \param B 2nd-order tensor
  /// \return 2nd-order tensor \f$ A:B \f$ as \f$ C_{ij}=A_{ijkl}B_{kl} \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor4<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// 2nd-order tensor 4th-order tensor double dot product
  /// \param B 2nd-order tensor
  /// \param A 4th-order tensor
  /// \return 2nd-order tensor \f$ B:A \f$ as \f$ C_{kl}=A_{ijkl}B_{ij} \f$
  ///
  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor<ScalarT> const & B, Tensor4<ScalarT> const & A);

  ///
  /// 2nd-order tensor 2nd-order tensor tensor product
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \otimes B \f$
  ///
  template<typename ScalarT>
  Tensor4<ScalarT>
  tensor(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// odot operator useful for \f$ \frac{\partial A^{-1}}{\partial A} \f$
  /// see Holzapfel eqn 6.165
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \odot B \f$ which is
  /// \f$ C_{ijkl} = \frac{1}{2}(A_{ik} B_{jl} + A_{il} B_{jk}) \f$
  ///
  template<typename ScalarT>
  Tensor4<ScalarT>
  odot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  ///
  /// Vector input
  /// \param u vector
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Vector<ScalarT> & u);

  ///
  /// Vector output
  /// \param u vector
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Vector<ScalarT> const & u);

  ///
  /// Tensor input
  /// \param A tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor<ScalarT> & A);

  ///
  /// Tensor output
  /// \param A tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor<ScalarT> const & A);

  ///
  /// 3rd-order tensor input
  /// \param A 3rd-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor3<ScalarT> & A);

  ///
  /// 3rd-order tensor output
  /// \param A 3rd-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<ScalarT> const & A);

  ///
  /// 4th-order input
  /// \param A 4th-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor4<ScalarT> & A);

  ///
  /// 4th-order output
  /// \param A 4th-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<ScalarT> const & A);

  ///
  /// 3rd-order tensor addition
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator+(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  ///
  /// 3rd-order tensor substraction
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator-(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  ///
  /// 3rd-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename ScalarT>
  Tensor3<ScalarT>
  operator-(Tensor3<ScalarT> const & A);

  ///
  /// 3rd-order tensor equality
  /// Tested by components
  ///
  template<typename ScalarT>
  bool
  operator==(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  ///
  /// 3rd-order tensor inequality
  /// Tested by components
  ///
  template<typename ScalarT>
  bool
  operator!=(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  ///
  /// 4th-order tensor addition
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator+(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  ///
  /// 4th-order tensor substraction
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator-(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  ///
  /// 4th-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename ScalarT>
  Tensor4<ScalarT>
  operator-(Tensor4<ScalarT> const & A);

  ///
  /// 4th-order equality
  /// Tested by components
  ///
  template<typename ScalarT>
  bool
  operator==(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  ///
  /// 4th-order inequality
  /// Tested by components
  ///
  template<typename ScalarT>
  bool
  operator!=(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

} // namespace LCM

#include "Tensor.i.cc"
#include "Tensor.t.cc"

#endif //LCM_Tensor_h
