//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor4_h)
#define tensor_Tensor4_h

#include "Vector.h"
#include "Tensor.h"
#include "Tensor3.h"

namespace LCM {

  ///
  /// Fourth order tensor in R^N.
  ///
  template<typename T>
  class Tensor4
  {
  public:

    ///
    /// Default constructor
    ///
    Tensor4();

    ///
    /// 4th-order tensor constructor with NaNs
    ///
    Tensor4(const Index N);

    ///
    /// 4th-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor4(const Index N, T const & s);

    ///
    /// Copy constructor
    /// 4th-order tensor constructor with 4th-order tensor
    /// \param A from which components are copied
    ///
    Tensor4(Tensor4<T> const & A);

    ///
    /// 4th-order tensor simple destructor
    ///
    ~Tensor4();

    ///
    /// Indexing for constant 4th-order tensor
    /// \param i index
    /// \param j index
    /// \param k index
    /// \param l index
    ///
    const T &
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
    T &
    operator()(
        const Index i,
        const Index j,
        const Index k,
        const Index l);

    ///
    /// \return dimension
    ///
    Index
    get_dimension() const;

    ///
    /// \param N dimension of 4th-order tensor
    ///
    void
    set_dimension(const Index N);

    ///
    /// 4th-order tensor copy assignment
    ///
    Tensor4<T> &
    operator=(Tensor4<T> const & A);

    ///
    /// 4th-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor4<T> &
    operator+=(Tensor4<T> const & A);

    ///
    /// 4th-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor4<T> &
    operator-=(Tensor4<T> const & A);

    ///
    /// Fill 4th-order tensor with zeros
    ///
    void
    clear();

  private:

    ///
    /// Tensor dimension
    ///
    Index
    dimension;

    ///
    /// Tensor components
    ///
    T *
    e;

  };

  ///
  /// 4th-order tensor addition
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename T>
  Tensor4<T>
  operator+(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// 4th-order tensor substraction
  /// \param A 4th-order tensor
  /// \param B 4th-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename T>
  Tensor4<T>
  operator-(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// 4th-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T>
  Tensor4<T>
  operator-(Tensor4<T> const & A);

  ///
  /// 4th-order equality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator==(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// 4th-order inequality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator!=(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// Scalar 4th-order tensor product
  /// \param s scalar
  /// \param A 4th-order tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor4<T>
  operator*(S const & s, Tensor4<T> const & A);

  ///
  /// 4th-order tensor scalar product
  /// \param A 4th-order tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor4<T>
  operator*(Tensor4<T> const & A, S const & s);


  /// Tensor4 Tensor4 double dot product
  /// \param A Tensor4
  /// \param B Tensor4
  /// \return a Tensor4 \f$ C_{ijkl} = A_{ijmn} : B){mnkl} \f$
  template<typename T>
  Tensor4<T>
  dotdot(Tensor4<T> const & A, Tensor4<T> const & B);

  ///
  /// 4th-order tensor transpose
  ///
  template<typename T>
  Tensor4<T>
  transpose(Tensor4<T> const & A);

  ///
  /// 4th-order identity I1
  /// \return \f$ \delta_{ik} \delta_{jl} \f$ such that \f$ A = I_1 A \f$
  ///
  template<typename T>
  const Tensor4<T>
  identity_1(const Index N);

  ///
  /// 4th-order identity I2
  /// \return \f$ \delta_{il} \delta_{jk} \f$ such that \f$ A^T = I_2 A \f$
  ///
  template<typename T>
  const Tensor4<T>
  identity_2(const Index N);

  ///
  /// 4th-order identity I3
  /// \return \f$ \delta_{ij} \delta_{kl} \f$ such that \f$ I_A I = I_3 A \f$
  ///
  template<typename T>
  const Tensor4<T>
  identity_3(const Index N);

  ///
  /// 4th-order tensor vector dot product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ A dot u \f$ as \f$ B_{ijk}=A_{ijkl}u_{l} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot(Tensor4<T> const & A, Vector<T> const & u);

  ///
  /// vector 4th-order tensor dot product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ u dot A \f$ as \f$ B_{jkl}=u_{i} A_{ijkl} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot(Vector<T> const & u, Tensor4<T> const & A);

  ///
  /// 4th-order tensor vector dot2 product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ A dot2 u \f$ as \f$ B_{ijl}=A_{ijkl}u_{k} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot2(Tensor4<T> const & A, Vector<T> const & u);

  ///
  /// vector 4th-order tensor dot2 product
  /// \param A 4th-order tensor
  /// \param u vector
  /// \return 3rd-order tensor \f$ u dot2 A \f$ as \f$ B_{ikl}=u_{j}A_{ijkl} \f$
  ///
  template<typename T>
  Tensor3<T>
  dot2(Vector<T> const & u, Tensor4<T> const & A);

  ///
  /// 4th-order tensor 2nd-order tensor double dot product
  /// \param A 4th-order tensor
  /// \param B 2nd-order tensor
  /// \return 2nd-order tensor \f$ A:B \f$ as \f$ C_{ij}=A_{ijkl}B_{kl} \f$
  ///
  template<typename T>
  Tensor<T>
  dotdot(Tensor4<T> const & A, Tensor<T> const & B);

  ///
  /// 2nd-order tensor 4th-order tensor double dot product
  /// \param B 2nd-order tensor
  /// \param A 4th-order tensor
  /// \return 2nd-order tensor \f$ B:A \f$ as \f$ C_{kl}=A_{ijkl}B_{ij} \f$
  ///
  template<typename T>
  Tensor<T>
  dotdot(Tensor<T> const & B, Tensor4<T> const & A);

  ///
  /// 2nd-order tensor 2nd-order tensor tensor product
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \otimes B \f$
  ///
  template<typename T>
  Tensor4<T>
  tensor(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// odot operator useful for \f$ \frac{\partial A^{-1}}{\partial A} \f$
  /// see Holzapfel eqn 6.165
  /// \param A 2nd-order tensor
  /// \param B 2nd-order tensor
  /// \return \f$ A \odot B \f$ which is
  /// \f$ C_{ijkl} = \frac{1}{2}(A_{ik} B_{jl} + A_{il} B_{jk}) \f$
  ///
  template<typename T>
  Tensor4<T>
  odot(Tensor<T> const & A, Tensor<T> const & B);

  ///
  /// 4th-order input
  /// \param A 4th-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor4<T> & A);

  ///
  /// 4th-order output
  /// \param A 4th-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<T> const & A);

} // namespace LCM

#include "Tensor4.i.cc"
#include "Tensor4.t.cc"

#endif //tensor_Tensor4_h
