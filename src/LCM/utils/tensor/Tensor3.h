//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor3_h)
#define tensor_Tensor3_h

#include "Vector.h"
#include "Tensor.h"

namespace LCM {

  ///
  /// Third order tensor in R^N.
  ///
  template<typename T>
  class Tensor3
  {
  public:

    ///
    /// Default constructor
    ///
    Tensor3();

    ///
    /// 3rd-order tensor constructor with NaNs
    ///
    explicit
    Tensor3(Index const N);

    ///
    /// 3rd-order tensor constructor with a scalar
    /// \param s all components set to this scalar
    ///
    Tensor3(Index const N, T const & s);

    ///
    /// Copy constructor
    /// 3rd-order tensor constructor from 3rd-order tensor
    /// \param A from which components are copied
    ///
    Tensor3(Tensor3<T> const & A);

    ///
    /// 3rd-order tensor simple destructor
    ///
    ~Tensor3();

    ///
    /// Indexing for constant 3rd-order tensor
    /// \param i index
    /// \param j index
    /// \param k index
    ///
    T const &
    operator()(Index const i, Index const j, Index const k) const;

    ///
    /// 3rd-order tensor indexing
    /// \param i index
    /// \param j index
    /// \param k index
    ///
    T &
    operator()(Index const i, Index const j, Index const k);

    ///
    /// \return dimension
    ///
    Index
    get_dimension() const;

    ///
    /// \param N dimension of 3rd-order tensor
    ///
    void
    set_dimension(Index const N);

    ///
    /// 3rd-order tensor copy assignment
    ///
    Tensor3<T> &
    operator=(Tensor3<T> const & A);

    ///
    /// 3rd-order tensor increment
    /// \param A added to this tensor
    ///
    Tensor3<T> &
    operator+=(Tensor3<T> const & A);

    ///
    /// 3rd-order tensor decrement
    /// \param A substracted from this tensor
    ///
    Tensor3<T> &
    operator-=(Tensor3<T> const & A);

    ///
    /// Fill 3rd-order tensor with zeros
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
  /// 3rd-order tensor addition
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A + B \f$
  ///
  template<typename T>
  Tensor3<T>
  operator+(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// 3rd-order tensor substraction
  /// \param A 3rd-order tensor
  /// \param B 3rd-order tensor
  /// \return \f$ A - B \f$
  ///
  template<typename T>
  Tensor3<T>
  operator-(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// 3rd-order tensor minus
  /// \return \f$ -A \f$
  ///
  template<typename T>
  Tensor3<T>
  operator-(Tensor3<T> const & A);

  ///
  /// 3rd-order tensor equality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator==(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// 3rd-order tensor inequality
  /// Tested by components
  ///
  template<typename T>
  bool
  operator!=(Tensor3<T> const & A, Tensor3<T> const & B);

  ///
  /// Scalar 3rd-order tensor product
  /// \param s scalar
  /// \param A 3rd-order tensor
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor3<T>
  operator*(S const & s, Tensor3<T> const & A);

  ///
  /// 3rd-order tensor scalar product
  /// \param A 3rd-order tensor
  /// \param s scalar
  /// \return \f$ s A \f$
  ///
  template<typename T, typename S>
  Tensor3<T>
  operator*(Tensor3<T> const & A, S const & s);

  ///
  /// 3rd-order tensor vector product
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T>
  Tensor<T>
  dot(Tensor3<T> const & A, Vector<T> const & u);

  ///
  /// vector 3rd-order tensor product
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ u A \f$
  ///
  template<typename T>
  Tensor<T>
  dot(Vector<T> const & u, Tensor3<T> const & A);

  ///
  /// 3rd-order tensor vector product2 (contract 2nd index)
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ A u \f$
  ///
  template<typename T>
  Tensor<T>
  dot2(Tensor3<T> const & A, Vector<T> const & u);

  ///
  /// vector 3rd-order tensor product2 (contract 2nd index)
  /// \param A 3rd-order tensor
  /// \param u vector
  /// \return \f$ u A \f$
  ///
  template<typename T>
  Tensor<T>
  dot2(Vector<T> const & u, Tensor3<T> const & A);

  ///
  /// 3rd-order tensor input
  /// \param A 3rd-order tensor
  /// \param is input stream
  /// \return is input stream
  ///
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor3<T> & A);

  ///
  /// 3rd-order tensor output
  /// \param A 3rd-order tensor
  /// \param os output stream
  /// \return os output stream
  ///
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<T> const & A);

} // namespace LCM

#include "Tensor3.i.cc"
#include "Tensor3.t.cc"

#endif //tensor_Tensor3_h
