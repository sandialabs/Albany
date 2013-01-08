//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor4_i_cc)
#define tensor_Tensor4_i_cc

namespace LCM
{

  //
  // R^N 4th-order tensor transpose
  // per Holzapfel 1.157
  //
  template<typename T>
  inline Tensor4<T> transpose(Tensor4<T> const & A)
  {
    const Index N = A.get_dimension();

    Tensor4<T> B(N);

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        for (Index k = 0; k < N; ++k) {
          for (Index l = 0; l < N; ++l) {
            B(i, j, k, l) = A(k, l, i, j);
          }
        }
      }
    }

    return B;
  }

  //
  // Dimension
  // \return dimension
  //
  template<typename T>
  inline
  Index
  Tensor4<T>::get_dimension() const
  {
    return e.size();
  }

  //
  // R^N indexing for constant 4th order tensor
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  const T &
  Tensor4<T>::operator()(
      const Index i, const Index j, const Index k, const Index l) const
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    assert(l < get_dimension());
    return e[i][j][k][l];
  }

  //
  // R^N 4th-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  // \param l index
  //
  template<typename T>
  inline
  T &
  Tensor4<T>::operator()(
      const Index i, const Index j, const Index k, const Index l)
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    assert(l < get_dimension());
    return e[i][j][k][l];
  }

} // namespace LCM

#endif // tensor_Tensor4_i_cc
