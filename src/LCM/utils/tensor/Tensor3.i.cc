//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor3_i_cc)
#define tensor_Tensor3_i_cc

namespace LCM {

  //
  // Dimension
  // get dimension
  //
  template<typename T>
  inline
  Index
  Tensor3<T>::get_dimension() const
  {
    return e.size();
  }

  //
  // R^N Indexing for constant 3rd order tensor
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  const T &
  Tensor3<T>::operator()(
    const Index i,
    const Index j,
    const Index k) const
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    return e[i][j][k];
  }

  //
  // R^N 3rd-order tensor indexing
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  T &
  Tensor3<T>::operator()(const Index i, const Index j, const Index k)
  {
    assert(i < get_dimension());
    assert(j < get_dimension());
    assert(k < get_dimension());
    return e[i][j][k];
  }

} // namespace LCM

#endif // tensor_Tensor3_i_cc
