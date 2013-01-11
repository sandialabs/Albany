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
    return dimension;
  }

  //
  // R^N Indexing for constant 3rd order tensor
  // \param i index
  // \param j index
  // \param k index
  //
  template<typename T>
  inline
  T const &
  Tensor3<T>::operator()(
    Index const i,
    Index const j,
    Index const k) const
  {
    Index const
    N = get_dimension();

    assert(i < N);
    assert(j < N);
    assert(k < N);

    return e[(i * N + j) * N + k];
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
  Tensor3<T>::operator()(Index const i, Index const j, Index const k)
  {
    Index const
    N = get_dimension();

    assert(i < N);
    assert(j < N);
    assert(k < N);

    return e[i * N * N + j * N + k];
  }

} // namespace LCM

#endif // tensor_Tensor3_i_cc
