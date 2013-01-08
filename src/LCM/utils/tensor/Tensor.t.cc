//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Tensor_t_cc)
#define tensor_Tensor_t_cc

namespace LCM {

  //
  // R^N tensor input
  // \param A tensor
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Tensor<T> & A)
  {

    const Index
    N = A.get_dimension();

    for (Index i = 0; i < N; ++i) {
      for (Index j = 0; j < N; ++j) {
        is >> A(i,j);
      }
    }

    return is;
  }

  //
  // R^N tensor output
  // \param A tensor
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    if (N == 0) {
      return os;
    }

    for (Index i = 0; i < N; ++i) {

      os << std::scientific << A(i,0);

      for (Index j = 1; j < N; ++j) {
        os << std::scientific << "," << A(i,j);
      }

      os << std::endl;
    }

    return os;
  }

} // namespace LCM

#endif // tensor_Tensor_t_cc
