//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Vector_t_cc)
#define tensor_Vector_t_cc

namespace LCM {

  //
  // R^N vector input
  // \param u vector
  // \param is input stream
  // \return is input stream
  //
  template<typename T>
  std::istream &
  operator>>(std::istream & is, Vector<T> & u)
  {
    Index const
    N = u.get_dimension();

    for (Index i = 0; i < N; ++i) {
      is >> u(i);
    }

    return is;
  }

  //
  // R^N vector output
  // \param u vector
  // \param os output stream
  // \return os output stream
  //
  template<typename T>
  std::ostream &
  operator<<(std::ostream & os, Vector<T> const & u)
  {
    Index const
    N = u.get_dimension();

    if (N == 0) {
      return os;
    }

    os << std::scientific << u(0);

    for (Index i = 1; i < N; ++i) {
      os << std::scientific << "," << u(i);
    }

    return os;
  }

} // namespace LCM

#endif // tensor_Vector_t_cc
