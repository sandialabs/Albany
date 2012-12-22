//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Utilities_i_cc)
#define tensor_Utilities_i_cc

#include <limits>

namespace LCM {

  //
  // Sign function
  //
  template <typename T>
  inline
  int
  sgn(T const & s) {
    return (T(0) < s) - (s < T(0));
  }

  //
  // Copysign function
  //
  template<typename T>
  inline
  T
  copysign(T const & a, T const & b)
  {
    return b >= 0 ? std::abs(a) : -std::abs(a);
  }

  //
  // NaN function. Necessary to choose the proper underlying NaN
  // for non-floating-point types.
  // Assumption: non-floating-point types have a typedef that
  // determines the underlying floating-point type.
  //
  template<typename T>
  inline
  typename Sacado::ScalarType<T>::type
  not_a_number()
  {
    return
        std::numeric_limits<typename Sacado::ScalarType<T>::type>::quiet_NaN();
  }

  //
  // Machine epsilon function. Necessary to choose the proper underlying
  // machine epsilon for non-floating-point types.
  // Assumption: non-floating-point types have a typedef that
  // determines the underlying floating-point type.
  //
  template<typename T>
  inline
  typename Sacado::ScalarType<T>::type
  machine_epsilon()
  {
    return
        std::numeric_limits<typename Sacado::ScalarType<T>::type>::epsilon();
  }

} // namespace LCM

#endif // tensor_Utilities_i_cc
