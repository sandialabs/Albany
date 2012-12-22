//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Utilities_h)
#define tensor_Utilities_h

#include "Sacado.hpp"

namespace LCM {

  ///
  /// Sign function
  ///
  template<typename T>
  int
  sgn(T const & s);

  ///
  /// Copysign function
  ///
  template<typename T>
  T
  copysign(T const & a, T const & b);

  ///
  /// NaN function. Necessary to choose the proper underlying NaN
  /// for non-floating-point types.
  /// Assumption: non-floating-point types have a typedef that
  /// determines the underlying floating-point type.
  ///
  template<typename T>
  typename Sacado::ScalarType<T>::type
  not_a_number();

  ///
  /// Machine epsilon function. Necessary to choose the proper underlying
  /// machine epsilon for non-floating-point types.
  /// Assumption: non-floating-point types have a typedef that
  /// determines the underlying floating-point type.
  ///
  template<typename T>
  typename Sacado::ScalarType<T>::type
  machine_epsilon();

} // namespace LCM

#include "Utilities.i.cc"

#endif // tensor_Utilities_h
