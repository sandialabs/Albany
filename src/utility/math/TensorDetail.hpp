//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_TENSORDETAIL_HPP
#define UTIL_TENSORDETAIL_HPP

#include "TensorCommon.hpp"

/**
 *  \file TensorDetail.hpp
 *  
 *  \brief 
 */
namespace util {
namespace detail {
const index_t DYNAMIC_SIZE = 0;

template<int P>
struct static_pow
{
  template<typename T>
  static constexpr KOKKOS_INLINE_FUNCTION T value (T val) {
    return val * static_pow<P - 1>::value(val);
  }
};

template<>
struct static_pow<0>
{
  template<typename T>
  static constexpr KOKKOS_INLINE_FUNCTION T value (T) {
    return T(1);
  }
};

constexpr KOKKOS_INLINE_FUNCTION int
arg_count() {
  return 0;
}

template<typename T, typename... Args>
constexpr KOKKOS_INLINE_FUNCTION int
arg_count(T first, Args... args) {
  return 1 + arg_count(args...);
}

template<typename T>
KOKKOS_INLINE_FUNCTION T power_series (T& coeff, T x, T last) {
  coeff = 1;
  return last;// * static_pow<0>::value(x);
}

template<typename T, typename... Coefficients>
KOKKOS_INLINE_FUNCTION T power_series (T& coeff, T x, T first, Coefficients... rest) {
  T rem = power_series(coeff,x,rest...);
  coeff *= x;
  return first * coeff + rem;
}

template<typename T, typename... Coefficients>
KOKKOS_INLINE_FUNCTION T power_series (T x, Coefficients... rest) {
  T coeff = 1;
  return power_series(coeff, x, rest...);
}

}
}

#endif  // UTIL_TENSORDETAIL_HPP
