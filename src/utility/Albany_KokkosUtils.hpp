#ifndef ALBANY_KOKKOS_UTILS_HPP
#define ALBANY_KOKKOS_UTILS_HPP

// Get KOKKOS_INLINE_FUNCTION
#include "Kokkos_Macros.hpp"

// Get Sacado max/min
#include "Sacado.hpp"

namespace KU
{

#ifdef KOKKOS_ENABLE_CUDA

template <typename T>
KOKKOS_INLINE_FUNCTION
const T& max (const T& a, const T& b) { return a > b ? a : b; }

template <typename T>
KOKKOS_INLINE_FUNCTION
const T& min (const T& a, const T& b) { return a < b ? a : b; }

inline bool
IsNearDeviceMemoryLimit()
{
  size_t max_free_t = 104857600; // "Near" is subjective, let's use 100 MiB
  size_t free_t, total_t;
  cudaMemGetInfo(&free_t,&total_t);
  return free_t < max_free_t;
}

#else

using std::max;
using std::min;

inline bool
IsNearDeviceMemoryLimit()
{
  return false;
}

#endif

using Sacado::Fad::max;
using Sacado::Fad::min;
using Sacado::Fad::Exp::max;
using Sacado::Fad::Exp::min;

}

#endif // ALBANY_KOKKOS_UTILS_HPP

