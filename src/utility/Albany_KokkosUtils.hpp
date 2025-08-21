#ifndef ALBANY_KOKKOS_UTILS_HPP
#define ALBANY_KOKKOS_UTILS_HPP

// Get KOKKOS_INLINE_FUNCTION
#include "Kokkos_Macros.hpp"

// Get Sacado max/min
#include "Sacado.hpp"

namespace KU
{

// Check device memory
#ifdef KOKKOS_ENABLE_CUDA
inline bool
IsNearDeviceMemoryLimit()
{
  size_t max_free_t = 104857600; // "Near" is subjective, let's use 100 MiB
  size_t free_t, total_t;
  cudaMemGetInfo(&free_t,&total_t);
  return free_t < max_free_t;
}
#else
inline bool
IsNearDeviceMemoryLimit()
{
  return false;
}
#endif

// Kernel min/max
#ifdef KOKKOS_ENABLE_CUDA
template <typename T>
KOKKOS_INLINE_FUNCTION
const T& max (const T& a, const T& b) { return a > b ? a : b; }

template <typename T>
KOKKOS_INLINE_FUNCTION
const T& min (const T& a, const T& b) { return a < b ? a : b; }
#else
using std::max;
using std::min;
#endif
using Sacado::Fad::max;
using Sacado::Fad::min;
using Sacado::Fad::Exp::max;
using Sacado::Fad::Exp::min;

// Choose atomic based on execution space
template <typename ExeSpace>
struct NeedsAtomic
{
  enum : bool
  {
    value = true
  };
};

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct NeedsAtomic<Kokkos::Serial>
{
  enum : bool
  {
    value = false
  };
};
#endif

// Kernel atomic add
template <typename ExeSpace, typename D, typename V>
struct AtomicAddImpl
{
  KOKKOS_FORCEINLINE_FUNCTION static void
  atomic_add(D dst, const V& val)
  {
    Kokkos::atomic_add(dst, val);
  }
};

#ifdef KOKKOS_ENABLE_SERIAL
template <typename D, typename V>
struct AtomicAddImpl<Kokkos::Serial, D, V>
{
  KOKKOS_FORCEINLINE_FUNCTION static void
  atomic_add(D dst, const V& val)
  {
    *dst += val;
  }
};
#endif

template <typename ExeSpace, typename D, typename V>
KOKKOS_FORCEINLINE_FUNCTION void
atomic_add(D dst, const V& val)
{
  KU::AtomicAddImpl<ExeSpace, D, V>::atomic_add(dst, val);
}

#ifdef KOKKOS_ENABLE_CUDA
typedef Kokkos::LaunchBounds<> AlbanyLaunchBounds;
#else
typedef Kokkos::LaunchBounds<128,2> AlbanyLaunchBounds;
#endif

}

#endif // ALBANY_KOKKOS_UTILS_HPP

