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

#else

using std::max;
using std::min;

#endif

using Sacado::Fad::max;
using Sacado::Fad::min;
using Sacado::Fad::Exp::max;
using Sacado::Fad::Exp::min;

}

#endif // ALBANY_KOKKOS_UTILS_HPP