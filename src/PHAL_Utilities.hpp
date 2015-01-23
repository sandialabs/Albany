//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_UTILITIES
#define PHAL_UTILITIES

#include "PHAL_AlbanyTraits.hpp"

namespace Albany { class Application; }

namespace PHAL {

/*! Collection of PHX::MDField utilities to perform basic operations.
 */

// Defined only within this file. See #undef at end.
#define loop(a, i, dim) for (PHAL::size_type i = 0; i < a.dimension(dim); ++i)

//! a(:) *= val
template<typename T1, typename T2, typename T3, typename T4>
inline void scale (PHX::MDField<T1, T2, T3>& a, const T4& val) {
  loop(a, i, 0) loop(a, j, 1) a(i,j) *= val;
}
template<typename T1, typename T2, typename T3, typename T4, typename T>
inline void scale (PHX::MDField<T1, T2, T3, T4>& a, const T& val) {
  loop(a, i, 0) loop(a, j, 1) loop(a, k, 2) a(i,j,k) *= val;
}

// a(:) = v
template<typename T1, typename T2, typename T3, typename T>
inline void set (PHX::MDField<T1, T2, T3>& a, const T& val) {
  loop(a, i, 0) loop(a, j, 1) a(i,j) = val;
}
template<typename T1, typename T2, typename T3, typename T4, typename T>
inline void set (PHX::MDField<T1, T2, T3, T4>& a, const T& val) {
  loop(a, i, 0) loop(a, j, 1) loop(a, k, 2) a(i,j,k) = val;
}

template<typename EvalT>
int getDerivativeDimensions (const Albany::Application* app,
                             const Albany::MeshSpecsStruct* ms);
// Convenience. Can call this once app has the discretization.
template<typename EvalT>
int getDerivativeDimensions (const Albany::Application* app,
                             const int element_block_idx);

#undef loop
}

#endif // PHAL_UTILITIES
