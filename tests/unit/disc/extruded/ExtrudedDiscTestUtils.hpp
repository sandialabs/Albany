//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Shared utility helpers for extruded discretization unit tests.
// These helpers compute LOCAL (per-rank) min/max of Kokkos DynRankViews
// without performing any MPI communication.  Callers should accumulate
// the per-workset local results and then issue a single Teuchos::reduceAll
// to obtain the global min/max, avoiding one collective call per workset.

#ifndef EXTRUDED_DISC_TEST_UTILS_HPP
#define EXTRUDED_DISC_TEST_UTILS_HPP

#include <algorithm>
#include <limits>
#include <utility>
#include <cstddef>

namespace ExtrudedDiscTestUtils {

// Compute the local (per-rank) min and max of a rank-2 view.
// No MPI reduction is performed.
template<typename ViewType>
std::pair<double,double> local_minmax2(const ViewType& v)
{
  double lmin =  std::numeric_limits<double>::max();
  double lmax = -std::numeric_limits<double>::max();
  for (size_t i=0; i<v.extent(0); ++i)
    for (size_t j=0; j<v.extent(1); ++j) {
      lmin = std::min(lmin, (double)v(i,j));
      lmax = std::max(lmax, (double)v(i,j));
    }
  return {lmin, lmax};
}

// Compute the local (per-rank) min and max of a rank-3 view.
// No MPI reduction is performed.
template<typename ViewType>
std::pair<double,double> local_minmax3(const ViewType& v)
{
  double lmin =  std::numeric_limits<double>::max();
  double lmax = -std::numeric_limits<double>::max();
  for (size_t i=0; i<v.extent(0); ++i)
    for (size_t j=0; j<v.extent(1); ++j)
      for (size_t k=0; k<v.extent(2); ++k) {
        lmin = std::min(lmin, (double)v(i,j,k));
        lmax = std::max(lmax, (double)v(i,j,k));
      }
  return {lmin, lmax};
}

} // namespace ExtrudedDiscTestUtils

#endif // EXTRUDED_DISC_TEST_UTILS_HPP
