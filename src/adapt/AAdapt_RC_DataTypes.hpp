//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_RC_DATATYPES
#define AADAPT_RC_DATATYPES

#include "PHAL_AlbanyTraits.hpp"

namespace AAdapt {
namespace rc {

/*! Data types used in ref config update.
 */

/*! Define Tensor<ad_type, rank> Tensor<ad_type, rank>::type is the
 *! corresponding MDField type, and Tensor<ad_type, rank>::rank gives the rank.
 */
template<typename ad_type, int rank> struct Tensor;
template<typename ad_type> struct Tensor<ad_type, 0> {
  enum { rank = 0 };
  typedef PHX::MDField<ad_type, Cell, QuadPoint> type;
};
template<typename ad_type> struct Tensor<ad_type, 1> {
  enum { rank = 1 };
  typedef PHX::MDField<ad_type, Cell, QuadPoint, Dim> type;
};
template<typename ad_type> struct Tensor<ad_type, 2> {
  enum { rank = 2 };
  typedef PHX::MDField<ad_type, Cell, QuadPoint, Dim, Dim> type;
};
/*! For convenience, define RTensor<rank> as shorthand for Tensor<RealType,
 *! rank>.
 */
template<int rank> struct RTensor : public Tensor<RealType, rank> {};

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_DATATYPES
