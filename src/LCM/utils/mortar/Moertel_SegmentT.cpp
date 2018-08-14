//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_ExplicitTemplateInstantiation.hpp"

#ifdef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_SegmentT.hpp"
#include "Moertel_SegmentT_Def.hpp"

namespace MoertelT {

MOERTEL_INSTANTIATE_TEMPLATE_CLASS(SegmentT)

}  // namespace MoertelT

// non-member operators at global scope
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_INT
template std::ostream&
operator<<(
    std::ostream&                                              os,
    const MoertelT::SegmentT<3, double, int, int, KokkosNode>& inter);
template std::ostream&
operator<<(
    std::ostream&                                              os,
    const MoertelT::SegmentT<2, double, int, int, KokkosNode>& inter);
#endif
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_LONGLONGINT
template std::ostream&
operator<<(
    std::ostream&                                                    os,
    const MoertelT::SegmentT<3, double, int, long long, KokkosNode>& inter);
template std::ostream&
operator<<(
    std::ostream&                                                    os,
    const MoertelT::SegmentT<2, double, int, long long, KokkosNode>& inter);
#endif

#endif
