//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_ExplicitTemplateInstantiation.hpp"

#ifdef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_Convexhull_Def.hpp"
#include "Moertel_OverlapT.hpp"
#include "Moertel_OverlapT_Def.hpp"
#include "Moertel_OverlapT_Utils_Def.hpp"
#endif

namespace MoertelT {

#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_INT
MOERTEL_INSTANTIATE_NESTED_TEMPLATE_CLASS_ST_LO_GO_N(
    MoertelT::OverlapT,
    MoertelT::InterfaceT,
    double,
    int,
    int,
    KokkosNode)
#endif
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_LONGLONGINT
MOERTEL_INSTANTIATE_NESTED_TEMPLATE_CLASS_ST_LO_GO_N(
    MoertelT::OverlapT,
    MoertelT::InterfaceT,
    double,
    int,
    long long,
    KokkosNode)
#endif

MOERTEL_INSTANTIATE_TEMPLATE_CLASS_ON_NAME_ORD(Overlap, Interface)

}  // namespace MoertelT

#endif
