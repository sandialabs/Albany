//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_ExplicitTemplateInstantiation.hpp"

#ifdef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_ManagerT.hpp"
#include "Moertel_ManagerT_Def.hpp"

namespace MoertelT {

MOERTEL_INSTANTIATE_TEMPLATE_CLASS(ManagerT)

}  // namespace MoertelT

#endif
