//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "FELIX_BasalFrictionCoefficientNode.hpp"
#include "FELIX_BasalFrictionCoefficientNode_Def.hpp"

PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::BasalFrictionCoefficientNode,true,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::BasalFrictionCoefficientNode,true,false)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::BasalFrictionCoefficientNode,false,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::BasalFrictionCoefficientNode,false,false)
