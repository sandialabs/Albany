//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "FELIX_HydrologyResidualMassEqn.hpp"
#include "FELIX_HydrologyResidualMassEqn_Def.hpp"

PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,true,true,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,true,true,false)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,true,false,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,true,false,false)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,false,true,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,false,true,false)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,false,false,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::HydrologyResidualMassEqn,false,false,false)
