//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_DOFGradInterpolation.hpp"
#include "PHAL_DOFGradInterpolation_Def.hpp"

PHAL_INSTANTIATE_TEMPLATE_CLASS_FOR_ALL_SCALARS(PHAL::DOFGradInterpolation)
PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::DOFGradInterpolation_noDeriv)

