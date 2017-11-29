//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_ConvertFieldType.hpp"
#include "PHAL_ConvertFieldType_Def.hpp"

PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES(PHAL::ConvertFieldType)
//PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ConvertFieldTypeRTtoMST)
//PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ConvertFieldTypeRTtoPST)
//PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ConvertFieldTypeRTtoST)
//PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ConvertFieldTypeMSTtoPST)
//PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ConvertFieldTypeMSTtoST)
//PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::ConvertFieldTypePSTtoST)
