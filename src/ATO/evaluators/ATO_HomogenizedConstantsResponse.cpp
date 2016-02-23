//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "ATO_HomogenizedConstantsResponse.hpp"
#include "ATO_HomogenizedConstantsResponse_Def.hpp"

template<typename EvalT, typename Traits>
const std::string ATO::HomogenizedConstantsResponse<EvalT, Traits>::className = "HomogenizedConstantsResponse";

PHAL_INSTANTIATE_TEMPLATE_CLASS(ATO::HomogenizedConstantsResponse)

