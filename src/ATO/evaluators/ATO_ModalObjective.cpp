//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "ATO_ModalObjective.hpp"
#include "ATO_ModalObjective_Def.hpp"

template<typename EvalT, typename Traits>
const std::string ATO::ModalObjectiveBase<EvalT, Traits>::className = "Modal_Objective";

PHAL_INSTANTIATE_TEMPLATE_CLASS(ATO::ModalObjective)
PHAL_INSTANTIATE_TEMPLATE_CLASS(ATO::ModalObjectiveBase)

