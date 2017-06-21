//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "ATO_InterfaceEnergy.hpp"
#include "ATO_InterfaceEnergy_Def.hpp"

template<typename EvalT, typename Traits>
const std::string ATO::InterfaceEnergyBase<EvalT, Traits>::className = "Interface_Energy";

PHAL_INSTANTIATE_TEMPLATE_CLASS(ATO::InterfaceEnergy)
PHAL_INSTANTIATE_TEMPLATE_CLASS(ATO::InterfaceEnergyBase)

