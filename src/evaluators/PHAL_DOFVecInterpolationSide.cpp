//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_DOFVecInterpolationSide.hpp"
#include "PHAL_DOFVecInterpolationSide_Def.hpp"

PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::DOFVecInterpolationSide)
PHAL_INSTANTIATE_TEMPLATE_CLASS(PHAL::DOFVecInterpolationSideParam)
/*
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits, RealType>;
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits, FadType>;
//template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits, TanFadType>;
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, RealType>;
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType>;
//template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, TanFadType>;
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, RealType>;
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, FadType>;
//template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType>;
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, RealType>;
template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, FadType>;
//template class PHAL::DOFVecInterpolationSide<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType>;
*/
