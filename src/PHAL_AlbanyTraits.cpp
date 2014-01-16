//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

const std::string PHX::TypeString<PHAL::AlbanyTraits::Residual>::value =
  "<Residual>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::Jacobian>::value =
  "<Jacobian>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::Tangent>::value =
  "<Tangent>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::DistParamDeriv>::value =
  "<DistParamDeriv>";

const std::string PHX::TypeString<RealType>::value =
  "double";

// AGS: This is not necessarily true -- need tograb from FadType?
const std::string PHX::TypeString<FadType>::
  value = "Sacado::ELRFad::DFad<double>";

#ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
// This is necessary iff TanFadType is different from FadType
const std::string PHX::TypeString<TanFadType>::
  value = "Sacado::ELRFad::DFad<double>";
#endif

#ifdef ALBANY_SG_MP
const std::string PHX::TypeString<PHAL::AlbanyTraits::SGResidual>::value =
  "<SGResidual>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::SGJacobian>::value =
  "<SGJacobian>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::SGTangent>::value =
  "<SGTangent>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::MPResidual>::value =
  "<MPResidual>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::MPJacobian>::value =
  "<MPJacobian>";

const std::string PHX::TypeString<PHAL::AlbanyTraits::MPTangent>::value =
  "<MPTangent>";

const std::string PHX::TypeString<SGType>::
  value = "Sacado::PCE::OrthogPoly<double>";

const std::string PHX::TypeString<SGFadType>::
  value = "Sacado::ELRCacheFad::DFad< Sacado::PCE::OrthogPoly<double> >";

const std::string PHX::TypeString<MPType>::
  value = "Sacado::ETV::Vector<double>";

const std::string PHX::TypeString<MPFadType>::
  value = "Sacado::ELRCacheFad::DFad< Sacado::ETV::Vector<double> >";
#endif //ALBANY_SG_MP
