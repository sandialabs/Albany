//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_RC_DATATYPES_IMPL
#define AADAPT_RC_DATATYPES_IMPL

#include "AAdapt_RC_DataTypes.hpp"

namespace AAdapt {
namespace rc {

/*! Some macros for ETI. For internal use in the AAdapt::rc namespace. Include
 *  only in .cpp files.
 */

/*! aadapt_rc_apply_to_all_ad_types(macro, arg2) applies macro(Type, arg2) to
 *  every AD type Albany_DataTypes.hpp defines. Type is RealType, FadType, etc,
 *  and arg2 is a user's argument.
 */
#ifdef ALBANY_SG
# ifdef ALBANY_ENSEMBLE
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)                                  \
  macro(TanFadType, arg2)                               \
  macro(SGType, arg2)                                   \
  macro(SGFadType, arg2)                                \
  macro(MPType, arg2)                                   \
  macro(MPFadType, arg2)
#  else
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)                                  \
  macro(SGType, arg2)                                   \
  macro(SGFadType, arg2)                                \
  macro(MPType, arg2)                                   \
  macro(MPFadType, arg2)
#  endif
# else
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)                                  \
  macro(TanFadType, arg2)                               \
  macro(SGType, arg2)                                   \
  macro(SGFadType, arg2)
#  else
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)                                  \
  macro(SGType, arg2)                                   \
  macro(SGFadType, arg2)
#  endif
# endif
#else 
# ifdef ALBANY_ENSEMBLE
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)                                  \
  macro(TanFadType, arg2)                               \
  macro(MPType, arg2)                                   \
  macro(MPFadType, arg2)
#  else
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)                                  \
  macro(MPType, arg2)                                   \
  macro(MPFadType, arg2)
#  endif
# else
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)                                  \
  macro(TanFadType, arg2)
#  else
#define aadapt_rc_apply_to_all_ad_types(macro, arg2)    \
  macro(RealType, arg2)                                 \
  macro(FadType, arg2)
#  endif
# endif
#endif 

/*! aadapt_rc_apply_to_all_eval_types(macro) applies a macro to every evaluation
 *  type PHAL::AlbanyTraits defines.
 */
#ifdef ALBANY_SG
# ifdef ALBANY_ENSEMBLE
#define aadapt_rc_apply_to_all_eval_types(macro)        \
  macro(PHAL::AlbanyTraits::Residual)                   \
  macro(PHAL::AlbanyTraits::Jacobian)                   \
  macro(PHAL::AlbanyTraits::Tangent)                    \
  macro(PHAL::AlbanyTraits::DistParamDeriv)             \
  macro(PHAL::AlbanyTraits::SGResidual)                 \
  macro(PHAL::AlbanyTraits::SGJacobian)                 \
  macro(PHAL::AlbanyTraits::SGTangent)                  \
  macro(PHAL::AlbanyTraits::MPResidual)                 \
  macro(PHAL::AlbanyTraits::MPJacobian)                 \
  macro(PHAL::AlbanyTraits::MPTangent)
# else
#define aadapt_rc_apply_to_all_eval_types(macro)        \
  macro(PHAL::AlbanyTraits::Residual)                   \
  macro(PHAL::AlbanyTraits::Jacobian)                   \
  macro(PHAL::AlbanyTraits::Tangent)                    \
  macro(PHAL::AlbanyTraits::DistParamDeriv)             \
  macro(PHAL::AlbanyTraits::SGResidual)                 \
  macro(PHAL::AlbanyTraits::SGJacobian)                 \
  macro(PHAL::AlbanyTraits::SGTangent)
# endif
#else
# ifdef ALBANY_ENSEMBLE
#define aadapt_rc_apply_to_all_eval_types(macro)        \
  macro(PHAL::AlbanyTraits::Residual)                   \
  macro(PHAL::AlbanyTraits::Jacobian)                   \
  macro(PHAL::AlbanyTraits::Tangent)                    \
  macro(PHAL::AlbanyTraits::DistParamDeriv)             \
  macro(PHAL::AlbanyTraits::MPResidual)                 \
  macro(PHAL::AlbanyTraits::MPJacobian)                 \
  macro(PHAL::AlbanyTraits::MPTangent)
# else
#define aadapt_rc_apply_to_all_eval_types(macro)        \
  macro(PHAL::AlbanyTraits::Residual)                   \
  macro(PHAL::AlbanyTraits::Jacobian)                   \
  macro(PHAL::AlbanyTraits::Tangent)                    \
  macro(PHAL::AlbanyTraits::DistParamDeriv)
# endif
#endif

/*! Perform ETI for a class \code template<int rank> Class \endcode.
 */
#define aadapt_rc_eti_class(Class)              \
  template class Class<0>;                      \
  template class Class<1>;                      \
  template class Class<2>;
/*! Apply \code aadapt_rc_apply_to_all_ad_types(eti, rank) \endcode to each \c
 *  rank.
 */
#define aadapt_rc_apply_to_all_ad_types_all_ranks(macro)        \
  aadapt_rc_apply_to_all_ad_types(macro, 0)                     \
  aadapt_rc_apply_to_all_ad_types(macro, 1)                     \
  aadapt_rc_apply_to_all_ad_types(macro, 2)

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_DATATYPES_IMPL
