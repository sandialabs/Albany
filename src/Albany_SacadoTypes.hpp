//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SACADO_TYPES_HPP
#define ALBANY_SACADO_TYPES_HPP

// Get all Albany configuration macros
#include "Albany_config.h"

// Get Albany's RealType
#include "Albany_ScalarOrdinalTypes.hpp"

// Include all of our AD types
#include "Sacado.hpp"
#include "Sacado_MathFunctions.hpp"
#include "Sacado_ELRFad_DFad.hpp"
#include "Sacado_ELRCacheFad_DFad.hpp"
#include "Sacado_Fad_DFad.hpp"
#include "Sacado_Fad_SLFad.hpp"
#include "Sacado_Fad_SFad.hpp"
#include "Sacado_ELRFad_SLFad.hpp"
#include "Sacado_ELRFad_SFad.hpp"
#include "Sacado_CacheFad_DFad.hpp"

// Include ScalarParameterLibrary to specialize its traits
#include "Sacado_ScalarParameterLibrary.hpp"
#include "Sacado_ScalarParameterVector.hpp"

// ******************************************************************
// Definition of Sacado::ParameterLibrary traits
// ******************************************************************

// Switch between dynamic and static FAD types
#if defined(ALBANY_FAD_TYPE_SFAD)
typedef Sacado::Fad::SFad<RealType, ALBANY_SFAD_SIZE> FadType;
#elif defined(ALBANY_FAD_TYPE_SLFAD)
typedef Sacado::Fad::SLFad<RealType, ALBANY_SLFAD_SIZE> FadType;
#else
typedef Sacado::Fad::DFad<RealType> FadType;
#endif

#if defined(ALBANY_TAN_FAD_TYPE_SFAD)
typedef Sacado::Fad::SFad<RealType, ALBANY_TAN_SFAD_SIZE> TanFadType;
#elif defined(ALBANY_TAN_FAD_TYPE_SLFAD)
typedef Sacado::Fad::SLFad<RealType, ALBANY_TAN_SLFAD_SIZE> TanFadType;
#else
typedef Sacado::Fad::DFad<RealType> TanFadType;
#endif

struct SPL_Traits {
  template <class T> struct apply {
    typedef typename T::ScalarT type;
  };
};

// Synonym for the ScalarParameterLibrary/Vector on our traits
typedef Sacado::ScalarParameterLibrary<SPL_Traits> ParamLib;
typedef Sacado::ScalarParameterVector<SPL_Traits> ParamVec;

namespace Albany
{

  // Function to get the underlying value out of a scalar type
  template <typename T>
  typename Sacado::ScalarType<T>::type
  KOKKOS_INLINE_FUNCTION
  ADValue(const T& x) { return Sacado::ScalarValue<T>::eval(x); }

  // Function to convert a ScalarType to a different one. This is used to convert
  // a ScalarT to a ParamScalarT.
  // Note, for all Evaluation types but one, ScalarT and ParamScalarT are the same,
  // and for those we want to keep the Fad derivatives (if any).
  // The only exception can be Jacobian (if mesh/param do not depend on solution),
  // where ParamScalarT=RealType and ScalarT=FadType.
  // Notice also that if the two scalar types are different, the conversion works
  // only if the target type has a constructor from the source type.
  template<typename ToST,typename FromST>
  struct ScalarConversionHelper
  {
    static typename std::enable_if<std::is_constructible<ToST,FromST>::value,ToST>::type
    apply (const FromST& x)
    {
      return ToST(x);
    }
  };

  template<typename FromST>
  struct ScalarConversionHelper<typename Sacado::ScalarType<FromST>::type,FromST>
  {
    static typename Sacado::ScalarType<FromST>::type apply (const FromST& x)
    {
      return ADValue(x);
    }
  };

  template<typename ToST,typename FromST>
  ToST convertScalar (const FromST& x)
  {
    return ScalarConversionHelper<ToST,FromST>::apply(x);
  }

} // namespace Albany

#endif // ALBANY_SACADO_TYPES_HPP
