//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DATATYPES
#define PHAL_DATATYPES

//TODO: this looks suspicious and temporary. remove it soon.
#define AMB_KOKKOS

//! Data Type Definitions that span the code.

// Include all of our AD types
#include "Sacado.hpp"
#include "Sacado_MathFunctions.hpp"
#include "Sacado_ELRFad_DFad.hpp"
#include "Sacado_ELRCacheFad_DFad.hpp"
#include "Sacado_Fad_DFad.hpp"
#include "Sacado_Fad_SLFad.hpp"
#include "Sacado_ELRFad_SLFad.hpp"
#include "Sacado_ELRFad_SFad.hpp"
#include "Sacado_CacheFad_DFad.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"

#include "TpetraCore_config.h"

#ifndef HAVE_TPETRA_INST_DOUBLE
#error "Albany needs Tpetra to enable double as a Scalar type"
#endif
typedef double RealType;

// Switch between dynamic and static FAD types
#ifdef ALBANY_FAST_FELIX
  // Code templated on data type need to know if FadType and TanFadType
  // are the same or different typdefs
#define ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
  typedef Sacado::Fad::SLFad<RealType, ALBANY_SLFAD_SIZE> FadType;
#else
#define ALBANY_SFAD_SIZE 300
  typedef Sacado::Fad::DFad<RealType> FadType;
#endif

typedef Sacado::Fad::DFad<RealType> TanFadType;

//Tpetra includes
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DistObject.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_MultiVector.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "Tpetra_RowMatrixTransposer.hpp"

//Kokkos includes
#include "Phalanx_KokkosDeviceTypes.hpp"

//Tpetra typedefs
typedef RealType                                             ST;
typedef std::int64_t                                         GO;
typedef std::int32_t                                         LO;
typedef Kokkos::Compat::KokkosDeviceWrapperNode<PHX::Device> KokkosNode;

typedef int Tpetra_LO;
#if defined( HAVE_TPETRA_INST_INT_LONG_LONG )
typedef long long Tpetra_GO;
#elif defined( HAVE_TPETRA_INST_INT_LONG )
static_assert(sizeof(long) == sizeof(GO),
    "Tpetra's biggest enabled GlobalOrdinal is long but thats not 64 bit");
typedef long Tpetra_GO;
#elif defined( HAVE_TPETRA_INST_INT_UNSIGNED_LONG )
static_assert(sizeof(long) == sizeof(GO),
    "Tpetra's biggest enabled GlobalOrdinal is unsigned long but thats not 64 bit");
typedef unsigned long Tpetra_GO;
#else
#error "Albany needs Tpetra to have a 64-bit GlobalOrdinal enabled"
#endif

typedef Teuchos::Comm<int>                                            Teuchos_Comm;
typedef Tpetra::Map<Tpetra_LO, Tpetra_GO, KokkosNode>                 Tpetra_Map;
typedef Tpetra::Details::LocalMap<Tpetra_LO, Tpetra_GO, KokkosNode>   Tpetra_LocalMap; 
typedef Tpetra::Export<Tpetra_LO, Tpetra_GO, KokkosNode>              Tpetra_Export;
typedef Tpetra::Import<Tpetra_LO, Tpetra_GO, KokkosNode>              Tpetra_Import;
typedef Tpetra::CrsGraph<Tpetra_LO, Tpetra_GO, KokkosNode>            Tpetra_CrsGraph;
typedef Tpetra::CrsMatrix<ST, Tpetra_LO, Tpetra_GO, KokkosNode>       Tpetra_CrsMatrix;
typedef Tpetra::RowMatrix<ST, Tpetra_LO, Tpetra_GO, KokkosNode>       Tpetra_RowMatrix;
typedef Tpetra::Operator<ST, Tpetra_LO, Tpetra_GO, KokkosNode>        Tpetra_Operator;
typedef Tpetra::Vector<ST, Tpetra_LO, Tpetra_GO, KokkosNode>          Tpetra_Vector;
typedef Tpetra::MultiVector<ST, Tpetra_LO, Tpetra_GO, KokkosNode>     Tpetra_MultiVector;
typedef Thyra::TpetraOperatorVectorExtraction<
    ST, Tpetra_LO, Tpetra_GO, KokkosNode> ConverterT;
typedef Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>                Tpetra_MatrixMarket_Writer;
typedef Thyra::TpetraVector<ST,Tpetra_LO,Tpetra_GO,KokkosNode>        ThyraVector;
typedef Thyra::TpetraMultiVector<ST,Tpetra_LO,Tpetra_GO,KokkosNode>   ThyraMultiVector;
typedef Tpetra::RowMatrixTransposer<
    ST, Tpetra_LO, Tpetra_GO, KokkosNode >                            Tpetra_RowMatrixTransposer; 

// Include ScalarParameterLibrary to specialize its traits
#include "Sacado_ScalarParameterLibrary.hpp"
#include "Sacado_ScalarParameterVector.hpp"

// ******************************************************************
// Definition of Sacado::ParameterLibrary traits
// ******************************************************************
struct SPL_Traits {
  template <class T> struct apply {
    typedef typename T::ScalarT type;
  };
};

// Synonym for the ScalarParameterLibrary/Vector on our traits
typedef Sacado::ScalarParameterLibrary<SPL_Traits> ParamLib;
typedef Sacado::ScalarParameterVector<SPL_Traits> ParamVec;

namespace Albany {

  // Function to get the underlying value out of a scalar type
  template <typename T>
  typename Sacado::ScalarType<T>::type
  ADValue(const T& x) { return Sacado::ScalarValue<T>::eval(x); }

  // Function to convert a ScalarType to a different one. This is used to convert
  // a ScalarT to a ParamScalarT.
  // Note, for all Evaluation types but one, ScalarT and ParamScalarT are the same,
  // and for those we want to keep the Fad derivatives (if any).
  // The only exception can be Jacobian (if mesh/param do not depend on solution),
  // where ParamScalarT=RealType and ScalarT=FadType.
  // Notice also that if the two scalar types are different, the conversion works
  // only if the target type has a constructor from the source type.
  template<typename FromST,typename ToST>
  struct ScalarConversionHelper
  {
    static ToST apply (const FromST& x)
    {
      return ToST(x);
    }
  };

  template<typename FromST>
  struct ScalarConversionHelper<FromST,typename Sacado::ScalarType<FromST>::type>
  {
    static typename Sacado::ScalarType<FromST>::type apply (const FromST& x)
    {
      return ADValue(x);
    }
  };

  template<typename FromST,typename ToST>
  ToST convertScalar (const FromST& x)
  {
    return ScalarConversionHelper<FromST,ToST>::apply(x);
  }
}

// Code macros to support deprecated warnings
#ifdef ALBANY_ENABLE_DEPRECATED
#  if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define ALBANY_DEPRECATED  __attribute__((__deprecated__))
#  else
#    define ALBANY_DEPRECATED
#  endif
#endif

#endif
