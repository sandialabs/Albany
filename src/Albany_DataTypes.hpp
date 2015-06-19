//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DATATYPES
#define PHAL_DATATYPES

#define AMB_KOKKOS

//! Data Type Definitions that span the code.

// Include all of our AD types
#include "Stokhos_Sacado_Kokkos.hpp"
#include "Sacado.hpp"
#include "Sacado_MathFunctions.hpp"
#include "Stokhos_Sacado_MathFunctions.hpp"
#include "Sacado_ELRFad_DFad.hpp"
#include "Sacado_ELRCacheFad_DFad.hpp"
#include "Sacado_Fad_DFad.hpp"
#include "Sacado_Fad_SLFad.hpp"
#include "Sacado_ELRFad_SLFad.hpp"
#include "Sacado_ELRFad_SFad.hpp"
#include "Sacado_CacheFad_DFad.hpp"
#include "Sacado_PCE_OrthogPoly.hpp"
#include "Sacado_MP_Vector.hpp"

//amb Need to move to configuration.
//#define ALBANY_SFAD_SIZE 27
//#define ALBANY_SLFAD_SIZE 27

//#define ALBANY_ENSEMBLE_SIZE 32  -- set in CMakeLists.txt

//#define ALBANY_FAST_FELIX
// Typedef AD types to standard names
typedef double RealType;

// SG data types
typedef Stokhos::StandardStorage<int,double> StorageType;
typedef Sacado::PCE::OrthogPoly<double,StorageType> SGType;

// Ensemble (a.k.a. MP) data types
#ifndef ALBANY_ENSEMBLE_SIZE
#define ALBANY_ENSEMBLE_SIZE 1
#endif
typedef Stokhos::StaticFixedStorage<int,double,ALBANY_ENSEMBLE_SIZE,Kokkos::Serial> MPStorageType;
typedef Sacado::MP::Vector<MPStorageType> MPType;

// Switch between dynamic and static FAD types
#ifdef ALBANY_FAST_FELIX
  // Code templated on data type need to know if FadType and TanFadType
  // are the same or different typdefs
#define ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
  typedef Sacado::Fad::SLFad<double, ALBANY_SLFAD_SIZE> FadType;
  typedef Sacado::Fad::SLFad<SGType, ALBANY_SLFAD_SIZE> SGFadType;
  typedef Sacado::Fad::SLFad<MPType, ALBANY_SLFAD_SIZE> MPFadType;
#else
  #define ALBANY_SFAD_SIZE 300
  typedef Sacado::Fad::DFad<double> FadType;
  typedef Sacado::Fad::DFad<SGType> SGFadType;
  typedef Sacado::Fad::DFad<MPType> MPFadType;
#endif

typedef Sacado::Fad::DFad<double> TanFadType;

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


//Kokkos includes
#include "Kokkos_SerialNode.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"

//Tpetra typedefs
typedef double                                      ST;
#ifdef ALBANY_64BIT_INT
typedef long long int                               GO;
#else
typedef int                                         GO;
#endif
typedef int                                         LO;
typedef Kokkos::Compat::KokkosDeviceWrapperNode<PHX::Device> KokkosNode;

//typedef Kokkos::DefaultNode::DefaultNodeType KokkosNode; // Whatever is Trilinos compiled to use?
//typedef Kokkos::SerialNode KokkosNode; // No threading
//typedef Kokkos::TPINode KokkosNode; // custom Pthreads
//typedef Kokkos::TBBNode KokkosNode; // Intel TBB
//typedef Kokkos::ThrustNode KokkosNode; // C++ Cuda wtapper

typedef Teuchos::Comm<int>                          Teuchos_Comm;
typedef Tpetra::Map<LO, GO, KokkosNode>             Tpetra_Map;
typedef Tpetra::Export<LO, GO, KokkosNode>          Tpetra_Export;
typedef Tpetra::Import<LO, GO, KokkosNode>          Tpetra_Import;
typedef Tpetra::CrsGraph<LO, GO, KokkosNode>        Tpetra_CrsGraph;
typedef Tpetra::CrsMatrix<ST, LO, GO, KokkosNode>   Tpetra_CrsMatrix;
typedef Tpetra::Operator<ST, LO, GO, KokkosNode>    Tpetra_Operator;
typedef Tpetra::Vector<ST, LO, GO, KokkosNode>      Tpetra_Vector;
typedef Tpetra::MultiVector<ST, LO, GO, KokkosNode> Tpetra_MultiVector;
typedef Thyra::TpetraOperatorVectorExtraction<ST, LO, GO, KokkosNode> ConverterT;
typedef Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix> Tpetra_MatrixMarket_Writer;
typedef Thyra::TpetraVector<ST,LO,GO,KokkosNode> ThyraVector;
typedef Thyra::TpetraMultiVector<ST,LO,GO,KokkosNode> ThyraMultiVector;


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

}

#endif
