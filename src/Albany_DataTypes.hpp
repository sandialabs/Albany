//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DATATYPES
#define PHAL_DATATYPES

//! Data Type Definitions that span the code.

// Include all of our AD types
#include "Sacado_MathFunctions.hpp"
#include "Stokhos_Sacado_MathFunctions.hpp"
#include "Sacado_ELRFad_DFad.hpp"
#include "Sacado_ELRCacheFad_DFad.hpp"
#include "Sacado_Fad_DFad.hpp"
#include "Sacado_CacheFad_DFad.hpp"
#include "Sacado_PCE_OrthogPoly.hpp"
#include "Sacado_ETV_Vector.hpp"

// Typedef AD types to standard names
typedef double RealType;
typedef Sacado::ELRFad::DFad<double> FadType;
typedef Stokhos::StandardStorage<int,double> StorageType;
typedef Sacado::PCE::OrthogPoly<double,StorageType> SGType;
typedef Sacado::Fad::DFad<SGType> SGFadType;
typedef Sacado::ETV::Vector<double,StorageType> MPType;
typedef Sacado::Fad::DFad<MPType> MPFadType;

//Tpetra includes
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_BlockMap.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_BlockCrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DistObject.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_MultiVector.hpp"
#include "MatrixMarket_Tpetra.hpp"

//Kokkos includes 
#include "Kokkos_SerialNode.hpp"


//Tpetra typedefs
typedef double                                      ST;
typedef int                                         GO;
typedef int                                         LO;
typedef KokkosClassic::SerialNode                   KokkosNode;
typedef Tpetra::Map<LO, GO, KokkosNode>             Tpetra_Map;
typedef Tpetra::BlockMap<LO, GO, KokkosNode>        Tpetra_BlockMap;
typedef Tpetra::Export<LO, GO, KokkosNode>          Tpetra_Export;
typedef Tpetra::Import<LO, GO, KokkosNode>          Tpetra_Import;
typedef Tpetra::CrsGraph<LO, GO, KokkosNode>        Tpetra_CrsGraph;
typedef Tpetra::BlockCrsGraph<LO, GO, KokkosNode>   Tpetra_BlockCrsGraph;
typedef Tpetra::CrsMatrix<ST, LO, GO, KokkosNode>   Tpetra_CrsMatrix;
typedef Tpetra::Operator<ST, LO, GO, KokkosNode>    Tpetra_Operator;
typedef Tpetra::Vector<ST, LO, GO, KokkosNode>      Tpetra_Vector;
typedef Tpetra::MultiVector<ST, LO, GO, KokkosNode> Tpetra_MultiVector;


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
