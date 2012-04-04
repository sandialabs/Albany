/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
#include "Tpetra_CrsGraph.hpp"
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
typedef Kokkos::SerialNode                          KokkosNode;
typedef Tpetra::Map<LO, GO, KokkosNode>             Tpetra_Map;
typedef Tpetra::Export<LO, GO, KokkosNode>          Tpetra_Export;
typedef Tpetra::Import<LO, GO, KokkosNode>          Tpetra_Import;
typedef Tpetra::CrsGraph<LO, GO, KokkosNode>        Tpetra_CrsGraph;
typedef Tpetra::CrsMatrix<ST, LO, GO, KokkosNode>   Tpetra_CrsMatrix;
typedef Tpetra::Vector<ST, LO, GO, KokkosNode>      Tpetra_Vector;


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
