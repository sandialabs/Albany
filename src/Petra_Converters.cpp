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

#include "Petra_Converters.hpp"
#include "Teuchos_TestForException.hpp"
#include <cstdlib>
#include <stdexcept>

  // Start of Utils to do with Communicators
#ifdef ALBANY_MPI

#else

#endif
  // End of Utils to do with Communicators



//TpetraMap_To_EpetraMap: takes in Tpetra::Map object, converts it to its equivalent Epetra_Map object, 
//and returns an RCP pointer to this Epetra_Map
Teuchos::RCP<Epetra_Map> Petra::TpetraMap_To_EpetraMap(const Teuchos::RCP<const Tpetra_Map>& tpetraMap_, 
                                                      const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  Teuchos::ArrayView<const GO> indicesAV = tpetraMap_->getNodeElementList(); 
  size_t numElements = tpetraMap_->getNodeNumElements(); 
  Teuchos::RCP<Epetra_Map> epetraMap_ = Teuchos::rcp(new Epetra_Map(-1, numElements, indicesAV.getRawPtr(), 0, *comm_));
  return epetraMap_; 
}


//TpetraCrsGraph_To_TpetraCrsGraph: takes in Tpetra::CrsGraph object, converts it to its equivalent Epetra_CrsGraph object, 
//and returns an RCP pointer to this Epetra_CrsGraph
Teuchos::RCP<Epetra_CrsGraph> Petra::TpetraCrsGraph_To_EpetraCrsGraph(const Teuchos::RCP<const Tpetra_CrsGraph>& tpetraCrsGraph_, 
                                                                     const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  Teuchos::RCP<const Tpetra_Map> tpetraMap_ = tpetraCrsGraph_->getRowMap();
  Teuchos::RCP<const Epetra_Map> epetraMap_ = TpetraMap_To_EpetraMap(tpetraMap_, comm_); 
  GO maxEntries = tpetraCrsGraph_->getGlobalMaxNumRowEntries(); 
  Teuchos::RCP<Epetra_CrsGraph> epetraCrsGraph_= Teuchos::rcp(new Epetra_CrsGraph(Copy, *epetraMap_, maxEntries));
  size_t NumEntries = 0;
  GO col; 
  Teuchos::Array<LO> Indices; 
  for (size_t i = 0; i<tpetraCrsGraph_->getNodeNumRows(); i++) { 
     NumEntries = tpetraCrsGraph_->getNumEntriesInLocalRow(i);
     Indices.resize(NumEntries);
     tpetraCrsGraph_->getLocalRowCopy(i, Indices(), NumEntries);
     GO globalRow = tpetraMap_->getGlobalElement(i); 
     for (size_t j = 0; j<NumEntries; j++){ 
         col = tpetraCrsGraph_->getColMap()->getGlobalElement(Indices[j]);
         epetraCrsGraph_->InsertGlobalIndices(globalRow, 1, &col);
     }
  }
  epetraCrsGraph_->FillComplete();
  return epetraCrsGraph_; 
}


//TpetraVector_To_EpetraVector: copies Tpetra::Vector object into its analogous 
//Epetra_Vector object 
void Petra::TpetraVector_To_EpetraVector(const Teuchos::RCP<const Tpetra_Vector>& tpetraVector_,
                                  Epetra_Vector& epetraVector_, const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  //check if maps of epetraVector_ and tpetraVector_ are the same
  const Epetra_BlockMap epetraMap_ = epetraVector_.Map();
  Teuchos::RCP<const Tpetra_Map> tpetraMap_ = tpetraVector_->getMap();  
  Teuchos::RCP<Epetra_Map> tpetraMapE_ = TpetraMap_To_EpetraMap(tpetraMap_, comm_); 
  bool isSame = tpetraMapE_->SameAs(epetraMap_); 
  //if epetraVector_ and tpetraVector_ do not have the same map, throw an exception
  TEUCHOS_TEST_FOR_EXCEPTION((isSame != true),
                             std::logic_error,
                             "Error in Petra::TpetraVector_To_EpetraVector! Arguments Epetra_Vector and Tpetra::Vector do not have same map." <<  std::endl) ;

  //Copy tpetraVector_ to epetraVector_
  Teuchos::Array<ST> array(tpetraVector_->getMap()->getNodeNumElements());
  tpetraVector_->get1dCopy(array);
  for (size_t i=0; i<tpetraVector_->getMap()->getNodeNumElements(); ++i) 
     epetraVector_[i] = array[i];
}


//EpetraVector_To_TpetraVectorConst: copies Epetra_Vector to const Tpetra_Vector
Teuchos::RCP<const Tpetra_Vector> Petra::EpetraVector_To_TpetraVectorConst(const Epetra_Vector& epetraVector_, 
                                                               Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               Teuchos::RCP<KokkosNode>& nodeT_)
{
  size_t numElements = epetraVector_.Map().NumMyElements(); 
  GO *indices = epetraVector_.Map().MyGlobalElements(); 
  Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
  Teuchos::RCP<const Tpetra_Map> mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);
  ST *values = new ST[mapT->getGlobalNumElements()]; 
  epetraVector_.ExtractCopy(values);
  Teuchos::ArrayView<ST> valuesAV = Teuchos::arrayView(values, mapT->getGlobalNumElements());
  Teuchos::RCP<const Tpetra_Vector> tpetraVector_ = Teuchos::rcp(new Tpetra_Vector(mapT, valuesAV));
  return tpetraVector_; 
}

//EpetraVector_To_TpetraVectorNonConst: copies Epetra_Vector to non-const Tpetra_Vector
Teuchos::RCP<Tpetra_Vector> Petra::EpetraVector_To_TpetraVectorNonConst(const Epetra_Vector& epetraVector_, 
                                                               Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               Teuchos::RCP<KokkosNode>& nodeT_)
{
  size_t numElements = epetraVector_.Map().NumMyElements(); 
  GO *indices = epetraVector_.Map().MyGlobalElements(); 
  Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
  Teuchos::RCP<const Tpetra_Map> mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);
  ST *values = new ST[mapT->getGlobalNumElements()]; 
  epetraVector_.ExtractCopy(values);
  Teuchos::ArrayView<ST> valuesAV = Teuchos::arrayView(values, mapT->getGlobalNumElements());
  Teuchos::RCP<Tpetra_Vector> tpetraVector_ = Teuchos::rcp(new Tpetra_Vector(mapT, valuesAV));
  return tpetraVector_; 
}
