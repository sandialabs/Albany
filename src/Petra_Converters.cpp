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

//EpetraMap_To_TpetraMap: takes in Epetra_Map object, converts it to its equivalent Tpetra::Map object,
//and returns an RCP pointer to this Tpetra::Map
Teuchos::RCP<const Tpetra_Map> Petra::EpetraMap_To_TpetraMap(const Teuchos::RCP<const Epetra_Map>& epetraMap_,
                                                      Teuchos::RCP<const Tpetra::Comm<int> >& commT_,
                                                      Teuchos::RCP<KokkosNode>& nodeT_)
{
  int numElements = epetraMap_->NumMyElements();
  int *indices = new int[numElements];
  epetraMap_->MyGlobalElements(indices);
  Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
  Teuchos::RCP<const Tpetra_Map> tpetraMap_ = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);
  return tpetraMap_;
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

//TpetraCrsMatrix_To_EpetraCrsMatrix: copies Tpetra::CrsMatrix object into its analogous
//Epetra_CrsMatrix object
void Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(Teuchos::RCP<Tpetra_CrsMatrix>& tpetraCrsMatrix_,
                                                  Epetra_CrsMatrix& epetraCrsMatrix_,
                                                  const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  //check if row maps of epetraCrsMatrix_ and tpetraCrsMatrix_ are the same
  const Epetra_BlockMap epetraMap_ = epetraCrsMatrix_.RowMap();
  Teuchos::RCP<const Tpetra_Map> tpetraMap_ = tpetraCrsMatrix_->getRowMap();
  Teuchos::RCP<Epetra_Map> tpetraMapE_ = TpetraMap_To_EpetraMap(tpetraMap_, comm_);
  bool isSame = tpetraMapE_->SameAs(epetraMap_);
  //if epetraCrsMatrix_ and tpetraCrsMatrix_ do not have the same row map, throw an exception
  TEUCHOS_TEST_FOR_EXCEPTION((isSame != true),
                             std::logic_error,
                             "Error in Petra::TpetraCrsMatrix_To_EpetraCrsMatrix! Arguments Epetra_CrsMatrix and Tpetra::CrsMatrix do not have same row map." <<  std::endl) ;

  size_t NumEntries = 0;
  GO col; ST val;
  Teuchos::Array<LO> Indices;
  Teuchos::Array<ST> Values;
  for (size_t i = 0; i<tpetraCrsMatrix_->getNodeNumRows(); i++) {
     NumEntries = tpetraCrsMatrix_->getNumEntriesInLocalRow(i);
     Indices.resize(NumEntries);
     Values.resize(NumEntries);
     tpetraCrsMatrix_->getLocalRowCopy(i, Indices(), Values(), NumEntries);
     GO globalRow = tpetraCrsMatrix_->getRowMap()->getGlobalElement(i);
     for (size_t j=0; j<NumEntries; j++) {
        col = tpetraCrsMatrix_->getColMap()->getGlobalElement(Indices[j]);
        val = Values[j];
        epetraCrsMatrix_.ReplaceGlobalValues(globalRow, 1, &val, &col);
     }
  }
}


//TpetraVector_To_EpetraVector: copies Tpetra::Vector object into its analogous
//Epetra_Vector object
void Petra::TpetraVector_To_EpetraVector(const Teuchos::RCP<const Tpetra_Vector>& tpetraVector_,
                                  Epetra_Vector& epetraVector_, const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  //check if maps of epetraVector_ and tpetraVector_ are the same
  //const Epetra_BlockMap epetraMap_ = epetraVector_.Map();
  //Teuchos::RCP<const Tpetra_Map> tpetraMap_ = tpetraVector_->getMap();
  //Teuchos::RCP<Epetra_Map> tpetraMapE_ = TpetraMap_To_EpetraMap(tpetraMap_, comm_);
  //bool isSame = tpetraMapE_->SameAs(epetraMap_);
  //if epetraVector_ and tpetraVector_ do not have the same map, throw an exception
  //TEUCHOS_TEST_FOR_EXCEPTION((isSame != true),
  //                          std::logic_error,
  //                          "Error in Petra::TpetraVector_To_EpetraVector! Arguments Epetra_Vector and Tpetra::Vector do not have same map." <<  std::endl) ;

  //Copy tpetraVector_ to epetraVector_
  Teuchos::Array<ST> array(tpetraVector_->getMap()->getNodeNumElements());
  tpetraVector_->get1dCopy(array);
  for (size_t i=0; i<tpetraVector_->getMap()->getNodeNumElements(); ++i)
     epetraVector_[i] = array[i];
}

//TpetraMultiVector_To_EpetraMultiVector: copies Tpetra::MultiVector object into its analogous
//Epetra_MultiVector object
void Petra::TpetraMultiVector_To_EpetraMultiVector(const Teuchos::RCP<const Tpetra_MultiVector>& tpetraMV_,
                                  Epetra_MultiVector& epetraMV_, const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  size_t numVectors = tpetraMV_->getNumVectors();
  size_t localLength = tpetraMV_->getLocalLength();
  //Copy tpetraMV_ to epetraMV_
  Teuchos::ArrayRCP<const ST> array;
  for (size_t i = 0; i<numVectors; i++) {
     array = tpetraMV_->getData(i);
     for (size_t j = 0; j<localLength; j++) {
        epetraMV_[i][j] = array[j];
     }
  }
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

//EpetraMultiVector_To_TpetraMultiVector: copies Epetra_MultiVector to non-const Tpetra_MultiVector
Teuchos::RCP<Tpetra_MultiVector> Petra::EpetraMultiVector_To_TpetraMultiVector(const Epetra_MultiVector& epetraMV_,
                                                               Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               Teuchos::RCP<KokkosNode>& nodeT_)
{
  //get map from epetraMV_ and convert to Tpetra::Map
  int numElements = epetraMV_.Map().NumMyElements();
  GO *indices = epetraMV_.Map().MyGlobalElements();
  Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
  const Teuchos::RCP<const Tpetra_Map> mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);
  //copy values from epetraMV_
  int Length = epetraMV_.MyLength();
  int numVectors = epetraMV_.NumVectors();
  ST *values = new ST[Length*numVectors];
  epetraMV_.ExtractCopy(values, Length);
  Teuchos::ArrayView<ST> valuesAV = Teuchos::arrayView(values, Length*numVectors);
  //create Tpetra_MultiVector copy of epetraMV_
  Teuchos::RCP<Tpetra_MultiVector> tpetraMV_ = Teuchos::rcp(new Tpetra_MultiVector(mapT, valuesAV, Length, numVectors));
  return tpetraMV_;
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

//EpetraCrsMatrix_To_TpetraCrsMatrix: copies Epetra_CrsMatrix to its analogous Tpetra_CrsMatrix
Teuchos::RCP<Tpetra_CrsMatrix> Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(Epetra_CrsMatrix& epetraCrsMatrix_,
                                                               Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               Teuchos::RCP<KokkosNode>& nodeT_)
{
    //get row map of Epetra::CrsMatrix & convert to Tpetra::Map
    Epetra_Map epetraMap_ = epetraCrsMatrix_.RowMap();
    size_t numElements = epetraMap_.NumMyElements();
    GO *indices = epetraMap_.MyGlobalElements();
    Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
    Teuchos::RCP<const Tpetra_Map> tpetraMap_ = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);

    //get CrsGraph of Epetra::CrsMatrix & convert to Tpetra::CrsGraph
    const Epetra_CrsGraph epetraCrsGraph_ = epetraCrsMatrix_.Graph();
    size_t maxEntries = epetraCrsGraph_.GlobalMaxNumIndices();
    Teuchos::RCP<Tpetra_CrsGraph> tpetraCrsGraph_ = Teuchos::rcp(new Tpetra_CrsGraph(tpetraMap_, maxEntries));
    int NumEntries = 0;
    Teuchos::Array<GO> col(1);
    for (int i=0; i<epetraCrsGraph_.NumMyRows(); i++) {
       NumEntries = epetraCrsGraph_.NumMyIndices(i);
       LO *Indices = new LO[NumEntries];
       epetraCrsGraph_.ExtractMyRowView(i, NumEntries, Indices);
       GO globalRow = epetraCrsGraph_.GRID(i);
       for (int j = 0; j<NumEntries; j++) {
          col[0] = epetraCrsGraph_.GCID(Indices[j]);
          tpetraCrsGraph_->insertGlobalIndices(globalRow, col);
       }
    }
    tpetraCrsGraph_->fillComplete();

    //convert Epetra::CrsMatrix to Tpetra::CrsMatrix, after creating Tpetra::CrsMatrix based on above Tpetra::CrsGraph
     Teuchos::RCP<Tpetra_CrsMatrix> tpetraCrsMatrix_ = Teuchos::rcp(new Tpetra_CrsMatrix(tpetraCrsGraph_));
    tpetraCrsMatrix_->setAllToScalar(0.0);
    tpetraCrsMatrix_->fillComplete();
    tpetraCrsMatrix_->resumeFill();
    Teuchos::Array<ST> val(1);
    for (size_t i=0; i<epetraCrsMatrix_.NumMyRows(); i++) {
       NumEntries = epetraCrsMatrix_.NumMyEntries(i);
       ST *ValuesM = new ST[NumEntries];
       LO *IndicesM = new LO[NumEntries];
       epetraCrsMatrix_.ExtractMyRowView(i, NumEntries, ValuesM, IndicesM);
       GO globalRow = epetraCrsMatrix_.GRID(i);
       for (size_t j = 0; j<NumEntries; j++) {
           col[0] = IndicesM[j];
           val[0] = ValuesM[j];
           tpetraCrsMatrix_->replaceLocalValues(i, col(), val());
       }
    }
   tpetraCrsMatrix_->fillComplete();
   return tpetraCrsMatrix_;

}


