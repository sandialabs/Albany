//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Petra_Converters.hpp"

#include "Epetra_LocalMap.h"

#include "Teuchos_as.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_TestForException.hpp"

#include <cstddef>
#include <stdexcept>
#include <ostream>

//TpetraMap_To_EpetraMap: takes in Tpetra::Map object, converts it to its equivalent Epetra_Map object,
//and returns an RCP pointer to this Epetra_Map
Teuchos::RCP<Epetra_Map> Petra::TpetraMap_To_EpetraMap(const Teuchos::RCP<const Tpetra_Map>& tpetraMap_,
                                                      const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  const int numElements = Teuchos::as<int>(tpetraMap_->getNodeNumElements());
  const int indexBase = Teuchos::as<int>(tpetraMap_->getIndexBase());
  if (tpetraMap_->isDistributed() || tpetraMap_->getComm()->getSize() == Teuchos::OrdinalTraits<int>::one()) {
    const Teuchos::ArrayView<const GO> indices = tpetraMap_->getNodeElementList();
    const int computeGlobalElements = -Teuchos::OrdinalTraits<int>::one();
    return Teuchos::rcp(new Epetra_Map(computeGlobalElements, numElements, indices.getRawPtr(), indexBase, *comm_));
  } else {
    return Teuchos::rcp(new Epetra_LocalMap(numElements, indexBase, *comm_));
  }
}

//EpetraMap_To_TpetraMap: takes in Epetra_Map object, converts it to its equivalent Tpetra::Map object,
//and returns an RCP pointer to this Tpetra::Map
Teuchos::RCP<const Tpetra_Map> Petra::EpetraMap_To_TpetraMap(const Teuchos::RCP<const Epetra_Map>& epetraMap_,
                                                      const Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                      const Teuchos::RCP<KokkosNode>& nodeT_)
{
  const std::size_t numElements = Teuchos::as<std::size_t>(epetraMap_->NumMyElements());
  const GO indexBase = Teuchos::as<GO>(epetraMap_->IndexBase());
  if (epetraMap_->DistributedGlobal() || epetraMap_->Comm().NumProc() == Teuchos::OrdinalTraits<int>::one()) {
    const Teuchos::ArrayView<const GO> indices(epetraMap_->MyGlobalElements(), numElements);
    const Tpetra::global_size_t computeGlobalElements = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    return Teuchos::rcp(new Tpetra_Map(computeGlobalElements, indices, indexBase, commT_, nodeT_));
  } else {
    return Teuchos::rcp(new Tpetra_Map(numElements, indexBase, commT_, Tpetra::LocallyReplicated, nodeT_));
  }
}


//TpetraCrsGraph_To_TpetraCrsGraph: takes in Tpetra::CrsGraph object, converts it to its equivalent Epetra_CrsGraph object,
//and returns an RCP pointer to this Epetra_CrsGraph
Teuchos::RCP<Epetra_CrsGraph> Petra::TpetraCrsGraph_To_EpetraCrsGraph(const Teuchos::RCP<const Tpetra_CrsGraph>& tpetraCrsGraph,
                                                                      const Teuchos::RCP<const Epetra_Comm>& comm,
                                                                      Tpetra::ProfileType pft)
{
  Teuchos::RCP<const Tpetra_Map> tpetraDomainMap = tpetraCrsGraph->getDomainMap();
  Teuchos::RCP<const Tpetra_Map> tpetraRangeMap  = tpetraCrsGraph->getRangeMap();
  Teuchos::RCP<const Epetra_Map> epetraDomainMap = TpetraMap_To_EpetraMap(tpetraDomainMap, comm);
  Teuchos::RCP<const Epetra_Map> epetraRangeMap  = TpetraMap_To_EpetraMap(tpetraRangeMap, comm);

  Teuchos::Array<LO> numRowEntries(tpetraCrsGraph->getNodeNumRows());
  for (std::size_t i = 0; i<tpetraCrsGraph->getNodeNumRows(); i++)
  {
    numRowEntries[i] = tpetraCrsGraph->getNumEntriesInLocalRow(i);
  }

  // If the user asks for static profile, but the graph is not filled, we automatically switch to dynamic profile
  bool staticProfile = tpetraCrsGraph->isFillActive() ? false : (pft==Tpetra::StaticProfile);
  if (pft==Tpetra::StaticProfile && tpetraCrsGraph->isFillActive())
    std::cout << "Warning [TpetraCrsGraph_To_EpetraCrsGraph]: input graph is not filled, but a static profile is requested. Switching to dynamic profile.\n";

  Teuchos::RCP<Epetra_CrsGraph> epetraCrsGraph= Teuchos::rcp(new Epetra_CrsGraph(Copy, *epetraRangeMap, numRowEntries.getRawPtr(), staticProfile));

  Teuchos::ArrayView<const GO> tIndices;
  GO col;
  if (tpetraCrsGraph->isLocallyIndexed())
  {
    Teuchos::RCP<const Tpetra_Map> tpetraColMap    = tpetraCrsGraph->getColMap();
    for (std::size_t i = 0; i<tpetraCrsGraph->getNodeNumRows(); ++i)
    {
       tpetraCrsGraph->getLocalRowView(i, tIndices);
       for (std::size_t j=0; j<tIndices.size(); ++j)
       {
          col = tpetraColMap->getGlobalElement(tIndices[j]);
          epetraCrsGraph->InsertGlobalIndices (epetraRangeMap->GID(i), 1, &col);
       }
    }

    // The input graph was locally indexed (i.e., filled), so we fill/optimize also the epetra graph.
    epetraCrsGraph->FillComplete(*epetraDomainMap, *epetraRangeMap);
    epetraCrsGraph->OptimizeStorage();
  }
  else
  {
    for (std::size_t i = 0; i<tpetraCrsGraph->getNodeNumRows(); ++i)
    {
       tpetraCrsGraph->getGlobalRowView(tpetraRangeMap->getGlobalElement(i), tIndices);
       for (std::size_t j=0; j<tIndices.size(); ++j)
       {
          col = tIndices[j];
          epetraCrsGraph->InsertGlobalIndices (epetraRangeMap->GID(i), 1, &col);
       }
    }
  }

  return epetraCrsGraph;
}

//TpetraCrsMatrix_To_EpetraCrsMatrix: copies Tpetra::CrsMatrix object into its analogous
//Epetra_CrsMatrix object
Teuchos::RCP<Epetra_CrsMatrix>
Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(const Teuchos::RCP<Tpetra_CrsMatrix>& tpetraCrsMatrix,
                                          const Teuchos::RCP<const Epetra_Comm>& comm)
{
  Teuchos::RCP<Epetra_CrsGraph> epetraCrsGraph = Petra::TpetraCrsGraph_To_EpetraCrsGraph(tpetraCrsMatrix->getCrsGraph(),comm,Tpetra::StaticProfile);
  Teuchos::RCP<Epetra_CrsMatrix> epetraCrsMatrix = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*epetraCrsGraph));

  GO col; ST val;
  Teuchos::ArrayView<const ST> tValues;
  int numEntries;
  double* eValues;
  if (tpetraCrsMatrix->isLocallyIndexed())
  {
    Teuchos::ArrayView<const LO> tIndices;
    Teuchos::RCP<const Tpetra_Map> tpetraColMap = tpetraCrsMatrix->getColMap();
    for (std::size_t i = 0; i<tpetraCrsMatrix->getNodeNumRows(); i++)
    {
       tpetraCrsMatrix->getLocalRowView(i, tIndices, tValues);
       epetraCrsMatrix->ExtractMyRowView(i, numEntries, eValues);
       for (std::size_t j=0; j<numEntries; ++j)
       {
          eValues[j] = tValues[j];
       }
    }
  }
  else
  {
    Teuchos::RCP<const Tpetra_Map> tpetraRangeMap  = tpetraCrsMatrix->getRangeMap();
    Teuchos::ArrayView<const GO> tIndices;
    for (std::size_t i = 0; i<tpetraCrsMatrix->getNodeNumRows(); i++)
    {
       GO row = tpetraRangeMap->getGlobalElement(i);
       tpetraCrsMatrix->getGlobalRowView(row, tIndices, tValues);
       epetraCrsMatrix->ExtractGlobalRowView(row, numEntries, eValues);
       for (std::size_t j=0; j<numEntries; ++j)
       {
          eValues[j] = tValues[j];
       }
    }
  }

  return epetraCrsMatrix;
}

//TpetraCrsMatrix_To_EpetraCrsMatrix: copies Tpetra::CrsMatrix object into its analogous
//Epetra_CrsMatrix object
void Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(const Teuchos::RCP<Tpetra_CrsMatrix>& tpetraCrsMatrix_,
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

  std::size_t NumEntries = 0;
  GO col; ST val;
  Teuchos::Array<LO> Indices;
  Teuchos::Array<ST> Values;
  for (std::size_t i = 0; i<tpetraCrsMatrix_->getNodeNumRows(); i++) {
     NumEntries = tpetraCrsMatrix_->getNumEntriesInLocalRow(i);
     Indices.resize(NumEntries);
     Values.resize(NumEntries);
     tpetraCrsMatrix_->getLocalRowCopy(i, Indices(), Values(), NumEntries);
     GO globalRow = tpetraCrsMatrix_->getRowMap()->getGlobalElement(i);
     for (std::size_t j=0; j<NumEntries; j++) {
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
  for (std::size_t i=0; i<tpetraVector_->getMap()->getNodeNumElements(); ++i)
     epetraVector_[i] = array[i];
}

void Petra::TpetraVector_To_EpetraVector(const Teuchos::RCP<const Tpetra_Vector>& tpetraVector_,
                                  Teuchos::RCP<Epetra_Vector>& epetraVector_, const Teuchos::RCP<const Epetra_Comm>& comm_)
{

  // Build the epetra vector if needed
  if(Teuchos::is_null(epetraVector_)){

    Teuchos::RCP<const Tpetra_Map> tpetraMap = tpetraVector_->getMap();
    Teuchos::RCP<Epetra_Map> emap = TpetraMap_To_EpetraMap(tpetraMap, comm_);
    epetraVector_ = Teuchos::rcp(new Epetra_Vector(*emap));

  }

  //Copy tpetraVector_ to epetraVector_
  Teuchos::Array<ST> array(tpetraVector_->getMap()->getNodeNumElements());
  tpetraVector_->get1dCopy(array);
  for (std::size_t i=0; i<tpetraVector_->getMap()->getNodeNumElements(); ++i)
     (*epetraVector_)[i] = array[i];

}

//TpetraMultiVector_To_EpetraMultiVector: copies Tpetra::MultiVector object into its analogous
//Epetra_MultiVector object
void Petra::TpetraMultiVector_To_EpetraMultiVector(const Teuchos::RCP<const Tpetra_MultiVector>& tpetraMV_,
                                  Epetra_MultiVector& epetraMV_, const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  std::size_t numVectors = tpetraMV_->getNumVectors();
  std::size_t localLength = tpetraMV_->getLocalLength();
  //Copy tpetraMV_ to epetraMV_
  Teuchos::ArrayRCP<const ST> array;
  for (std::size_t i = 0; i<numVectors; i++) {
     array = tpetraMV_->getData(i);
     for (std::size_t j = 0; j<localLength; j++) {
        epetraMV_[i][j] = array[j];
     }
  }
}

//EpetraVector_To_TpetraVectorConst: copies Epetra_Vector to const Tpetra_Vector
Teuchos::RCP<const Tpetra_Vector> Petra::EpetraVector_To_TpetraVectorConst(const Epetra_Vector& epetraVector_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               const Teuchos::RCP<KokkosNode>& nodeT_)
{
  std::size_t numElements = epetraVector_.Map().NumMyElements();
  GO *indices = epetraVector_.Map().MyGlobalElements();
  Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
  Teuchos::RCP<const Tpetra_Map> mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);
  ST *values;
  epetraVector_.ExtractView(&values);
  Teuchos::ArrayView<ST> valuesAV = Teuchos::arrayView(values, mapT->getNodeNumElements());
  Teuchos::RCP<const Tpetra_Vector> tpetraVector_ = Teuchos::rcp(new Tpetra_Vector(mapT, valuesAV));
  return tpetraVector_;
}

//EpetraMultiVector_To_TpetraMultiVector: copies Epetra_MultiVector to non-const Tpetra_MultiVector
Teuchos::RCP<Tpetra_MultiVector> Petra::EpetraMultiVector_To_TpetraMultiVector(const Epetra_MultiVector& epetraMV_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               const Teuchos::RCP<KokkosNode>& nodeT_)
{
  //get map from epetraMV_ and convert to Tpetra::Map
  int numElements = epetraMV_.Map().NumMyElements();
  GO *indices = epetraMV_.Map().MyGlobalElements();
  Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
  const Teuchos::RCP<const Tpetra_Map> mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);
  //copy values from epetraMV_
  int numVectors = epetraMV_.NumVectors();
  ST *values;
  int Length;
  epetraMV_.ExtractView(&values, &Length);
  Teuchos::ArrayView<ST> valuesAV = Teuchos::arrayView(values, Length*numVectors);
  //create Tpetra_MultiVector copy of epetraMV_
  Teuchos::RCP<Tpetra_MultiVector> tpetraMV_ = Teuchos::rcp(new Tpetra_MultiVector(mapT, valuesAV, Length, numVectors));
  return tpetraMV_;
}


//EpetraVector_To_TpetraVectorNonConst: copies Epetra_Vector to non-const Tpetra_Vector
Teuchos::RCP<Tpetra_Vector> Petra::EpetraVector_To_TpetraVectorNonConst(const Epetra_Vector& epetraVector_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               const Teuchos::RCP<KokkosNode>& nodeT_)
{
  std::size_t numElements = epetraVector_.Map().NumMyElements();
  GO *indices = epetraVector_.Map().MyGlobalElements();
  Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
  Teuchos::RCP<const Tpetra_Map> mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);
  ST *values;
  epetraVector_.ExtractView(&values);
  Teuchos::ArrayView<ST> valuesAV = Teuchos::arrayView(values, mapT->getNodeNumElements());
  Teuchos::RCP<Tpetra_Vector> tpetraVector_ = Teuchos::rcp(new Tpetra_Vector(mapT, valuesAV));
  return tpetraVector_;
}

//EpetraCrsMatrix_To_TpetraCrsMatrix: copies Epetra_CrsMatrix to its analogous Tpetra_CrsMatrix
Teuchos::RCP<Tpetra_CrsMatrix> Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(Epetra_CrsMatrix& epetraCrsMatrix_,
                                                               const Teuchos::RCP<const Teuchos::Comm<int> >& commT_,
                                                               const Teuchos::RCP<KokkosNode>& nodeT_)
{
    //get row map of Epetra::CrsMatrix & convert to Tpetra::Map
    Epetra_Map epetraMap_ = epetraCrsMatrix_.RowMap();
    std::size_t numElements = epetraMap_.NumMyElements();
    GO *indices = epetraMap_.MyGlobalElements();
    Teuchos::ArrayView<GO> indicesAV = Teuchos::arrayView(indices, numElements);
    Teuchos::RCP<const Tpetra_Map> tpetraMap_ = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesAV, commT_, nodeT_);

    //get CrsGraph of Epetra::CrsMatrix & convert to Tpetra::CrsGraph
    const Epetra_CrsGraph epetraCrsGraph_ = epetraCrsMatrix_.Graph();
    std::size_t maxEntries = epetraCrsGraph_.GlobalMaxNumIndices();
    Teuchos::RCP<Tpetra_CrsGraph> tpetraCrsGraph_ = Teuchos::rcp(new Tpetra_CrsGraph(tpetraMap_, maxEntries));
    Teuchos::Array<GO> col(1);
    for (int i=0; i<epetraCrsGraph_.NumMyRows(); i++) {
       int NumEntries;
       LO *Indices;
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
    for (std::size_t i=0; i<epetraCrsMatrix_.NumMyRows(); i++) {
       int NumEntries;
       ST *ValuesM;
       LO *IndicesM;
       epetraCrsMatrix_.ExtractMyRowView(i, NumEntries, ValuesM, IndicesM);
       GO globalRow = epetraCrsMatrix_.GRID(i);
       for (std::size_t j = 0; j<NumEntries; j++) {
           col[0] = IndicesM[j];
           val[0] = ValuesM[j];
           tpetraCrsMatrix_->replaceLocalValues(i, col(), val());
       }
    }
   tpetraCrsMatrix_->fillComplete();
   return tpetraCrsMatrix_;

}
}

#include "Albany_Utils.hpp"

Petra::Converter::Converter (const Teuchos::RCP<const Teuchos_Comm>& commT)
  : commT_(commT),
    commE_(Albany::createEpetraCommFromTeuchosComm(commT))
{}
