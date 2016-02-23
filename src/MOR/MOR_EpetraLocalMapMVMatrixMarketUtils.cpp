//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_EpetraLocalMapMVMatrixMarketUtils.hpp"

#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_MultiVectorIn.h"

#include "Epetra_LocalMap.h"
#include "Epetra_Export.h"
#include "Epetra_Import.h"

#include "Teuchos_TestForException.hpp"

namespace MOR {

namespace Detail {

bool isRegularMap(const Epetra_BlockMap &candidate)
{
  return candidate.ConstantElementSize() && candidate.ElementSize() == 1;
}

bool hasMatchingLocalAndGlobalIDs(const Epetra_BlockMap &candidate)
{
  return candidate.MinAllGID() == 0 && (candidate.MaxAllGID() + 1 == candidate.NumGlobalElements());
}

bool isRegularMapLocal(const Epetra_BlockMap &regularCandidate)
{
  return hasMatchingLocalAndGlobalIDs(regularCandidate) && !regularCandidate.DistributedGlobal();
}

bool isLocalMap(const Epetra_Map &candidate)
{
  return isRegularMapLocal(candidate);
}

bool isLocalMap(const Epetra_BlockMap &candidate)
{
  return isRegularMap(candidate) && isRegularMapLocal(candidate);
}

Epetra_Map makeMasterMap(const Epetra_Comm &comm, int vectorSize)
{
  const int myElementCount = comm.MyPID() == 0 ? vectorSize : 0;
  return Epetra_Map(vectorSize, myElementCount, 0, comm);
}

} // end namespace Detail

using namespace Detail;

void writeLocalMapMultiVectorToMatrixMarket(
    const std::string &fileName,
    const Epetra_MultiVector &localMapMv)
{
  const Epetra_BlockMap sourceMap = localMapMv.Map();
  TEUCHOS_TEST_FOR_EXCEPT(!isLocalMap(sourceMap));

  const Epetra_BlockMap masterMap = makeMasterMap(sourceMap.Comm(), sourceMap.NumGlobalElements());
  const Epetra_Export exportFromSourceToMaster(sourceMap, masterMap);

  Epetra_MultiVector masterMv(masterMap, localMapMv.NumVectors(), true);
  masterMv.Export(localMapMv, exportFromSourceToMaster, Insert);

  {
    const int ierr = EpetraExt::MultiVectorToMatrixMarketFile(fileName.c_str(), masterMv);
    TEUCHOS_TEST_FOR_EXCEPT(ierr != 0);
  }
}


Teuchos::RCP<Epetra_MultiVector> readLocalMapMultiVectorFromMatrixMarket(
  const std::string &fileName, const Epetra_Comm &comm, int vectorSize)
{
  const Epetra_Map masterMap = makeMasterMap(comm, vectorSize);

  Teuchos::RCP<Epetra_MultiVector> masterMv;
  {
    Epetra_MultiVector *masterMvRawPtr;
    const int ierr = EpetraExt::MatrixMarketFileToMultiVector(fileName.c_str(), masterMap, masterMvRawPtr);
    TEUCHOS_TEST_FOR_EXCEPT(ierr != 0);
    masterMv = Teuchos::rcp(masterMvRawPtr);
  }

  const Epetra_LocalMap localMap(vectorSize, 0, comm);
  const Epetra_Import importer(localMap, masterMap);
  const Teuchos::RCP<Epetra_MultiVector> result(new Epetra_MultiVector(localMap, masterMv->NumVectors()));
  result->Import(*masterMv, importer, Insert);

  return result;
}

} // end namespace MOR
