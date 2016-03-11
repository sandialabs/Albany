//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ProjectionError.hpp"

#include "MOR_MultiVectorOutputFile.hpp"
#include "MOR_ReducedSpace.hpp"

#include "Epetra_Comm.h"
#include "Epetra_LocalMap.h"
#include "Epetra_Vector.h"

// TODO remove dependency
#include <iostream>

namespace MOR {

namespace { // anonymous

const int ZERO_BASED_INDEXING = 0;
const bool NO_INIT = false;

} // anonymous namespace

ProjectionError::ProjectionError(
    const Teuchos::RCP<ReducedSpace> &projectionSpace,
    const Teuchos::RCP<MultiVectorOutputFile> &errorFile) :
  projectionSpace_(projectionSpace),
  errorFile_(errorFile)
{
  // Nothing to do
}

// TODO: Do no actual work in the destructor
ProjectionError::~ProjectionError()
{
#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif
  Epetra_LocalMap entryMap(
      static_cast<GlobalIndex>(relativeErrorNorms_.size()),
      ZERO_BASED_INDEXING,
      projectionSpace_->comm());
  Epetra_Vector entries(entryMap, NO_INIT);

  for (int i = 0; i < entries.MyLength(); ++i) {
    entries[i] = relativeErrorNorms_[i];
  }

  errorFile_->write(entries);
}

const Epetra_Comm &ProjectionError::projectionBasisComm() const {
  return projectionSpace_->comm();
}

void ProjectionError::process(const Epetra_MultiVector &v)
{
  // components <- orthonormalBasis^T * v
  const Teuchos::RCP<const Epetra_MultiVector> components = projectionSpace_->reduction(v);

  // absoluteError <- orthonormalBasis * components - v
  const Teuchos::RCP<Epetra_MultiVector> absoluteError = projectionSpace_->expansion(*components);
  absoluteError->Update(-1.0, v, 1.0);

  // Norm computations
  const Epetra_LocalMap normMap(components->NumVectors(), ZERO_BASED_INDEXING, projectionSpace_->comm());
  Epetra_Vector absoluteErrorNorm(normMap, NO_INIT);
  absoluteError->Norm2(absoluteErrorNorm.Values());

  Epetra_Vector referenceNorm(normMap, NO_INIT);
  v.Norm2(referenceNorm.Values());

  // ||relative_error|| <- ||absolute_error|| / ||v||
  Epetra_Vector relativeErrorNorm(normMap, NO_INIT);
  relativeErrorNorm.ReciprocalMultiply(1.0, referenceNorm, absoluteErrorNorm, 0.0);

  // Collect output data
  for (int i = 0; i < relativeErrorNorm.MyLength(); ++i) {
    relativeErrorNorms_.push_back(relativeErrorNorm[i]);
  }

  // Write to standard output
  // TODO remove
  components->Print(std::cout);
  referenceNorm.Print(std::cout);
  absoluteErrorNorm.Print(std::cout);
  relativeErrorNorm.Print(std::cout);
}

} // namespace MOR
