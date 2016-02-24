//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_GeneralizedCoordinatesOutput.hpp"

#include "MOR_EpetraLocalMapMVMatrixMarketUtils.hpp"

#include "Epetra_LocalMap.h"

#include "Teuchos_as.hpp"

namespace MOR {

GeneralizedCoordinatesOutput::GeneralizedCoordinatesOutput(
    const std::string &filename,
    const std::string &stampsFilename) :
  filename_(filename),
  stampsFilename_(stampsFilename),
  projectionComponents_(),
  stamps_()
{}

int
GeneralizedCoordinatesOutput::vectorCount() const
{
  return projectionComponents_.size();
}

void
GeneralizedCoordinatesOutput::vectorAdd(const Epetra_Vector &v)
{
  const double defaultStamp = static_cast<double>(this->vectorCount());
  this->stampedVectorAdd(defaultStamp, v);
}

void
GeneralizedCoordinatesOutput::stampedVectorAdd(double stamp, const Epetra_Vector &v)
{
  stamps_.push_back(stamp);
  projectionComponents_.push_back(v);
}

GeneralizedCoordinatesOutput::~GeneralizedCoordinatesOutput()
{
  // Warning: Destructor performs actual work and might fail !
  // Avoding such heresy without killing performance would require
  // to modify the EpetraExt Matrix Market I/O code.

  if (!projectionComponents_.empty()) {
    const Epetra_BlockMap commonMap = projectionComponents_.front().Map();
    Epetra_MultiVector allComponents(commonMap, projectionComponents_.size(), false);
    for (int i = 0; i < allComponents.NumVectors(); ++i) {
      *allComponents(i) = projectionComponents_[i];
    }
    writeLocalMapMultiVectorToMatrixMarket(filename_, allComponents);

    const Epetra_LocalMap scalarMap(1, 0, commonMap.Comm());
    Epetra_MultiVector allStamps(View, scalarMap, stamps_.getRawPtr(), 1, Teuchos::as<int>(stamps_.size()));
    writeLocalMapMultiVectorToMatrixMarket(stampsFilename_, allStamps);
  }
}

} // end namespace MOR
