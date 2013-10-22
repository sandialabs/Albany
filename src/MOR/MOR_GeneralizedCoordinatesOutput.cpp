//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_GeneralizedCoordinatesOutput.hpp"

#include "MOR_EpetraLocalMapMVMatrixMarketUtils.hpp"

namespace MOR {

GeneralizedCoordinatesOutput::GeneralizedCoordinatesOutput(const std::string &filename) :
  filename_(filename),
  projectionComponents_()
{}

int
GeneralizedCoordinatesOutput::vectorCount() const
{
  return projectionComponents_.size();
}

void
GeneralizedCoordinatesOutput::vectorAdd(const Epetra_Vector &v)
{
  projectionComponents_.push_back(v);
}

GeneralizedCoordinatesOutput::~GeneralizedCoordinatesOutput()
{
  // Warning: Destructor performs actual work and might fail !
  // Avoding such heresy without killing performance would require
  // rewriting the EpetraExt Matrix Market I/O code.

  if (!projectionComponents_.empty()) {
    const Epetra_BlockMap commonMap = projectionComponents_.front().Map();
    Epetra_MultiVector allComponents(commonMap, projectionComponents_.size(), false);
    for (int i = 0; i < allComponents.NumVectors(); ++i) {
      *allComponents(i) = projectionComponents_[i];
    }
    writeLocalMapMultiVectorToMatrixMarket(filename_, allComponents);
  }
}

} // end namespace MOR
