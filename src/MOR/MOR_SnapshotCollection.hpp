//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_SNAPSHOTCOLLECTION_HPP
#define MOR_SNAPSHOTCOLLECTION_HPP

#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"

#include <deque>

namespace MOR {

class MultiVectorOutputFile;

class SnapshotCollection {
public:
  SnapshotCollection(
      int period,
      const Teuchos::RCP<MultiVectorOutputFile> &snapshotFile);

  ~SnapshotCollection();
  void addVector(double stamp, const Epetra_Vector &value);

private:
  int period_;
  Teuchos::RCP<MultiVectorOutputFile> snapshotFile_;

  int skipCount_;
  std::deque<double> stamps_;
  std::deque<Epetra_Vector> snapshots_;

  // Disallow copy and assignment
  SnapshotCollection(const SnapshotCollection &);
  SnapshotCollection &operator=(const SnapshotCollection &);
};

} // namespace MOR

#endif /*MOR_SNAPSHOTCOLLECTION_HPP*/
