//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_SNAPSHOTCOLLECTIONOBSERVER_HPP
#define MOR_SNAPSHOTCOLLECTIONOBSERVER_HPP

#include "NOX_Epetra_Observer.H"

#include "MOR_SnapshotCollection.hpp"

#include "Teuchos_RCP.hpp"

namespace MOR {

class MultiVectorOutputFile;

class SnapshotCollectionObserver : public NOX::Epetra::Observer
{
public:
  SnapshotCollectionObserver(
      int period,
      const Teuchos::RCP<MultiVectorOutputFile> &snapshotFile);

  virtual void observeSolution(const Epetra_Vector& solution);
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  SnapshotCollection snapshotCollector_;

  // Disallow copy & assignment
  SnapshotCollectionObserver(const SnapshotCollectionObserver &);
  SnapshotCollectionObserver operator=(const SnapshotCollectionObserver &);
};

} // namespace MOR

#endif /*MOR_SNAPSHOTCOLLECTIONOBSERVER_HPP*/
