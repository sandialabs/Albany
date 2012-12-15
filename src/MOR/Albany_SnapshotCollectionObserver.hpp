//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_SNAPSHOTCOLLECTIONOBSERVER_HPP
#define ALBANY_SNAPSHOTCOLLECTIONOBSERVER_HPP

#include "NOX_Epetra_Observer.H"

#include "Albany_SnapshotCollection.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

class MultiVectorOutputFile;

class SnapshotCollectionObserver : public NOX::Epetra::Observer
{
public:
  SnapshotCollectionObserver(
      int period,
      const Teuchos::RCP<MultiVectorOutputFile> &snapshotFile,
      const Teuchos::RCP<NOX::Epetra::Observer> &decoratedObserver);

  //! Calls underlying observer then perform snapshot collection
  virtual void observeSolution(const Epetra_Vector& solution);

  //! Calls underlying observer then perform snapshot collection
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  SnapshotCollection snapshotCollector_;

  Teuchos::RCP<NOX::Epetra::Observer> decoratedObserver_;

  // Disallow copy & assignment
  SnapshotCollectionObserver(const SnapshotCollectionObserver &);
  SnapshotCollectionObserver operator=(const SnapshotCollectionObserver &);
};

} // end namespace Albany

#endif /*ALBANY_SNAPSHOTCOLLECTIONOBSERVER_HPP*/
