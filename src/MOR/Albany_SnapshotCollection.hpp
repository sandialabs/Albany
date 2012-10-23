//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_SNAPSHOTCOLLECTION_HPP
#define ALBANY_SNAPSHOTCOLLECTION_HPP

#include "Albany_MultiVectorOutputFileFactory.hpp"

#include "Epetra_Vector.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <deque>
#include <string>
#include <cstddef>

namespace Albany {

class SnapshotCollection {
public:
  explicit SnapshotCollection(const Teuchos::RCP<Teuchos::ParameterList> &params);

  ~SnapshotCollection();
  void addVector(double stamp, const Epetra_Vector &value);

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;
  static Teuchos::RCP<Teuchos::ParameterList> fillDefaultParams(const Teuchos::RCP<Teuchos::ParameterList> &params);
  
  MultiVectorOutputFileFactory snapshotFileFactory_;
  
  std::size_t period_;
  void initPeriod();

  std::size_t skipCount_;
  std::deque<double> stamps_;
  std::deque<Epetra_Vector> snapshots_;

  // Disallow copy and assignment
  SnapshotCollection(const SnapshotCollection &);
  SnapshotCollection &operator=(const SnapshotCollection &);
};

} // end namespace Albany

#endif /*ALBANY_SNAPSHOTCOLLECTION_HPP*/
