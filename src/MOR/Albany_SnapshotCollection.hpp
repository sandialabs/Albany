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
