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

#ifndef ALBANY_SNAPSHOTCOLLECTIONOBSERVER_HPP
#define ALBANY_SNAPSHOTCOLLECTIONOBSERVER_HPP

#include "NOX_Epetra_Observer.H"

#include "Albany_SnapshotCollection.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

class SnapshotCollectionObserver : public NOX::Epetra::Observer
{
public:
  SnapshotCollectionObserver(const Teuchos::RCP<Teuchos::ParameterList> &params,
                             const Teuchos::RCP<NOX::Epetra::Observer> &decoratedObserver);

  //! Calls underlying observer then perform snapshot collection
  virtual void observeSolution(const Epetra_Vector& solution);
  
  //! Calls underlying observer then perform snapshot collection
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  Teuchos::RCP<NOX::Epetra::Observer> decoratedObserver_;

  SnapshotCollection snapshotCollector_;

  // Disallow copy & assignment
  SnapshotCollectionObserver(const SnapshotCollectionObserver &);
  SnapshotCollectionObserver operator=(const SnapshotCollectionObserver &);
};

} // end namespace Albany

#endif /*ALBANY_SNAPSHOTCOLLECTIONOBSERVER_HPP*/
