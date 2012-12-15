//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_PROJECTIONERROROBSERVER_HPP
#define ALBANY_PROJECTIONERROROBSERVER_HPP

#include "NOX_Epetra_Observer.H"

#include "Albany_ProjectionError.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

class ReducedSpace;
class MultiVectorOutputFile;

class ProjectionErrorObserver : public NOX::Epetra::Observer
{
public:
  ProjectionErrorObserver(
      const Teuchos::RCP<ReducedSpace> &projectionSpace,
      const Teuchos::RCP<MultiVectorOutputFile> &errorFile,
      const Teuchos::RCP<NOX::Epetra::Observer> &decoratedObserver);

  //! Calls underlying observer then evalates projection error
  virtual void observeSolution(const Epetra_Vector& solution);

  //! Calls underlying observer then evalates projection error
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  ProjectionError projectionError_;

  Teuchos::RCP<NOX::Epetra::Observer> decoratedObserver_;

  // Disallow copy & assignment
  ProjectionErrorObserver(const ProjectionErrorObserver &);
  ProjectionErrorObserver operator=(const ProjectionErrorObserver &);
};

} // end namespace Albany

#endif /*ALBANY_PROJECTIONERROROBSERVER_HPP*/
