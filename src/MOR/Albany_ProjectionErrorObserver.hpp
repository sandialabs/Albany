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
#include "Teuchos_ParameterList.hpp"

class Epetra_Map;

namespace Albany {

class ProjectionErrorObserver : public NOX::Epetra::Observer
{
public:
  ProjectionErrorObserver(const Teuchos::RCP<Teuchos::ParameterList> &params,
                          const Teuchos::RCP<NOX::Epetra::Observer> &decoratedObserver,
                          const Teuchos::RCP<const Epetra_Map> &decoratedMap);

  //! Calls underlying observer then evalates projection error
  virtual void observeSolution(const Epetra_Vector& solution);
  
  //! Calls underlying observer then evalates projection error
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  Teuchos::RCP<NOX::Epetra::Observer> decoratedObserver_;

  ProjectionError projectionError_;

  // Disallow copy & assignment
  ProjectionErrorObserver(const ProjectionErrorObserver &);
  ProjectionErrorObserver operator=(const ProjectionErrorObserver &);
};

} // end namespace Albany

#endif /*ALBANY_PROJECTIONERROROBSERVER_HPP*/
