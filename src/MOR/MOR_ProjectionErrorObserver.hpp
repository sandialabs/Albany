//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_PROJECTIONERROROBSERVER_HPP
#define MOR_PROJECTIONERROROBSERVER_HPP

#include "NOX_Epetra_Observer.H"

#include "MOR_ProjectionError.hpp"

#include "Teuchos_RCP.hpp"

namespace MOR {

class ReducedSpace;
class MultiVectorOutputFile;

class ProjectionErrorObserver : public NOX::Epetra::Observer
{
public:
  ProjectionErrorObserver(
      const Teuchos::RCP<ReducedSpace> &projectionSpace,
      const Teuchos::RCP<MultiVectorOutputFile> &errorFile);

  virtual void observeSolution(const Epetra_Vector& solution);
  virtual void observeSolution(const Epetra_Vector& solution, double time_or_param_val);

private:
  ProjectionError projectionError_;

  // Disallow copy & assignment
  ProjectionErrorObserver(const ProjectionErrorObserver &);
  ProjectionErrorObserver operator=(const ProjectionErrorObserver &);
};

} // namespace MOR

#endif /*MOR_PROJECTIONERROROBSERVER_HPP*/
