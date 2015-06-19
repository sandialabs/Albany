//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_LINEARSYSTEM_HPP
#define GOAL_LINEARSYSTEM_HPP

#include "Albany_DataTypes.hpp"

namespace PHAL { class Workset; }

namespace GOAL {

class Discretization;

class LinearSystem
{
  public:
    LinearSystem(Teuchos::RCP<Discretization>& d);
    ~LinearSystem();
    void setWorksetSolutionInfo(PHAL::Workset& workset);
    void completeJacobianFill();
    void writeLinearSystem(int ctr);
  private:
    void init();
    void fillSolution();
    Teuchos::RCP<Tpetra_Vector> x;
    Teuchos::RCP<Tpetra_Vector> xdot;
    Teuchos::RCP<Tpetra_Vector> xdotdot;
    Teuchos::RCP<Tpetra_Vector> rhs;
    Teuchos::RCP<Tpetra_CrsMatrix> jac;
    Teuchos::RCP<Tpetra_Vector> overlapX;
    Teuchos::RCP<Tpetra_Vector> overlapXdot;
    Teuchos::RCP<Tpetra_Vector> overlapXdotdot;
    Teuchos::RCP<Tpetra_Vector> overlapRhs;
    Teuchos::RCP<Tpetra_CrsMatrix> overlapJac;
    Teuchos::RCP<Tpetra_Import> importer;
    Teuchos::RCP<Tpetra_Export> exporter;
    Teuchos::RCP<Discretization> disc;
};

}

#endif
