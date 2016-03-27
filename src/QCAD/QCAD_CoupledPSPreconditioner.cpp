//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_CoupledPSPreconditioner.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
#include "Epetra_LocalMap.h"


QCAD::CoupledPSPreconditioner::CoupledPSPreconditioner(int nEigenvals, 
						       const Teuchos::RCP<const Epetra_Map>& discretizationMap, 
						       const Teuchos::RCP<const Epetra_Map>& fullPSMap, 
						       const Teuchos::RCP<const Epetra_Comm>& comm)
{
  discMap = discretizationMap;
  domainMap = rangeMap = fullPSMap;
  myComm = comm;
  bUseTranspose = false;
  bInitialized = false;
  nEigenvalues = nEigenvals;
}

QCAD::CoupledPSPreconditioner::~CoupledPSPreconditioner()
{
}


//! Initialize the operator with everything needed to apply it
void QCAD::CoupledPSPreconditioner::initialize(const Teuchos::RCP<Epetra_Operator>& poissonPrecond, const Teuchos::RCP<Epetra_Operator>& schrodingerPrecond)
{
  // Set member variables
  poissonPreconditioner = poissonPrecond;
  schrodingerPreconditioner = schrodingerPrecond;

  // Prepare the distributed and local eigenvalue maps
  int num_discMap_myEls = discMap->NumMyElements();
  int my_nEigenvals = domainMap->NumMyElements() - num_discMap_myEls * (1+nEigenvalues);
  dist_evalMap  = Teuchos::rcp(new Epetra_Map(nEigenvalues, my_nEigenvals, 0, *myComm));
  //local_evalMap = Teuchos::rcp(new Epetra_LocalMap(nEigenvalues, 0, *myComm));
  //eval_importer = Teuchos::rcp(new Epetra_Import(*local_evalMap, *dist_evalMap));
}


//! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
int QCAD::CoupledPSPreconditioner::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{ 
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
  std::endl << "QCAD::CoupledPSPreconditioner::Apply is not implemented." << std::endl);
  return 0;
}

//! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
int QCAD::CoupledPSPreconditioner::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  //int disc_nGlobalElements = discMap->NumGlobalElements();
  int disc_nMyElements = discMap->NumMyElements();

  Teuchos::RCP<Epetra_Vector> x_poisson, y_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_schrodinger, y_schrodinger;
  Teuchos::RCP<Epetra_Vector> x_neg_evals, y_neg_evals;
  double *x_data, *y_data;

  for(int i=0; i < X.NumVectors(); i++) {

    // Split X(i) and Y(i) into vector views for Poisson and Schrodinger parts
    if(X(i)->ExtractView(&x_data) != 0 || Y(i)->ExtractView(&y_data) != 0) 
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				 "Error!  QCAD::CoupledPSPreconditioner -- cannot extract vector views");

    x_poisson = Teuchos::rcp(new Epetra_Vector(::View, *discMap, &x_data[0]));
    x_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *discMap, &x_data[disc_nMyElements], disc_nMyElements, nEigenvalues));
    x_neg_evals = Teuchos::rcp(new Epetra_Vector(::View, *dist_evalMap, &x_data[(1+nEigenvalues)*disc_nMyElements]));

    y_poisson = Teuchos::rcp(new Epetra_Vector(::View, *discMap, &y_data[0]));
    y_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *discMap, &y_data[disc_nMyElements], disc_nMyElements, nEigenvalues));
    y_neg_evals = Teuchos::rcp(new Epetra_Vector(::View, *dist_evalMap, &y_data[(1+nEigenvalues)*disc_nMyElements]));

    // Communicate all the x_evals to every processor, since all parts of the mesh need them
    //x_neg_evals_local->Import(*x_neg_evals, *eval_importer, Insert);


    // Preconditioner Matrix is:
    //
    //                   Phi                    Psi[i]                            -Eval[i]
    //          | ---------------------------------------------------------------------------------|
    //          |                      |                             |                             |
    // Poisson  |    Precond_poisson   |           0                 |               0             |
    //          |                      |                             |                             |
    //          | ---------------------------------------------------------------------------------|
    //          |                      |                             |                             |
    // Schro[j] |          0           |     Precond_schrodinger     |               0             |    
    //          |                      |                             |                             |
    //          | ---------------------------------------------------------------------------------|
    //          |                      |                             |                             |
    // Norm[j]  |          0           |            0                |           Identity          |
    //          |                      |                             |                             |
    //          | ---------------------------------------------------------------------------------|
    //
    //

    // Do multiplication block-wise

    // y_poisson =     
    //poissonPreconditioner->Apply(*x_poisson, *y_poisson);
    poissonPreconditioner->ApplyInverse(*x_poisson, *y_poisson); //Ifpack operators: call ApplyInverse to apply preconditioner

    // y_schrodinger =
    for(int j=0; j<nEigenvalues; j++) {
      //schrodingerPreconditioner->Apply( *((*x_schrodinger)(j)), *((*y_schrodinger)(j)) ); // y_schrodinger[j] = H * x_schrodinger[j]
      schrodingerPreconditioner->ApplyInverse( *((*x_schrodinger)(j)), *((*y_schrodinger)(j)) ); // y_schrodinger[j] = H * x_schrodinger[j]
    }
      
    // y_neg_evals =
    y_neg_evals->Scale(1.0, *x_neg_evals);
  }
  return 0;
}
