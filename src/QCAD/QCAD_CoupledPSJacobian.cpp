//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_CoupledPSJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"

QCAD::CoupledPSJacobian::CoupledPSJacobian(int nEigenvals, const Teuchos::RCP<const Epetra_Map>& discretizationMap, 
		  const Teuchos::RCP<const Epetra_Map>& fullPSMap, const Teuchos::RCP<const Epetra_Comm>& comm)
{
  discMap = discretizationMap;
  domainMap = rangeMap = fullPSMap;
  myComm = comm;
  bUseTranspose = false;
  bInitialized = false;
}

QCAD::CoupledPSJacobian::~CoupledPSJacobian()
{
}


//! Initialize the operator with everything needed to apply it
void QCAD::CoupledPSJacobian::initialize(const Teuchos::RCP<Epetra_CrsMatrix>& poissonJac, const Teuchos::RCP<Epetra_CrsMatrix>& schrodingerJac, 
					 const Teuchos::RCP<Epetra_CrsMatrix>& massMatrix,
					 const Teuchos::RCP<Epetra_Vector>& eigenvals, const Teuchos::RCP<Epetra_MultiVector>& eigenvecs)
{
  poissonJacobian = poissonJac;
  schrodingerJacobian = schrodingerJac;
  overlapMatrix = massMatrix;
  eigenvalues = eigenvals;
  eigenvectors = eigenvecs;
}


//! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
int QCAD::CoupledPSJacobian::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{ 
  //int disc_nGlobalElements = discMap->NumGlobalElements();
  int disc_nMyElements = discMap->NumMyElements();
  int nEigenvals = eigenvalues->GlobalLength();

  Teuchos::RCP<Epetra_Vector> x_poisson, y_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_schrodinger, y_schrodinger;
  double *x_data, *y_data;

  for(int i=0; i < X.NumVectors(); i++) {

    // Split X(i) and Y(i) into vector views for Poisson and Schrodinger parts
    if(X(i)->ExtractView(&x_data) != 0 || Y(i)->ExtractView(&y_data) != 0) 
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				 "Error!  QCAD::CoupledPSJacobian -- cannot extract vector views");

    x_poisson = Teuchos::rcp(new Epetra_Vector(::View, *discMap, &x_data[0]));
    x_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *discMap, &x_data[disc_nMyElements], disc_nMyElements, nEigenvals));

    y_poisson = Teuchos::rcp(new Epetra_Vector(::View, *discMap, &y_data[0]));
    y_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *discMap, &y_data[disc_nMyElements], disc_nMyElements, nEigenvals));

    // Do multiplication block-wise
    // y_poisson = poissonJacobian * x_poisson + ...

    //HERE -- need to encode logic to multiply X by full Poisson-Schrodinger Jacobian and return result in Y
  }
  return 0;
}

//! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
int QCAD::CoupledPSJacobian::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       std::endl << "QCAD::CoupledPSJacobian::ApplyInverse is not implemented." << std::endl);
    return 0;
}

