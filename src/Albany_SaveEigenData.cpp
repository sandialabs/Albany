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


#include "Albany_SaveEigenData.hpp"
#include "NOX_Abstract_MultiVector.H"
#include "NOX_Epetra_MultiVector.H"
#include "Epetra_Vector.h"
#include <string>

Albany::SaveEigenData::
SaveEigenData(Teuchos::ParameterList& locaParams, Teuchos::RCP<NOX::Epetra::Observer> observer)
   :  nsave(0)
{
  bool doEig = locaParams.sublist("Stepper").get("Compute Eigenvalues", false);
  if (doEig) nsave = locaParams.sublist("Stepper").
              sublist("Eigensolver").get("Save Eigenvectors",0);

  cout << "\nSaveEigenData: Will save up to " 
       << nsave << " eigenvectors." << endl;
  
  noxObserver = observer;
}

Albany::SaveEigenData::~SaveEigenData()
{
}

NOX::Abstract::Group::ReturnType
Albany::SaveEigenData::save(
		 Teuchos::RCP< std::vector<double> >& evals_r,
		 Teuchos::RCP< std::vector<double> >& evals_i,
		 Teuchos::RCP< NOX::Abstract::MultiVector >& evecs_r,
	         Teuchos::RCP< NOX::Abstract::MultiVector >& evecs_i)
{
  if (nsave==0) return NOX::Abstract::Group::Ok;

  Teuchos::RCP<NOX::Epetra::MultiVector> ne_r =
    Teuchos::rcp_dynamic_cast<NOX::Epetra::MultiVector>(evecs_r);
  Teuchos::RCP<NOX::Epetra::MultiVector> ne_i =
    Teuchos::rcp_dynamic_cast<NOX::Epetra::MultiVector>(evecs_i);
  Epetra_MultiVector& e_r = ne_r->getEpetraMultiVector();
  Epetra_MultiVector& e_i = ne_i->getEpetraMultiVector();

  int ns = nsave;
  if (ns > evecs_r->numVectors())
    ns = evecs_r->numVectors();

  for (int i=0; i<ns; i++) {
    if ((*evals_i)[i]==0) {
      cout << setprecision(8) 
           << "Eigenvalue " << i << " with value: " << (*evals_r)[i] 
           << "\n   Has Eigenvector: " << *(e_r(i)) << "\n" << endl;

      std::fstream file;
      char buf[100];
      sprintf(buf,"evec%d.txtdump",i);
      file.open (buf, fstream::out);
      file << setprecision(8) 
           << "# Eigenvalue " << i << " with value: " << (*evals_r)[i] 
           << "\n# Has Eigenvector: \n" << *(e_r(i)) << "\n" << endl;
      file.close();
      cout << "Saved to " << buf << endl;
    }
    else {
      cout << setprecision(8) 
           << "Eigenvalue " << i << " with value: " << (*evals_r)[i] 
           << " +  " << (*evals_i)[i] << " i \nHas Eigenvector Re, Im" 
           << *(e_r(i)) << "\n" << *(e_i(i)) << endl;
    }

    //export real part to exodus
    //noxObserver->observeSolution( *(e_r(i)) );
  }

  return NOX::Abstract::Group::Ok;
}
