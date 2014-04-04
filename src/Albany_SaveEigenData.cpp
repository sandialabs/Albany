//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SaveEigenData.hpp"
#include "Albany_EigendataInfoStruct.hpp"
#include "NOX_Abstract_MultiVector.H"
#include "NOX_Epetra_MultiVector.H"
#include "Epetra_Import.h"
#include "Epetra_Vector.h"
#include <string>

Albany::SaveEigenData::
SaveEigenData(Teuchos::ParameterList& locaParams, Teuchos::RCP<NOX::Epetra::Observer> observer, Albany::StateManager* pStateMgr)
  :  nsave(0),
     nSaveAsStates(0)
{
  bool doEig = locaParams.sublist("Stepper").get("Compute Eigenvalues", false);
  if (doEig) {
    nsave = locaParams.sublist("Stepper").
      sublist("Eigensolver").get("Save Eigenvectors",0);

    nSaveAsStates = nsave; //in future, perhaps allow this to be set in LOCA params?
  }

//IK, 4/4/14: commented out b/c there was too much output 
//  std::cout << "\nSaveEigenData: Will save up to " 
//       << nsave << " eigenvectors, and output "
//       << nSaveAsStates << " as states." << std::endl;
  
  noxObserver = observer;
  pAlbStateMgr = pStateMgr;
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
  using namespace std;

  if (nsave==0) return NOX::Abstract::Group::Ok;

  Teuchos::RCP<NOX::Epetra::MultiVector> ne_r =
    Teuchos::rcp_dynamic_cast<NOX::Epetra::MultiVector>(evecs_r);
  Teuchos::RCP<NOX::Epetra::MultiVector> ne_i =
    Teuchos::rcp_dynamic_cast<NOX::Epetra::MultiVector>(evecs_i);
  Epetra_MultiVector& e_r = ne_r->getEpetraMultiVector();
  Epetra_MultiVector& e_i = ne_i->getEpetraMultiVector();

  char buf[100];

  int ns = std::min(nsave, evecs_r->numVectors());

  // Store *overlapped* eigenvectors in state manager
  Teuchos::RCP<EigendataStruct> eigenData = Teuchos::rcp( new EigendataStruct );

  Teuchos::RCP<Albany::AbstractDiscretization> disc = 
    pAlbStateMgr->getDiscretization();

  eigenData->eigenvalueRe = evals_r;
  eigenData->eigenvalueIm = evals_i;

  eigenData->eigenvectorRe = 
    Teuchos::rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), ns));
  eigenData->eigenvectorIm =
    Teuchos::rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), ns));

  // Importer for overlapped data
  Teuchos::RCP<Epetra_Import> importer =
    Teuchos::rcp(new Epetra_Import(*(disc->getOverlapMap()), *(disc->getMap())));

  // Overlapped eigenstate vectors
  for(int i=0; i<ns; i++) {
    (*(eigenData->eigenvectorRe))(i)->Import( *(e_r(i)), *importer, Insert );
    (*(eigenData->eigenvectorIm))(i)->Import( *(e_i(i)), *importer, Insert );
  }

  pAlbStateMgr->setEigenData(eigenData);


  // Output to files
  std::fstream evecFile;
  std::fstream evalFile;
  evalFile.open ("evals.txtdump", std::fstream::out);
  evalFile << "# Eigenvalues: index, Re, Im" << std::endl;
  for (int i=0; i<ns; i++) {
    evalFile << i << "  " << (*evals_r)[i] << "  " << (*evals_i)[i] << std::endl;

    if ( fabs((*evals_i)[i]) == 0 ) {
      //Print to stdout -- good for debugging but too much output in most cases
      //std::cout << setprecision(8) 
      //     << "Eigenvalue " << i << " with value: " << (*evals_r)[i] 
      //     << "\n   Has Eigenvector: " << *(e_r(i)) << "\n" << std::endl;

      //write text format to evec<i>.txtdump file
      // sprintf(buf,"evec%d.txtdump",i);
      sprintf(buf,"evec%d.csv",i);
      evecFile.open (buf, std::fstream::out);
      evecFile << std::setprecision(8) 
           << "# Eigenvalue " << i << " with value: " << (*evals_r)[i] 
           << "\n# Has Eigenvector: \n" << *(e_r(i)) << "\n" << std::endl;
      evecFile.close();

      double norm; e_r(i)->Norm2(&norm);
      std::cout << "Saved to " << buf << " (norm = " << norm << ")" << std::endl;

      //export to exodus
      noxObserver->observeSolution( *(e_r(i)) , (*evals_r)[i]);
    }
    else {
      //Print to stdout -- good for debugging but too much output in most cases
      //std::cout << setprecision(8) 
      //     << "Eigenvalue " << i << " with value: " << (*evals_r)[i] 
      //     << " +  " << (*evals_i)[i] << " i \nHas Eigenvector Re, Im" 
      //     << *(e_r(i)) << "\n" << *(e_i(i)) << std::endl;

      //write text format to evec<i>.txtdump file
      // sprintf(buf,"evec%d.txtdump",i);
      sprintf(buf,"evec%d.csv",i);
      evecFile.open (buf, std::fstream::out);
      evecFile << std::setprecision(8) 
           << "# Eigenvalue " << i << " with value: " 
	   << (*evals_r)[i] <<" +  " << (*evals_i)[i] << "\n"
           << "# Has Eigenvector Re,Im: \n" 
	   << *(e_r(i)) << "\n" << *(e_i(i)) << "\n" << std::endl;
      evecFile.close();
      std::cout << "Saved Re, Im to " << buf << std::endl;

      //export real and imaginary parts to exodus
      noxObserver->observeSolution( *(e_r(i)), (*evals_r)[i] );
      noxObserver->observeSolution( *(e_i(i)), (*evals_i)[i] );
    }
  }
  evalFile.close();

  return NOX::Abstract::Group::Ok;
}
