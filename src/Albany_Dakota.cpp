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


#ifdef ALBANY_DAKOTA
#include <iostream>
#include "Albany_Dakota.hpp"
#include "Albany_Utils.hpp"

using namespace std;

// Standard use case for TriKota
//   Dakota is run in library mode with its interface
//   implemented with an EpetraExt::ModelEvaluator
int Albany_Dakota()
{ // Assumes MPI_Init() already called, and using MPI_COMM_WORLD

  cout << "\nStarting Albany_Dakota!" << endl;

  // Construct driver with default file names
  TriKota::Driver dakota;

  // Construct a ModelEvaluator for your application with the
  // MPI_Comm chosen by Dakota. This example ModelEvaluator 
  // only takes an input file name and MPI_Comm to construct,
  // and must not be constructed if Dakota assigns MPI_COMM_NULL.
  
  Teuchos::RCP<Albany::SolverFactory> slvrfctry;
  Teuchos::RCP<EpetraExt::ModelEvaluator> App;

  MPI_Comm analysis_comm = dakota.getAnalysisComm();

  if (analysis_comm != MPI_COMM_NULL) {
    slvrfctry = Teuchos::rcp(new Albany::SolverFactory("input.xml", analysis_comm));
    Teuchos::RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(analysis_comm);
    App = slvrfctry->create(appComm, appComm);
  } else {
    cout << " Got  MPI_COMM_NULL\n" << endl;
  }

  // Construct a concrete Dakota interface with an EpetraExt::ModelEvaluator
  // trikota_interface is freed in the destructor for the Dakota interface class.
  Teuchos::RCP<TriKota::DirectApplicInterface> trikota_interface =
    Teuchos::rcp(new TriKota::DirectApplicInterface(dakota.getProblemDescDB(), App), false);

  // Run the requested Dakota strategy using this interface
  dakota.run(trikota_interface.get());

  if (dakota.rankZero()) {
    Dakota::RealVector finalValues = dakota.getFinalSolution().continuous_variables();
    cout << "\nAlbany_Dakota: Final Values from Dakota = " 
         << setprecision(8) << finalValues << endl;

    return slvrfctry->checkTestResults(NULL, NULL, &finalValues);
  }
  else return 0;
}
#endif
