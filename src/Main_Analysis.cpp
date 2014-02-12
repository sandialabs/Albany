//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <iostream>

#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Thyra_EpetraModelEvaluator.hpp"
#include "Piro_PerformAnalysis.hpp"
#include "Thyra_VectorBase.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"


int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  char * xmlfilename=0;
  char defaultfile[20]={"inputAnalysis.xml"};
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albanyAnalysis [inputfile.xml]\n");
      exit(1);
    }
    else
      xmlfilename=argv[1];
  }
  else
    xmlfilename=defaultfile;


  try {
    Teuchos::RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("AlbanyAnalysis: ***Total Time***");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer

    using namespace std;

    *out << "\nStarting Albany Analysis via Piro!" << std::endl;

    // Construct a ModelEvaluator for your application;
  
    Teuchos::RCP<Albany::SolverFactory> slvrfctry =
      Teuchos::rcp(new Albany::SolverFactory(xmlfilename, Albany_MPI_COMM_WORLD));
    Teuchos::RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    Teuchos::RCP<EpetraExt::ModelEvaluator> App = slvrfctry->create(appComm, appComm);


    Thyra::EpetraModelEvaluator appThyra;
    appThyra.initialize(App, Teuchos::null);

    Teuchos::RCP< Thyra::VectorBase<double> > p;

    // If no analysis section set in input file, default to simple "Solve"
    std::string analysisPackage = slvrfctry->getAnalysisParameters().get("Analysis Package","Solve");
    status = Piro::PerformAnalysis(appThyra, slvrfctry->getAnalysisParameters(), p); 

//    Dakota::RealVector finalValues = dakota.getFinalSolution().continuous_variables();
//    std::cout << "\nAlbany_Dakota: Final Values from Dakota = " 
//         << setprecision(8) << finalValues << std::endl;

    status =  slvrfctry->checkAnalysisTestResults(0, p);

    // Regression comparisons for Dakota runs only valid on Proc 0.
    if (mpiSession.getRank()>0)  status=0;
    else *out << "\nNumber of Failed Comparisons: " << status << std::endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;
  
  Teuchos::TimeMonitor::summarize(std::cout, false, true, false);
  return status;
}
