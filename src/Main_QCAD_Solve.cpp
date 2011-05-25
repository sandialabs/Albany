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


#include <iostream>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow

void CreateSolver(char* xmlfilename, Teuchos::RCP<Albany::Application>& albApp, Teuchos::RCP<EpetraExt::ModelEvaluator>& App, 
		  EpetraExt::ModelEvaluator::InArgs& params_in, EpetraExt::ModelEvaluator::OutArgs& responses_out);
void SolveModel(Teuchos::RCP<Albany::Application>& albApp, Teuchos::RCP<EpetraExt::ModelEvaluator>& App, 
		EpetraExt::ModelEvaluator::InArgs params_in, EpetraExt::ModelEvaluator::OutArgs responses_out,
		Teuchos::RCP<std::vector<Albany::StateVariables> >& initialStates, 
		Teuchos::RCP<std::vector<Albany::StateVariables> >& finalStates);
void CopyState(Teuchos::RCP<std::vector<Albany::StateVariables> >& dest, 
	       Teuchos::RCP<std::vector<Albany::StateVariables> >& src,
	       std::string stateNameToCopy);
bool checkConvergence(Teuchos::RCP<std::vector<Albany::StateVariables> >& newStates, 
		      Teuchos::RCP<std::vector<Albany::StateVariables> >& oldStates,
		      std::string stateNameToCompare, double tol);


int main(int argc, char *argv[]) {

  int status=0;
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  char * InitPoissonXmlFilename=0;
  char * PoissonXmlFilename=0;
  char * SchrodingerXmlFilename=0;
  if(argc>3){
    if(!strcmp(argv[1],"--help")){
      printf("albanyQCAD [InitPoissonInputfile.xml] [PoissonInputfile.xml] [SchrodingerInputfile.xml]\n");
      exit(1);
    }
    else {
      InitPoissonXmlFilename=argv[1];
      PoissonXmlFilename=argv[2];
      SchrodingerXmlFilename=argv[3];
    }
  }
  else {
    printf("albanyQCAD [InitPoissonInputfile.xml] [PoissonInputfile.xml] [SchrodingerInputfile.xml]\n");
    exit(1);
  }
  
  try {
    RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

    RCP<Albany::Application> initPoissonApp, poissonApp, schrodingerApp;
    RCP<EpetraExt::ModelEvaluator> initPoissonSolver, poissonSolver, schrodingerSolver;

    EpetraExt::ModelEvaluator::InArgs initPoisson_params_in, 
                                      poisson_params_in,
                                      schrodinger_params_in;

    EpetraExt::ModelEvaluator::OutArgs initPoisson_responses_out, 
                                       poisson_responses_out, 
                                       schrodinger_responses_out;

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    *out << "QCAD Solve: creating initial Poisson solver using input " << InitPoissonXmlFilename << endl;
    CreateSolver(InitPoissonXmlFilename, initPoissonApp, initPoissonSolver, 
		 initPoisson_params_in, initPoisson_responses_out);

    *out << "QCAD Solve: creating Poisson solver using input " << PoissonXmlFilename << endl;
    CreateSolver(PoissonXmlFilename, poissonApp, poissonSolver, 
		 poisson_params_in, poisson_responses_out);

    *out << "QCAD Solve: creating Schrodinger solver using input " << SchrodingerXmlFilename << endl;
    CreateSolver(SchrodingerXmlFilename, schrodingerApp, schrodingerSolver, 
		 schrodinger_params_in, schrodinger_responses_out);

    setupTimer.~TimeMonitor();

    //state variables
    RCP< std::vector<Albany::StateVariables> > statesToPass = Teuchos::null;
    RCP< std::vector<Albany::StateVariables> > statesToLoop = Teuchos::null;

    RCP< std::vector<Albany::StateVariables> > lastSavedPotential = 
      rcp(new std::vector<Albany::StateVariables>);


    *out << "QCAD Solve: Initial Poisson solve (no quantum region) " << endl;
    SolveModel(initPoissonApp, initPoissonSolver, 
	       initPoisson_params_in, initPoisson_responses_out,
	       statesToPass, statesToLoop);

    *out << "QCAD Solve: Beginning Poisson-Schrodinger solve loop" << endl;
    bool bConverged = false; 
    int iter = 0;
    int maxIter = 0;
    do {
      *out << "QCAD Solve: Schrodinger iteration " << iter << endl;
      SolveModel(schrodingerApp, schrodingerSolver, 
		 schrodinger_params_in, schrodinger_responses_out,
		 statesToLoop, statesToPass);

      *out << "QCAD Solve: Poisson iteration " << iter << endl;
      SolveModel(poissonApp, poissonSolver, 
		 poisson_params_in, poisson_responses_out,
		 statesToPass, statesToLoop);

      //Test single iteration first, then test these
      //bConverged = checkConvergence(statesToLoop, lastSavedPotential, "potential", 1e-3);
      //CopyState(lastSavedPotential, statesToLoop, "potential");
      iter++;
    } while(!bConverged && iter <= maxIter);
    *out << "QCAD Solve: Completed Poisson-Schrodinger solve loop after " << iter << " iterations." << endl;

    //*out << "Finished eval of first model: Params, Responses " 
    //     << std::setprecision(12) << endl;
    //if (num_p>0) p1->Print(*out << "\nParameters!\n");
    //if (num_g>1) g1->Print(*out << "\nResponses!\n");
    //if (supportsSensitivities)
    //  dgdp->Print(*out << "\nSensitivities!\n");
    //double mnv; xfinal->MeanValue(&mnv);
    //*out << "Main_Solve: MeanValue of final solution " << mnv << endl;
    //
    ////cout << "Final Solution \n" << *xfinal << endl;
    //
    //status += slvrfctry.checkTestResults(g1.get(), dgdp.get());
    //*out << "\nNumber of Failed Comparisons: " << status << endl;


  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  //Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}

void CreateSolver(char* xmlfilename, Teuchos::RCP<Albany::Application>& albApp, Teuchos::RCP<EpetraExt::ModelEvaluator>& App, 
		EpetraExt::ModelEvaluator::InArgs& params_in, EpetraExt::ModelEvaluator::OutArgs& responses_out)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

    Albany::SolverFactory slvrfctry(xmlfilename, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    App = slvrfctry.createAndGetAlbanyApp(albApp, appComm, appComm);

    params_in = App->createInArgs();
    responses_out = App->createOutArgs();
    int num_p = params_in.Np();     // Number of *vectors* of parameters
    int num_g = responses_out.Ng(); // Number of *vectors* of responses
    RCP<Epetra_Vector> p1;
    RCP<Epetra_Vector> g1;

    if (num_p > 0)
      p1 = rcp(new Epetra_Vector(*(App->get_p_init(0))));
    if (num_g > 1)
      g1 = rcp(new Epetra_Vector(*(App->get_g_map(0))));
    RCP<Epetra_Vector> xfinal =
      rcp(new Epetra_Vector(*(App->get_g_map(num_g-1)),true) );

    // Sensitivity Analysis stuff
    bool supportsSensitivities = false;
    RCP<Epetra_MultiVector> dgdp;

    if (num_p>0 && num_g>1) {
      supportsSensitivities =
        !responses_out.supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, 0, 0).none();

      if (supportsSensitivities) {
        //*out << "Main: model supports sensitivities, so will request DgDp" << endl;
        //dgdp = rcp(new Epetra_MultiVector(p1->Map(), g1->GlobalLength() ));
        //*out << " Num Responses: " << g1->GlobalLength() 
        //     << ",   Num Parameters: " << p1->GlobalLength() << endl;

        if (p1->GlobalLength() > 0)
          dgdp = rcp(new Epetra_MultiVector(g1->Map(), p1->GlobalLength() ));
        else
          supportsSensitivities = false;
      }
    }
    
    if (num_p > 0)  params_in.set_p(0,p1);
    if (num_g > 1)  responses_out.set_g(0,g1);
    responses_out.set_g(num_g-1,xfinal);

    if (supportsSensitivities) responses_out.set_DgDp(0,0,dgdp);
}

void SolveModel(Teuchos::RCP<Albany::Application>& albApp, Teuchos::RCP<EpetraExt::ModelEvaluator>& App, 
		EpetraExt::ModelEvaluator::InArgs params_in, EpetraExt::ModelEvaluator::OutArgs responses_out,
		Teuchos::RCP<std::vector<Albany::StateVariables> >& initialStates, 
		Teuchos::RCP<std::vector<Albany::StateVariables> >& finalStates)
{
  albApp->getStateMgr().reinitializeStateVariables( initialStates, albApp->getNumWorksets() );
  App->evalModel(params_in, responses_out);
  finalStates = albApp->getStateMgr().getAllOldStateVariables();
}



void CopyState(Teuchos::RCP<std::vector<Albany::StateVariables> >& dest, 
	       Teuchos::RCP<std::vector<Albany::StateVariables> >& src,
	       std::string stateNameToCopy)
{
  int numWorksets = src->size();

  if(dest->size() != (unsigned int)numWorksets)
    dest->resize(numWorksets); //deallocation necessary?

  for (int ws = 0; ws < numWorksets; ws++)
  {
    Albany::StateVariables& srcForWorkset = (*src)[ws];
    Albany::StateVariables::iterator st = srcForWorkset.begin();

    while (st != srcForWorkset.end())
    {
      std::string srcStateName = st->first;

      if(srcStateName == stateNameToCopy) {

        // we assume operating on the last two indices is correct
        std::vector<PHX::DataLayout::size_type> dims;
        srcForWorkset[srcStateName]->dimensions(dims);

        int size = dims.size();
        TEST_FOR_EXCEPTION(size != 4, std::logic_error,
            "Something is wrong during copy state variable operation");
        int cells = dims[0];
        int qps = dims[1];
        int dim = dims[2];
        int dim2 = dims[3];

        TEST_FOR_EXCEPT( ! (dim == dim2) );

	//allocate space in destination if necessary -- will RCP take care of freeing if assign to already alloc'd?
	std::vector<PHX::DataLayout::size_type> destDims;
	(*dest)[ws][srcStateName]->dimensions(destDims);
	if( dims[0] != destDims[0] || dims[1] != destDims[1] || dims[2] != destDims[2] || dims[3] != destDims[3])
	  (*dest)[ws][srcStateName] = Teuchos::rcp(new Intrepid::FieldContainer<RealType>(dims));

        for (int cell = 0; cell < cells; ++cell)
        {
          for (int qp = 0; qp < qps; ++qp)
          {
            for (int i = 0; i < dim; ++i)
            {
              (*((*dest)[ws][srcStateName]))(cell, qp, i, i) = (*(st->second))(cell, qp, i, i);
            }
          }
        }
      }
      st++;
    }
  }
}



bool checkConvergence(Teuchos::RCP<std::vector<Albany::StateVariables> >& newStates, 
		      Teuchos::RCP<std::vector<Albany::StateVariables> >& oldStates,
		      std::string stateNameToCompare, double tol)
{
  int numWorksets = oldStates->size();
  TEST_FOR_EXCEPT( ! ((unsigned int)numWorksets == newStates->size()) );
  
  for (int ws = 0; ws < numWorksets; ws++)
  {
    Albany::StateVariables& newStateVarsForWorkset = (*newStates)[ws];
    Albany::StateVariables& oldStateVarsForWorkset = (*oldStates)[ws];

    // we assume operating on the last two indices is correct
    std::vector<PHX::DataLayout::size_type> dims;
    oldStateVarsForWorkset[stateNameToCompare]->dimensions(dims);

    int size = dims.size();
    TEST_FOR_EXCEPTION(size != 4, std::logic_error,
		       "Something is wrong during copy state variable operation");
    int cells = dims[0];
    int qps = dims[1];
    int dim = dims[2];
    int dim2 = dims[3];
    TEST_FOR_EXCEPT( ! (dim == dim2) );

    for (int cell = 0; cell < cells; ++cell)  {
      for (int qp = 0; qp < qps; ++qp)  {
	for (int i = 0; i < dim; ++i) {
	  if( fabs( (*(newStateVarsForWorkset[stateNameToCompare]))(cell, qp, i, i) -
		    (*(oldStateVarsForWorkset[stateNameToCompare]))(cell, qp, i, i) ) > tol )
	    return false;
	}
      }
    }
  }
  return true;
}
