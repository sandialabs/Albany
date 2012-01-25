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


#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//#define ALBANY_CHECK_FPE

#ifdef ALBANY_CHECK_FPE
#include <math.h>
//#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
#endif

/////////////////////////////////////////////////////////////
// begin jrr: Now cull solution vector's (x) map using
// assumptions on various size and location parameters 

// Setting for function definition; parameters for now look like:
// RCP<Epetra_Vector>& x_map  (input)
// vector<int>& keepDOF   (input - vector of length total number dof/node in
//                                the solution vector; for now, number of equations
//                                per node. keepDOF[i]={0 - delete; 1 - keep})
// RCP<Epetra_Vector>& x_new_map (output - new map for culled solution vector)

void cullDistributedResponseMap( Teuchos::RCP<Epetra_Map>& x_map,
				 std::vector<int>& keepDOF,
				 Teuchos::RCP<Epetra_Map>& x_new_map )
{

  int numKeepDOF = accumulate(keepDOF.begin(), keepDOF.end(), 0);
  int Neqns = keepDOF.size();
  int N = x_map->NumMyElements(); // x_map is map for solution vector

  TEUCHOS_ASSERT( !(N % Neqns) ); // Need to be sure that N is exactly Neqns-divisible

  int nnodes = N / Neqns; // number of fem nodes
  int N_new = nnodes * numKeepDOF; // length of local x_new 

  std::vector<int> gids(N);
  std::vector<int> gids_new;

  x_map->MyGlobalElements(&gids[0]); // Fill local x_map into gids array
  
  for ( int inode = 0; inode < N/Neqns ; ++inode) // 
    {
      for ( int ieqn = 0; ieqn < Neqns; ++ieqn )
	if( keepDOF[ieqn] == 1 )  // then want to keep this dof
	  gids_new.push_back( gids[(inode*Neqns)+ieqn] );
    }
  // end cull
  
  x_new_map = Teuchos::rcp( new Epetra_Map( -1, gids_new.size(), &gids_new[0], 0, x_map->Comm() ) );
}

// now cull the response vector itself
void cullDistributedResponse( Teuchos::RCP<Epetra_Vector>& x,
			      Teuchos::RCP<Epetra_Map>& x_map,
			      Teuchos::RCP<Epetra_Vector>& x_new,
			      Teuchos::RCP<Epetra_Map>& x_new_map )
{
  Epetra_Import importer( *x_new_map, *x_map );
  x_new = Teuchos::rcp( new Epetra_Vector( *x_new_map ) );

  x_new->Import( *x, importer, Insert );
}

// end jrr
////////////////////////////////////////////////

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

#ifdef ALBANY_CHECK_FPE
	_mm_setcsr(_MM_MASK_MASK &~
		(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO) );
#endif

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  char * xmlfilename=0;
  char defaultfile[10]={"input.xml"};
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albany [inputfile.xml]\n");
      exit(1);
    }
    else
      xmlfilename=argv[1];
  }
  else
    xmlfilename=defaultfile;
  
  try {

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    Albany::SolverFactory slvrfctry(xmlfilename, Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    RCP<EpetraExt::ModelEvaluator> App = slvrfctry.create(appComm, appComm);

    EpetraExt::ModelEvaluator::InArgs params_in = App->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out = App->createOutArgs();
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
        *out << "Main: model supports sensitivities, so will request DgDp" << endl;
        //dgdp = rcp(new Epetra_MultiVector(p1->Map(), g1->GlobalLength() ));
        *out << " Num Responses: " << g1->GlobalLength() 
             << ",   Num Parameters: " << p1->GlobalLength() << endl;

        if (p1->GlobalLength() > 0)
          dgdp = rcp(new Epetra_MultiVector(g1->Map(), p1->GlobalLength() ));
        else
          supportsSensitivities = false;
      }
    }

    // Evaluate first model
    if (num_p > 0)  params_in.set_p(0,p1);
    if (num_g > 1)  responses_out.set_g(0,g1);
    responses_out.set_g(num_g-1,xfinal);

    if (supportsSensitivities) responses_out.set_DgDp(0,0,dgdp);
    setupTimer.~TimeMonitor();
    App->evalModel(params_in, responses_out);

    *out << "Finished eval of first model: Params, Responses " 
         << std::setprecision(12) << endl;

    /////////////////////////////////////////////////////////////////
    // begin jrr: Need the map for our global function call to cull
    //            the solution vector, which for this case is xfinal;
    //            resulting object is xfinal_new

    Teuchos::RCP<Epetra_Map> xfinal_map =
      rcp(new Epetra_Map( *(App->get_g_map(num_g-1)) ) );
    Teuchos::RCP<Epetra_Vector> xfinal_new;  // placeholder for culled response vector
    Teuchos::RCP<Epetra_Map> xfinal_new_map; // placeholder for culled map

    // Locate the vector components we want to keep manually.
    // For thermoelasticity2d problem, it's 2 and we want to
    // cull a temperature variable on each node in
    // the 3rd dof.

    int neq = 4;                 // thermoelasticity3d
    std::vector<int> keepDOF(neq, 1); // Initialize to keep all dof, then,
    keepDOF[3] = 0;              // as stated, cull the 4th dof

    int ndim = std::accumulate(keepDOF.begin(), keepDOF.end(), 0);
  
    // Now, create the new, culled map...
    cullDistributedResponseMap( xfinal_map, keepDOF, xfinal_new_map );

    // ...and with the new map, create culled solution vector:
    cullDistributedResponse( xfinal, xfinal_map, xfinal_new, xfinal_new_map );

    cout << "First 3*neq values of xfinal and xfinal_new:" << endl;
    {
      for ( int i=0; i < 3*neq; i++ )
	cout << "xfinal[" << i << "] = " << (*xfinal)[i] << "    " 
	     <<"xfinal_new[" << i << "] = " << (*xfinal_new)[i] << endl;
    }

    cout << "Last 3*neq values of xfinal and xfinal_new:" << endl;
    {
      int N = xfinal_map->NumMyElements();
      for ( int i=0; i < 3*neq; i++ )
	cout << "xfinal[" << N-i-1 << "] = " << (*xfinal)[N - (i+1)] << "    " 
	     <<"xfinal_new[" << ndim*N/neq - (i+1) << "] = " << (*xfinal_new)[ndim*N/neq - (i+1)] << endl;
    }

    // end jrr
    /////////////////////////////////////////////////////////////////

    
    if (num_p>0) p1->Print(*out << "\nParameters!\n");
    if (num_g>1) g1->Print(*out << "\nResponses!\n");
    if (supportsSensitivities)
      dgdp->Print(*out << "\nSensitivities!\n");
    double mnv; xfinal->MeanValue(&mnv);
    *out << "Main_Solve: MeanValue of final solution " << mnv << endl;

    //cout << "Final Solution \n" << *xfinal << endl;

    status += slvrfctry.checkTestResults(g1.get(), dgdp.get());
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}

