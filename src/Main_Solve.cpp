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

/////////////////////////////////////////////////////////////
// begin jrr: Now cull solution, xfinal, using assumptions on
// various size and location parameters 

// Setting for function definition; parameters for now look like:
// RCP<Epetra_Vector>& xfinal (input)
// Epetra_Vector& xfinal_map (input)
// int ndim (input - physical dimension, 2d, 3d, etc)
// int neq  (input - number of equations/node; phys dim+temp+pressure, e.g.)
// RCP<Epetra_Vector>& xfinal_new (output - new, culled solution vector)

void cullDistributedResponse( Teuchos::RCP<Epetra_Vector>& xfinal,
			      Epetra_Map& xfinal_map,
			      int ndim,
			      int neq,
			      int offset,
			      Teuchos::RCP<Epetra_Vector>& xfinal_new );
// end jrr
////////////////////////////////////////////////

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

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

    Epetra_Map xfinal_map = *(App->get_g_map(num_g-1));
    RCP<Epetra_Vector> xfinal_new; // placeholder for now; filled in function
  
    // Assume ndim for now; manually change when going from 2d to
    // 3d, etc
    int ndim = 3;     // For thermoelasticity2d problem, it's 2
    int neq = ndim+1; // Again, for thermoelasticity problem,
                      // there's a temperature variable on each node
    int offset = 0;   // Locate the vector components we want to keep
  
    cullDistributedResponse( xfinal, xfinal_map, ndim, neq, offset, xfinal_new );

    cout << "First 3*neq values of xfinal and xfinal_new:" << endl;
    {
      for ( int i=0; i < 3*neq; i++ )
	cout << "xfinal[" << i << "] = " << (*xfinal)[i] << "    " 
	     <<"xfinal_new[" << i << "] = " << (*xfinal_new)[i] << endl;
    }

    cout << "Last 3*neq values of xfinal and xfinal_new:" << endl;
    {
      int N = xfinal_map.NumMyElements();
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
/////////////////////////////////////////////////////////////
// begin jrr: Now cull solution, xfinal, using assumptions on
// various size and location parameters 

// Setting for function definition; parameters for now look like:
// RCP<Epetra_Vector>& xfinal (input)
// Epetra_Vector& xfinal_map (input)
// int ndim (input - physical dimension, 2d, 3d, etc)
// int neq  (input - number of equations/node; phys dim+temp+pressure, e.g.)
// RCP<Epetra_Vector>& xfinal_new (output - new, culled solution vector)

void cullDistributedResponse( Teuchos::RCP<Epetra_Vector>& xfinal,
			      Epetra_Map& xfinal_map,
			      int ndim,
			      int neq,
			      int offset,
			      Teuchos::RCP<Epetra_Vector>& xfinal_new )
{
  int numToErase = neq - ndim; // number of dof to erase/node
                               // Assume these dof reside either at the
                               // or the bottom of the nodes dof

  int N = xfinal_map.NumMyElements(); // xfinal_map def'd above, outside block
  std::vector<int> gids(N);

  xfinal_map.MyGlobalElements(&gids[0]);
  
  // cull using criteria and STL entities
  // Have to go from the back of the vector to the front
  {
    std::vector<int>::iterator gids_begin = gids.begin();
    for ( std::vector<int>::iterator gids_iter = gids.end(); gids_iter >  gids_begin ; --gids_iter)
      {
	if ( ((int)( gids_iter - gids_begin) + 1) % neq  == 0 )
	  for (int icount = 0; icount < numToErase; ++icount ) // solution vector contains auxiliary info;
	    {                                                  // may need to erase more than one dof/node

	      gids.erase( gids_iter - offset*ndim - icount);  // crude start to the
		   				              // offset problem: Assumes what we want
	                                                      // to keep is either at the top or the
	                                                      // bottom of the dof vector/node
						              // (Note: still need to
						              // check this out more carefully!)
	    }
      }
  }
  // end cull
  
  Epetra_Map xfinal_map_new( -1, gids.size(), &gids[0], 0, xfinal_map.Comm() );
  Epetra_Import importer( xfinal_map_new, xfinal_map );

  xfinal_new = Teuchos::rcp(new Epetra_Vector( xfinal_map_new ));
  xfinal_new->Import( *xfinal, importer, Insert );
}
// end jrr
////////////////////////////////////////////////

