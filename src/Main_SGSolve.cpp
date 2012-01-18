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
#include "Albany_NOXObserver.hpp"

#include "Piro_Epetra_StokhosSolver.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Stokhos_PCEAnasaziKL.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////
// begin jrr
//

// Functions to cull solution vector's (sg_u) map using
// assumptions on various size and location parameters 
//
// Setting for function definition; parameters for now look like:
// RCP<Epetra_Vector>& x_map  (input)
// vector<int>& keepDOF   (input - vector of length total number dof/node in
//                                the solution vector; for now, number of equations
//                                per node. keepDOF[i]={0 - delete; 1 - keep})
// RCP<Epetra_Vector>& x_new_map (output - new map for culled solution vector)

void cullDistributedResponseMap( Teuchos::RCP<const Epetra_BlockMap>& x_map,
				 vector<int>& keepDOF,
				 Teuchos::RCP<Epetra_BlockMap>& x_new_map )
{

  int numKeepDOF = accumulate(keepDOF.begin(), keepDOF.end(), 0);
  int Neqns = keepDOF.size();
  int N = x_map->NumMyElements(); // x_map is map for solution vector

  TEUCHOS_ASSERT( !(N % Neqns) ); // Assume that all the equations for
                                  // a given node are on the assigned
                                  // processor. I.e. need to ensure
                                  // that N is exactly Neqns-divisible

  int nnodes = N / Neqns;          // number of fem nodes
  int N_new = nnodes * numKeepDOF; // length of local x_new 

  std::vector<int> gids(N);
  std::vector<int> gids_new;

  x_map->MyGlobalElements(&gids[0]); // Fill local x_map into gids array
  
  for ( int inode = 0; inode < N/Neqns ; ++inode) // For every node 
    {
      for ( int ieqn = 0; ieqn < Neqns; ++ieqn )  // Check every dof on the node
	if( keepDOF[ieqn] == 1 )  // then want to keep this dof
	  gids_new.push_back( gids[(inode*Neqns)+ieqn] );
    }
  // end cull
  
  x_new_map = Teuchos::rcp( new Epetra_BlockMap( -1, gids_new.size(), &gids_new[0],
						  1, 0, x_map->Comm() ) );
}

// now cull the response vector itself
void cullDistributedResponse( Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& sg_u,
			      Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& sg_u_new,
			      Teuchos::RCP<Epetra_BlockMap>& sg_u_new_coeff_map )
{

  // Fill in basic info for the RCP<Stokhos::EpetraVectorOrthogPoly>
  // object, sg_u_new, using pertinent stuff from the original object,
  // sg_u, and the culled coeff_mapp...
  sg_u_new = Teuchos::rcp( new Stokhos::EpetraVectorOrthogPoly( sg_u->basis(),
								sg_u->map(),
								sg_u_new_coeff_map,
								sg_u->productComm() ) );

  // ...then, finally, import the data from sg_u into new object,
  // sg_u_new:
  Epetra_Import importer( *(sg_u_new->productMap()), *(sg_u->productMap()));

  importer.Print(std::cout);

  sg_u_new->getBlockVector()->Import(*(sg_u->getBlockVector()), importer,
					       Insert);
}


// Global function to encapsulate KL solution computation...
//

bool KL_OnSolutionMultiVector( 
  const Teuchos::RCP<Piro::Epetra::StokhosSolver>& App_sg, 
  const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& sg_u,
  const int NumKL,
  Teuchos::Array<double>& evals,
  Teuchos::RCP<Epetra_MultiVector>& evecs)
{

  /*
  Teuchos::RCP<EpetraExt::BlockVector> X;
  X = Teuchos::rcp(new EpetraExt::BlockVector((*sg_u)[0].Map(),
					      sg_u->getBlockVector()->Map()));
  sg_u->assignToBlockVector(*X);
  */

  Teuchos::RCP<EpetraExt::BlockVector> X_ov = 
    App_sg->get_sg_model()->import_solution( *(sg_u->getBlockVector()) );
  Teuchos::RCP<const EpetraExt::BlockVector> cX_ov = X_ov;

  // pceKL is object with member functions that explicitly call anasazi
  Stokhos::PCEAnasaziKL pceKL(cX_ov, *(sg_u->basis()), NumKL);

  // Set parameters for anasazi
  Teuchos::ParameterList anasazi_params = pceKL.getDefaultParams();
  //anasazi_params.set("Num Blocks", 10);
  //anasazi_params.set("Step Size", 50);
  anasazi_params.set("Verbosity",  
		     Anasazi::FinalSummary + 
		     //Anasazi::StatusTestDetails + 
		     //Anasazi::IterationDetails + 
		     Anasazi::Errors + 
		     Anasazi::Warnings);

  // Self explanatory
  bool result = pceKL.computeKL(anasazi_params);
   
  // Retrieve evals/evectors into return argument slots...
  evals = pceKL.getEigenvalues();
  evecs = pceKL.getEigenvectors();

  return result;
}

// end jrr
////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  char * xmlfilename=0;
  char * sg_xmlfilename=0;
  char defaultfile[10]={"input.xml"};
  char sg_defaultfile[12]={"inputSG.xml"};
  bool do_initial_guess;
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albanySG [inputfile.xml inputfileSG.xml]\n");
      exit(1);
    }
    else {
      if (argc == 2) {
	sg_xmlfilename = argv[1];
	do_initial_guess = false;
      }
      else {
	xmlfilename=argv[1];
	sg_xmlfilename = argv[2];
	do_initial_guess = true;
      }
    }
  }
  else {
    xmlfilename=defaultfile;
    sg_xmlfilename=sg_defaultfile;
    do_initial_guess = true;
  }
       
  
  try {

    Teuchos::RCP<Teuchos::Time> totalTime =
      Teuchos::TimeMonitor::getNewTimer("AlbanySG: ***Total Time***");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer

    // Setup communication objects
    Teuchos::RCP<Epetra_Comm> globalComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    // Parse parameters
    Albany::SolverFactory sg_slvrfctry(sg_xmlfilename, Albany_MPI_COMM_WORLD);
    Teuchos::ParameterList& albanyParams = sg_slvrfctry.getParameters();
    Teuchos::RCP< Teuchos::ParameterList> piroParams = 
      Teuchos::rcp(&(albanyParams.sublist("Piro")),false);
    
    // Create stochastic Galerkin solver
    Teuchos::RCP<Piro::Epetra::StokhosSolver> sg_solver =
      Teuchos::rcp(new Piro::Epetra::StokhosSolver(piroParams, globalComm));

    // Get comm for spatial problem
    Teuchos::RCP<const Epetra_Comm> app_comm = sg_solver->getSpatialComm();

    // Compute initial guess if requested
    Teuchos::RCP<Epetra_Vector> ig;
    if (do_initial_guess) {

      // Create solver
      Albany::SolverFactory slvrfctry(
	xmlfilename,
	Albany::getMpiCommFromEpetraComm(*app_comm));
      Teuchos::RCP<EpetraExt::ModelEvaluator> solver = 
	slvrfctry.create(app_comm, app_comm);

      // Setup in/out args
      EpetraExt::ModelEvaluator::InArgs params_in = solver->createInArgs();
      EpetraExt::ModelEvaluator::OutArgs responses_out = 
	solver->createOutArgs();
      int np = params_in.Np();
      for (int i=0; i<np; i++) {
	Teuchos::RCP<const Epetra_Vector> p = solver->get_p_init(i);
	params_in.set_p(i, p);
      }
      int ng = responses_out.Ng();
      for (int i=0; i<ng; i++) {
	Teuchos::RCP<Epetra_Vector> g = 
	  Teuchos::rcp(new Epetra_Vector(*(solver->get_g_map(i))));
	responses_out.set_g(i, g);
      }

      // Evaluate model
      solver->evalModel(params_in, responses_out);

      // Print responses (not last one since that is x)
      *out << std::endl;
      out->precision(8);
      for (int i=0; i<ng-1; i++) {
	if (responses_out.get_g(i) != Teuchos::null)
	  *out << "Response " << i << " = " << std::endl 
	       << *(responses_out.get_g(i)) << std::endl;
      }

      Teuchos::TimeMonitor::summarize(std::cout,false,true,false);
    }

    // Create SG solver
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model = 
      sg_slvrfctry.createAlbanyAppAndModel(app, app_comm, ig);
    Teuchos::RCP<NOX::Epetra::Observer > NOX_observer = 
      Teuchos::rcp(new Albany_NOXObserver(app));
    sg_solver->setup(model, NOX_observer);

    // Evaluate SG responses at SG parameters
    EpetraExt::ModelEvaluator::InArgs sg_inArgs = sg_solver->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs sg_outArgs = 
      sg_solver->createOutArgs();
    int np = sg_inArgs.Np();
    for (int i=0; i<np; i++) {
      if (sg_inArgs.supports(EpetraExt::ModelEvaluator::IN_ARG_p_sg, i)) {
	Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> p_sg = 
	  sg_solver->get_p_sg_init(i);
	sg_inArgs.set_p_sg(i, p_sg);
      }
    }

    bool computeSensitivities = 
      albanyParams.sublist("Problem").get("Compute Sensitivities", true);
    int ng = sg_outArgs.Ng();
    for (int i=0; i<ng; i++) {
      if (sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_g_sg, i)) {
	Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg =
	  sg_solver->create_g_sg(i);
	sg_outArgs.set_g_sg(i, g_sg);
      }

      for (int j=0; j<np; j++) {
	EpetraExt::ModelEvaluator::DerivativeSupport ds =
	  sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp_sg,i,j);
	if (computeSensitivities && 
	    ds.supports(EpetraExt::ModelEvaluator::DERIV_MV_BY_COL)) {
	  int ncol = sg_solver->get_p_map(j)->NumMyElements();
	  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdp_sg =
	    sg_solver->create_g_mv_sg(i,ncol);
	  sg_outArgs.set_DgDp_sg(i, j, dgdp_sg);
	}
      }
    }

    sg_solver->evalModel(sg_inArgs, sg_outArgs);

    totalTimer.~TimeMonitor();
    Teuchos::TimeMonitor::summarize(std::cout,false,true,false);
    Teuchos::TimeMonitor::zeroOutTimers();

    for (int i=0; i<ng-1; i++) {
      // Don't loop over last g which is x, since it is a long vector
      // to print out.
      if (sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_g_sg, i)) {

	// Print mean and standard deviation      
	Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg = 
	  sg_outArgs.get_g_sg(i);
	if (g_sg != Teuchos::null) {
	  Epetra_Vector g_mean(*(sg_solver->get_g_map(i)));
	  Epetra_Vector g_std_dev(*(sg_solver->get_g_map(i)));
	  g_sg->computeMean(g_mean);
	  g_sg->computeStandardDeviation(g_std_dev);
	  out->precision(12);
	  *out << "Response " << i << " Mean =      " << std::endl 
	       << g_mean << std::endl;
	  *out << "Response " << i << " Std. Dev. = " << std::endl 
	       << g_std_dev << std::endl;
	  *out << "Response " << i << "           = " << std::endl 
	       << *g_sg << std::endl;
	  for (int j=0; j<np; j++) {
	    EpetraExt::ModelEvaluator::DerivativeSupport ds =
	      sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp_sg,i,j);
	    if (!ds.none()) {
	      Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdp_sg =
		sg_outArgs.get_DgDp_sg(i,j).getMultiVector();
	      if (dgdp_sg != Teuchos::null)
		*out << "Response " << i << " Derivative " << j << " = " 
		     << std::endl << *dgdp_sg << std::endl;
	    }
	  }

	  status += sg_slvrfctry.checkTestResults(NULL, NULL, NULL, 
						  Teuchos::null, g_sg,
						  &g_mean, &g_std_dev);
	}
      }
    }
    *out << "\nNumber of Failed Comparisons: " << status << endl;

    //////////////////////////////////////////////////////////////////////
    // begin jrr

    // pull out params for Solution KL
    Teuchos::ParameterList& sgParams =
       piroParams->sublist("Stochastic Galerkin");
    Teuchos::ParameterList& solKLParams =
      sgParams.sublist("Response KL");
    bool computeKLOnResponse = solKLParams.get("ComputeKLOnResponse", false);
    int NumKL = solKLParams.get("NumKL", 0);
    
    // Tim, this is a new parameter in the Response KL section:
    //bool CullResponse = solKLParams.get("CullResponse", true);

    // Let's set some temporary overrides to parameters that we just
    // got since they're not in the xml file for the NS solution
    computeKLOnResponse = true;
    NumKL = 2; // max 4 kl terms in input xml file(?)
    bool CullResponse = true;

    // Finish setup for, then call KL solver if asked...
    if( computeKLOnResponse )
      {
	Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_u = 
	  sg_outArgs.get_g_sg(ng-1);
	
	// Boolean return value for KL solution function call 
	bool KL_Success;	    
	    
	// placeholders for our KL solution (they get filled in in the
	// function)
	Teuchos::Array<double> evals;
	Teuchos::RCP<Epetra_MultiVector> evecs;

	if( CullResponse )
	  {
	    // Assume we want to cull the solution expansion...
	    
	    Teuchos::RCP<const Epetra_BlockMap> sg_u_coeff_map = sg_u->coefficientMap();
	    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_u_new; // placeholder for culled coeffs
	    Teuchos::RCP<Epetra_BlockMap> sg_u_new_coeff_map; // placeholder for culled coeff map
	    

	    // Locate the vector components we want to keep manually.
	    // For quad2d problem, it's 2 and we want to
	    // cull the second variable on each node in
	    
	    int neq = 4;                 // NSRayleighBernard2D
	    vector<int> keepDOF(neq, 1); // Initialize to keep all dof, then,
	    keepDOF[2] = 0;              // as stated, cull the 2nd
	    keepDOF[3] = 0;	         // and 3rd dof (temp and
					 // pressure)

	    int ndim = accumulate(keepDOF.begin(), keepDOF.end(), 0);
	    
	    // Now, create the new, culled coefficient map...
	    cullDistributedResponseMap( sg_u_coeff_map, keepDOF, sg_u_new_coeff_map );
	    
	    // ...and import the correct data from sg_u into sg_u_new
	    cullDistributedResponse( sg_u, sg_u_new, sg_u_new_coeff_map );
	    
	    cout << "First 5*neq values of sg_u and sg_u_new:" << endl;
	    {
	      for ( int i=0; i < 5*neq; i++ )
		cout << "sg_u->getBlockVector()[" << i << "] = " <<
		  (*(sg_u->getBlockVector()))[i] << "    " 
		     <<"sg_u_new->getBlockVector()[" << i << "] = " <<
		  (*(sg_u_new->getBlockVector()))[i] << endl;
	    }
	    
	    cout << "Last 5*neq values of sg_u and sg_u_new:" << endl;
	    {
	      int N = sg_u->productMap()->NumMyElements();
	      cout << "Number of elements in sg_u BlockVector: " << N << std::endl;
	      cout << "Number of elements in sg_u_new BlockVector: " << 
		sg_u_new->productMap()->NumMyElements() << std::endl;

	      for ( int i=0; i < 5*neq; i++ )
		cout << "sg_u->getBlockVector()[" << N-i-1 << "] = " <<
		  (*(sg_u->getBlockVector()))[N - (i+1)] << "    " 
		     <<"sg_u_new->getBlockVector()[" << ndim*N/neq - (i+1) << "] = " <<
		  (*(sg_u_new->getBlockVector()))[ndim*N/neq - (i+1)] << endl;
	    }
	    
	    KL_Success = KL_OnSolutionMultiVector(sg_solver, 
						  sg_u_new,
						  NumKL,
						  evals,
						  evecs);
	  }
	else // then run KL on full response, sg_u
	  {
	    KL_Success = KL_OnSolutionMultiVector(sg_solver, 
						  sg_u,
						  NumKL,
						  evals,
						  evecs);
	  }

	if (!KL_Success) 
	  *out << "KL Eigensolver did not converge!" << std::endl;
	
	*out << "Eigenvalues = " << std::endl;
	for (int i=0; i< NumKL; i++)
	  *out << evals[i] << std::endl;
      }
    // for now, we'll look at the numbers in a debugger... :)
    // end jrr
    //////////////////////////////////////////////////////////////////////
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;
  
  return status;
}
