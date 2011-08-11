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
#include "ENAT_SGNOXSolver.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Stokhos_PCEAnasaziKL.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////
// begin jrr
//
// Global function to encapsulate KL solution computation...
//

bool KL_OnSolutionMultiVector( const Teuchos::RCP<ENAT::SGNOXSolver>& App_sg, 
					const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& sg_u,
					const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
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
    App_sg->get_sg_model()->import_solution(*(sg_u->getBlockVector()));
  Teuchos::RCP<const EpetraExt::BlockVector> cX_ov = X_ov;

  // pceKL is object with member functions that explicitly call anasazi
  Stokhos::PCEAnasaziKL pceKL(cX_ov, *basis, NumKL);

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
    
    // First instantiate the stochastic basis 
    // (we need this to get stochastic parallelism right)
    Albany::SolverFactory sg_slvrfctry(sg_xmlfilename, Albany_MPI_COMM_WORLD);
    Teuchos::ParameterList& appParams = sg_slvrfctry.getParameters();
    Teuchos::ParameterList& problemParams = appParams.sublist("Problem");

    Teuchos::ParameterList& sgParams =
      problemParams.sublist("Stochastic Galerkin");
    Teuchos::ParameterList& sg_parameterParams = 
      sgParams.sublist("SG Parameters");
    Teuchos::ParameterList& sg_basisParams = sgParams.sublist("Basis");
    int numParameters = sg_parameterParams.get("Number", 0);
    if (!sg_basisParams.isParameter("Dimension"))
      sg_basisParams.set("Dimension", numParameters);
    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > basis = 
      Stokhos::BasisFactory<int,double>::create(sgParams);

    //////////////////////////////////////////////////////////////////////
    // begin jrr
    // pull out params for Solution KL
    Teuchos::ParameterList& solKLParams =
      sgParams.sublist("Response KL");
    bool computeKLOnResponse = solKLParams.get("ComputeKLOnResponse", false);
    int NumKL = solKLParams.get("NumKL", 0);
    // end jrr
    //////////////////////////////////////////////////////////////////////



    // Create multi-level comm and spatial comm
    int num_stoch_blocks;
    std::string sg_type = sgParams.get("SG Method", "AD");
    if (sg_type == "Multi-point Non-intrusive") {
      Teuchos::RCP<const Stokhos::Quadrature<int,double> > quad =
	Stokhos::QuadratureFactory<int,double>::create(sgParams);
      num_stoch_blocks = quad->size();
    }
    else
      num_stoch_blocks = basis->size();
    int num_spatial_procs = 
      problemParams.get("Number of Spatial Processors", -1);
    Teuchos::RCP<Epetra_Comm> globalComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    Teuchos::RCP<const EpetraExt::MultiComm> sg_comm =
      Stokhos::buildMultiComm(*globalComm, num_stoch_blocks, num_spatial_procs);
    Teuchos::RCP<const Epetra_Comm> app_comm = Stokhos::getSpatialComm(sg_comm);

    // Compute initial guess if requested
    Teuchos::RCP<Epetra_Vector> g2;
    if (do_initial_guess) {

      Albany::SolverFactory slvrfctry(xmlfilename,
                                      Albany::getMpiCommFromEpetraComm(*sg_comm));
      Teuchos::RCP<EpetraExt::ModelEvaluator> App = 
	slvrfctry.create(app_comm, app_comm);

      Teuchos::RCP<Epetra_Vector> p = 
	Teuchos::rcp(new Epetra_Vector(*(App->get_p_init(0))));
      Teuchos::RCP<Epetra_Vector> g1 = 
	Teuchos::rcp(new Epetra_Vector(*(App->get_g_map(0))));
      g2 = Teuchos::rcp(new Epetra_Vector(*(App->get_g_map(1))));
      
      EpetraExt::ModelEvaluator::InArgs params_in = App->createInArgs();
      EpetraExt::ModelEvaluator::OutArgs responses_out = App->createOutArgs();

      // Evaluate first model
      params_in.set_p(0,p);
      responses_out.set_g(0,g1);
      responses_out.set_g(1,g2);
      App->evalModel(params_in, responses_out);

      *out << "Finished eval of first model: Params, Responses " 
	   << std::setprecision(12) << endl;
      p->Print(*out << "\nParameters!\n");
      g1->Print(*out << "\nResponses!\n");

      Teuchos::TimeMonitor::summarize(std::cout,false,true,false);
    }

    Teuchos::RCP<ENAT::SGNOXSolver> App_sg = 
      Teuchos::rcp_dynamic_cast<ENAT::SGNOXSolver>(sg_slvrfctry.create(app_comm, sg_comm, g2));
//     Teuchos::ParameterList& params = sg_slvrfctry.getParameters();
    int sg_p_index = 1;
    if (sg_type == "AD" || sg_type == "Multi-point Non-intrusive")
      sg_p_index = 0;

    Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> p_sg = 
      App_sg->get_p_sg_init(sg_p_index);
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg =
      App_sg->get_sg_model()->create_g_sg(0);

    // begin jrr
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_u = 
      App_sg->get_sg_model()->create_x_sg();
    // end jrr

    EpetraExt::ModelEvaluator::InArgs params_in_sg = App_sg->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out_sg = 
      App_sg->createOutArgs();

    // Evaluate sg model
    params_in_sg.set_p_sg(sg_p_index,p_sg);
    responses_out_sg.set_g_sg(0,g_sg);

    // begin jrr
    responses_out_sg.set_g_sg(1, sg_u);
    // end jrr

    App_sg->evalModel(params_in_sg, responses_out_sg);

     // *out << "Finished eval of sg model: Params, Responses " 
     //      << std::setprecision(12) << endl;
     // p_sg->print(*out << "\nParameters!\n");
     // g_sg->print(*out << "\nResponses!\n");

    totalTimer.~TimeMonitor();
    Teuchos::TimeMonitor::summarize(std::cout,false,true,false);
    Teuchos::TimeMonitor::zeroOutTimers();

    status += sg_slvrfctry.checkTestResults(NULL, NULL, NULL, Teuchos::null,
					    g_sg);
    *out << "\nNumber of Failed Comparisons: " << status << endl;

    Epetra_Vector mean(*(App_sg->get_g_sg_map(0)));
    Epetra_Vector std_dev(*(App_sg->get_g_sg_map(0)));
    g_sg->computeMean(mean);
    g_sg->computeStandardDeviation(std_dev);

    // Print out mean & standard deviation
    *out << "Mean = " << std::endl;
    *out << setprecision(16) << mean << std::endl;
    *out << "Standard Deviation = " << std::endl;
    *out << setprecision(16) << std_dev << std::endl;

    // begin jrr
    // Finish setup for, then call KL solver if asked...
    if( computeKLOnResponse )
      {
	//    int NumKL = 5; // Get this from input xml file parameters
	Teuchos::Array<double> evals;
	Teuchos::RCP<Epetra_MultiVector> evecs;
	
	bool KL_success = KL_OnSolutionMultiVector(App_sg, 
						   sg_u,
						   basis,
						   NumKL,
						   evals,
						   evecs);
	
	if (!KL_success) 
	  *out << "KL Eigensolver did not converge!" << std::endl;
    
	*out << "Eigenvalues = " << std::endl;
	for (int i=0; i< NumKL; i++)
	  *out << evals[i] << std::endl;
      }
    // for now, we'll look at the numbers in a debugger... :)
    // end jrr
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  return status;
}
