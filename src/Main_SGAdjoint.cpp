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

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  char * xmlfilename=0;
  char * sg_xmlfilename=0;
  char * adj_xmlfilename=0;
  char * adjsg_xmlfilename=0;
  char defaultfile[10]={"input.xml"};
  char sg_defaultfile[12]={"inputSG.xml"};
  char adj_defaultfile[18]={"input_adjoint.xml"};
  char adjsg_defaultfile[20]={"inputSG_adjoint.xml"};
  bool do_initial_guess;
  if(argc>1){
    if(!strcmp(argv[1],"--help")){
      printf("albanySG [inputfile.xml inputfileSG.xml inputfileadjoint.xml inputfileSGadjoint.xml]\n");
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
	adj_xmlfilename=argv[3];
	adjsg_xmlfilename = argv[4];
	do_initial_guess = true;
      }
    }
  }
  else {
    xmlfilename=defaultfile;
    sg_xmlfilename=sg_defaultfile;
    adj_xmlfilename=adj_defaultfile;
    adjsg_xmlfilename=adjsg_defaultfile;
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

    // Create multi-level comm and spatial comm
    int num_stoch_blocks = basis->size();
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
    Teuchos::ParameterList& params = sg_slvrfctry.getParameters();
    std::string sg_type = sgParams.get("SG Method", "AD");
    int sg_p_index = 1;
    if (sg_type == "AD")
      sg_p_index = 0;

    Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> p_sg = 
      App_sg->get_p_sg_init(sg_p_index);
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg =
      App_sg->get_sg_model()->create_g_sg(0);
   

    // Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg =
    //  Teuchos::rcp(new Stokhos::EpetraVectorOrthogPoly(
    //		     App_sg->getBasis(), *(App_sg->get_g_sg_map(0))));

    EpetraExt::ModelEvaluator::InArgs params_in_sg = App_sg->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out_sg = 
      App_sg->createOutArgs();

    // Evaluate sg model
    params_in_sg.set_p_sg(sg_p_index,p_sg);
    responses_out_sg.set_g_sg(0,g_sg);
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
  


    /* Space reserved for the projection of the forward solution onto
       the higher order basis for the adjoint solution.  
       In general, this will require projecting is physical space
       as well as in stochastic space.
    */



    // Start solving the adjoint problem
    Teuchos::RCP<Teuchos::Time> adj_totalTime =
      Teuchos::TimeMonitor::getNewTimer("AlbanySG: ***Total Time***");
    Teuchos::TimeMonitor adjtotalTimer(*adj_totalTime); //start timer

    // First instantiate the stochastic basis 
    // (we need this to get stochastic parallelism right)
    Albany::SolverFactory adjsg_slvrfctry(adjsg_xmlfilename, Albany_MPI_COMM_WORLD);
    Teuchos::ParameterList& adjappParams = adjsg_slvrfctry.getParameters();
    Teuchos::ParameterList& adjproblemParams = adjappParams.sublist("Problem");
    Teuchos::ParameterList& adjsgParams =
      adjproblemParams.sublist("Stochastic Galerkin");
    Teuchos::ParameterList& adjsg_parameterParams = 
      adjsgParams.sublist("SG Parameters");
    Teuchos::ParameterList& adjsg_basisParams = adjsgParams.sublist("Basis");
    int adjnumParameters = adjsg_parameterParams.get("Number", 0);
    if (!adjsg_basisParams.isParameter("Dimension"))
      adjsg_basisParams.set("Dimension", adjnumParameters);
    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > adjbasis = 
      Stokhos::BasisFactory<int,double>::create(adjsgParams);

    // Create multi-level comm and spatial comm
    int adjnum_stoch_blocks = adjbasis->size();
    int adjnum_spatial_procs = 
      adjproblemParams.get("Number of Spatial Processors", -1);
    Teuchos::RCP<Epetra_Comm> adjglobalComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    Teuchos::RCP<const EpetraExt::MultiComm> adjsg_comm =
      Stokhos::buildMultiComm(*adjglobalComm, adjnum_stoch_blocks, adjnum_spatial_procs);
    Teuchos::RCP<const Epetra_Comm> adjapp_comm = Stokhos::getSpatialComm(adjsg_comm);

    // Compute initial guess if requested
    Teuchos::RCP<Epetra_Vector> adjg2;
    if (do_initial_guess) {

      Albany::SolverFactory slvrfctry(adj_xmlfilename,
                                      Albany::getMpiCommFromEpetraComm(*adjsg_comm));
      Teuchos::RCP<EpetraExt::ModelEvaluator> adjApp = 
	slvrfctry.create(adjapp_comm, adjapp_comm);

      Teuchos::RCP<Epetra_Vector> adjp = 
	Teuchos::rcp(new Epetra_Vector(*(adjApp->get_p_init(0))));
      Teuchos::RCP<Epetra_Vector> adjg1 = 
	Teuchos::rcp(new Epetra_Vector(*(adjApp->get_g_map(0))));
      adjg2 = Teuchos::rcp(new Epetra_Vector(*(adjApp->get_g_map(1))));
      
      EpetraExt::ModelEvaluator::InArgs adjparams_in = adjApp->createInArgs();
      EpetraExt::ModelEvaluator::OutArgs adjresponses_out = adjApp->createOutArgs();

      // Evaluate first model
      adjparams_in.set_p(0,adjp);
      adjresponses_out.set_g(0,adjg1);
      adjresponses_out.set_g(1,adjg2);
      adjApp->evalModel(adjparams_in, adjresponses_out);

      *out << "Finished eval of first model: Params, Responses " 
	   << std::setprecision(12) << endl;
      adjp->Print(*out << "\nParameters!\n");
      adjg1->Print(*out << "\nResponses!\n");

      Teuchos::TimeMonitor::summarize(std::cout,false,true,false);
    }

    Teuchos::RCP<ENAT::SGNOXSolver> adjApp_sg = 
      Teuchos::rcp_dynamic_cast<ENAT::SGNOXSolver>(adjsg_slvrfctry.create(adjapp_comm, adjsg_comm, adjg2));
    Teuchos::ParameterList& adjparams = adjsg_slvrfctry.getParameters();
    std::string adjsg_type = adjsgParams.get("SG Method", "AD");
    int adjsg_p_index = 1;
    if (adjsg_type == "AD")
      adjsg_p_index = 0;

    Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> adjp_sg = 
      adjApp_sg->get_p_sg_init(adjsg_p_index);
    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> adjg_sg =
      adjApp_sg->get_sg_model()->create_g_sg(0);
  

    //Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> adjg_sg =
    //  Teuchos::rcp(new Stokhos::EpetraVectorOrthogPoly(
    //		     adjApp_sg->getBasis(), *(adjApp_sg->get_g_sg_map(0))));

    EpetraExt::ModelEvaluator::InArgs adjparams_in_sg = adjApp_sg->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs adjresponses_out_sg = 
      adjApp_sg->createOutArgs();

    // Evaluate sg model
    params_in_sg.set_p_sg(adjsg_p_index,adjp_sg);
    responses_out_sg.set_g_sg(0,adjg_sg);
    adjApp_sg->evalModel(adjparams_in_sg, adjresponses_out_sg);

    // *out << "Finished eval of sg model: Params, Responses " 
    //      << std::setprecision(12) << endl;
    // p_sg->print(*out << "\nParameters!\n");
    // g_sg->print(*out << "\nResponses!\n");

    adjtotalTimer.~TimeMonitor();
    Teuchos::TimeMonitor::summarize(std::cout,false,true,false);
    Teuchos::TimeMonitor::zeroOutTimers();

    status += adjsg_slvrfctry.checkTestResults(NULL, NULL, NULL, Teuchos::null,
					    adjg_sg);
    *out << "\nNumber of Failed Comparisons: " << status << endl;

    Epetra_Vector adjmean(*(adjApp_sg->get_g_sg_map(0)));
    Epetra_Vector adjstd_dev(*(adjApp_sg->get_g_sg_map(0)));
    adjg_sg->computeMean(adjmean);
    adjg_sg->computeStandardDeviation(adjstd_dev);

    // Print out mean & standard deviation
    *out << "Mean = " << std::endl;
    *out << setprecision(16) << adjmean << std::endl;
    *out << "Standard Deviation = " << std::endl;
    *out << setprecision(16) << adjstd_dev << std::endl;


    /* Space reserved for computing the error representation which involves
       integrating over both physical and stochastic space and may require
       a number of projections.
    */


  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  return status;
}
