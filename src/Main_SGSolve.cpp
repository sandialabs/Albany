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

#include "Albany_SolverFactory.hpp"
#include "ENAT_SGNOXSolver.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented

  // Initialize MPI and timer
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

#ifdef ALBANY_MPI
  double total_time = -MPI_Wtime();
#endif
  
  // Create a communicator for Epetra objects
  Teuchos::RCP<Epetra_Comm> globalComm;
#ifdef ALBANY_MPI
  globalComm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
#else
  globalComm = Teuchos::rcp(new Epetra_SerialComm);
#endif

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
    Albany::SolverFactory sg_slvrfctry(sg_xmlfilename, *globalComm);
    Teuchos::ParameterList& appParams = sg_slvrfctry.getParameters();
    Teuchos::ParameterList& problemParams = appParams.sublist("Problem");
    Teuchos::ParameterList& sgParams =
      problemParams.sublist("Stochastic Galerkin");
    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > basis = 
      Stokhos::BasisFactory<int,double>::create(sgParams);

    // Create multi-level comm and spatial comm
    int num_stoch_blocks = basis->size();
    int num_spatial_procs = 
      problemParams.get("Number of Spatial Processors", -1);
    Teuchos::RCP<const EpetraExt::MultiComm> sg_comm =
      Stokhos::buildMultiComm(*globalComm, num_stoch_blocks, num_spatial_procs);
    Teuchos::RCP<const Epetra_Comm> app_comm = Stokhos::getSpatialComm(sg_comm);

    // Compute initial guess if requested
    Teuchos::RCP<Epetra_Vector> g2;
    if (do_initial_guess) {

      Albany::SolverFactory slvrfctry(xmlfilename, *sg_comm);
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
      Teuchos::TimeMonitor::zeroOutTimers();
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
      Teuchos::rcp(new Stokhos::EpetraVectorOrthogPoly(
		     App_sg->getBasis(), *(App_sg->get_g_sg_map(0))));

    EpetraExt::ModelEvaluator::InArgs params_in_sg = App_sg->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs responses_out_sg = 
      App_sg->createOutArgs();

    // Evaluate sg model
    params_in_sg.set_p_sg(sg_p_index,p_sg);
    responses_out_sg.set_g_sg(0,g_sg);
    App_sg->evalModel(params_in_sg, responses_out_sg);

    *out << "Finished eval of sg model: Params, Responses " 
         << std::setprecision(12) << endl;
    p_sg->print(*out << "\nParameters!\n");
    g_sg->print(*out << "\nResponses!\n");

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
    *out << setprecision(16) << std_dev << std::endl;
    
  }

  catch (std::exception& e) {
    cout << e.what() << endl;
    status = 10;
  }
  catch (string& s) {
    cout << s << endl;
    status = 20;
  }
  catch (char *s) {
    cout << s << endl;
    status = 30;
  }
  catch (...) {
    cout << "Caught unknown exception!" << endl;
    status = 40;
  }

#ifdef ALBANY_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  total_time +=  MPI_Wtime();
  *out << "\n\nTOTAL TIME     " << total_time << "  " << total_time << endl;
#else
  *out << "\tTOTAL TIME =     -999.0  -999.0" << endl;
#endif

  return status;
}
