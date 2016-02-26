//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Piro_Epetra_StokhosSolver.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Petra_Converters.hpp"
#include "Albany_Utils.hpp"

/* GAH FIXME - Silence warning:
TRILINOS_DIR/../../../include/pecos_global_defs.hpp:17:0: warning:
        "BOOST_MATH_PROMOTE_DOUBLE_POLICY" redefined [enabled by default]
Please remove when issue is resolved
*/
#undef BOOST_MATH_PROMOTE_DOUBLE_POLICY

#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"

// Global variable that denotes this is the Tpetra executable
bool TpetraBuild = false;

int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd("input.xml", "inputSG.xml", "inputSG_adjoint.xml");
  cmd.parse_cmdline(argc, argv, *out);
  std::string xmlfilename;
  std::string sg_xmlfilename;
  std::string adjsg_xmlfilename;
  bool do_initial_guess;
  if (cmd.has_third_xml_file) {
    xmlfilename = cmd.xml_filename;
    sg_xmlfilename = cmd.xml_filename2;
    adjsg_xmlfilename = cmd.xml_filename3;
    do_initial_guess = true;
  }
  else if (cmd.has_second_xml_file) {
    xmlfilename = "";
    sg_xmlfilename = cmd.xml_filename;
    adjsg_xmlfilename = cmd.xml_filename2;
    do_initial_guess = false;
  }
  else {
    *out << argv[0] << ":  must supply at least 2 input files!\n";
    std::exit(1);
  }

  try {

    Teuchos::RCP<Teuchos::Time> totalTime =
      Teuchos::TimeMonitor::getNewTimer("AlbanySGAdjoint: ***Total Time***");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    
    // Setup communication objects
    Teuchos::RCP<Epetra_Comm> globalComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_forward_solution;

    //
    // Solve forward problem
    //
    {
      
    Teuchos::RCP<Teuchos::Time> forwardTime =
      Teuchos::TimeMonitor::getNewTimer("AlbanySGAdjoint: ***Forward Solver Time***");
    Teuchos::TimeMonitor forwardTimer(*forwardTime); //start timer

    // Parse parameters
    Teuchos::RCP<const Teuchos_Comm> comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
    // Connect vtune for performance profiling
    if (cmd.vtune) {
      Albany::connect_vtune(comm->getRank());
    }
    Albany::SolverFactory sg_slvrfctry(sg_xmlfilename, comm);
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
      Albany::SolverFactory slvrfctry(xmlfilename, 
         Albany::createTeuchosCommFromEpetraComm(app_comm));
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

    }

    // Create SG solver
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model; {
      Teuchos::RCP<const Teuchos_Comm>
        commT = Albany::createTeuchosCommFromEpetraComm(app_comm);
      model = sg_slvrfctry.createAlbanyAppAndModel(
        app, app_comm, Petra::EpetraVector_To_TpetraVectorConst(
          *ig, commT));
    }
    sg_solver->setup(model);

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

    int ng = sg_outArgs.Ng();
    for (int i=0; i<ng; i++) {
      if (sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_g_sg, i)) {
	Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg =
	  sg_solver->create_g_sg(i);
	sg_outArgs.set_g_sg(i, g_sg);
      }
    }

    sg_solver->evalModel(sg_inArgs, sg_outArgs);

    for (int i=0; i<ng-1; i++) {
      // Don't loop over last g which is x, since it is a long vector
      // to print out.
      if (sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_g_sg, i)) {

	// Print mean and standard deviation      
	Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg = 
	  sg_outArgs.get_g_sg(i);
	Epetra_Vector g_mean(*(sg_solver->get_g_map(i)));
	Epetra_Vector g_std_dev(*(sg_solver->get_g_map(i)));
	g_sg->computeMean(g_mean);
	g_sg->computeStandardDeviation(g_std_dev);
	out->precision(12);
	*out << "Response " << i << " Mean =      " << std::endl 
	     << g_mean << std::endl;
	*out << "Response " << i << " Std. Dev. = " << std::endl 
	     << g_std_dev << std::endl;

	status += sg_slvrfctry.checkSGTestResults(0, g_sg);
      }
    }
    *out << "\nNumber of Failed Comparisons: " << status << std::endl;
  
    sg_forward_solution = sg_outArgs.get_g_sg(ng-1);

    }


    /* Space reserved for the projection of the forward solution onto
       the higher order basis for the adjoint solution.  
       In general, this will require projecting is physical space
       as well as in stochastic space.
    */



    // 
    // Solve adjoint problem
    //
    {

    Teuchos::RCP<Teuchos::Time> adjointTime =
      Teuchos::TimeMonitor::getNewTimer("AlbanySG: ***Adjoint Solver Time***");
    Teuchos::TimeMonitor adjtotalTimer(*adjointTime); //start timer

    // Parse parameters
    Albany::SolverFactory sg_slvrfctry(adjsg_xmlfilename, 
      Albany::createTeuchosCommFromMpiComm(Albany_MPI_COMM_WORLD));
    Teuchos::ParameterList& albanyParams = sg_slvrfctry.getParameters();
    Teuchos::RCP< Teuchos::ParameterList> piroParams = 
      Teuchos::rcp(&(albanyParams.sublist("Piro")),false);
    
    // Create stochastic Galerkin solver
    Teuchos::RCP<Piro::Epetra::StokhosSolver> sg_solver =
      Teuchos::rcp(new Piro::Epetra::StokhosSolver(piroParams, globalComm));

    // Get comm for spatial problem
    Teuchos::RCP<const Epetra_Comm> app_comm = sg_solver->getSpatialComm();

    // Create SG solver
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model = 
      sg_slvrfctry.createAlbanyAppAndModel(app, app_comm);
    sg_solver->setup(model);

    // Set projected forward solution as the initial guess
    sg_solver->set_x_sg_init(*sg_forward_solution);

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

    int ng = sg_outArgs.Ng();
    for (int i=0; i<ng; i++) {
      if (sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_g_sg, i)) {
	Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg =
	  sg_solver->create_g_sg(i);
	sg_outArgs.set_g_sg(i, g_sg);
      }
    }

    sg_solver->evalModel(sg_inArgs, sg_outArgs);

    for (int i=0; i<ng-1; i++) {
      // Don't loop over last g which is x, since it is a long vector
      // to print out.
      if (sg_outArgs.supports(EpetraExt::ModelEvaluator::OUT_ARG_g_sg, i)) {

	// Print mean and standard deviation      
	Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg = 
	  sg_outArgs.get_g_sg(i);
	Epetra_Vector g_mean(*(sg_solver->get_g_map(i)));
	Epetra_Vector g_std_dev(*(sg_solver->get_g_map(i)));
	g_sg->computeMean(g_mean);
	g_sg->computeStandardDeviation(g_std_dev);
	out->precision(12);
	*out << "Response " << i << " Mean =      " << std::endl 
	     << g_mean << std::endl;
	*out << "Response " << i << " Std. Dev. = " << std::endl 
	     << g_std_dev << std::endl;

	status += sg_slvrfctry.checkSGTestResults(0, g_sg);
      }
    }
    *out << "\nNumber of Failed Comparisons: " << status << std::endl;

    /* Space reserved for computing the error representation which involves
       integrating over both physical and stochastic space and may require
       a number of projections.
    */

    }

    totalTimer.~TimeMonitor();
    Teuchos::TimeMonitor::summarize(std::cout,false,true,false);

  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  return status;
}
