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
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Epetra_Map.h"  //Needed for serial, somehow
#include "Piro_Epetra_NECoupledModelEvaluator.hpp"
#include "Piro_Epetra_Factory.hpp"
#include "Epetra_LocalMap.h"
#include "Epetra_Import.h"

class ReactorNetworkModel : public Piro::Epetra::AbstractNetworkModel {
  
public:
  
  //! Constructor
  ReactorNetworkModel(int n_) : n(n_) {}
  
  //! Destructor
  virtual ~ReactorNetworkModel() {}
  
  //! evaluate model
  virtual void evalModel(
    const Teuchos::Array<EpetraExt::ModelEvaluator::InArgs>& model_inargs, 
    const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs>& model_outargs,
    const EpetraExt::ModelEvaluator::InArgs& network_inargs, 
    const EpetraExt::ModelEvaluator::OutArgs& network_outargs,
    const Teuchos::Array<int>& n_p,
    const Teuchos::Array<int>& n_g,
    const Teuchos::Array< Teuchos::RCP<Epetra_Vector> >& p,
    const Teuchos::Array< Teuchos::RCP<Epetra_Vector> >& g,
    const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dgdp,
    const Teuchos::Array<EpetraExt::ModelEvaluator::EDerivativeMultiVectorOrientation>& dgdp_layout,
    const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs::sg_vector_t>& p_sg,
    const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs::sg_vector_t>& g_sg,
    const Teuchos::Array<Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> >& dgdp_sg,
    const Teuchos::Array<EpetraExt::ModelEvaluator::EDerivativeMultiVectorOrientation>& dgdp_sg_layout) const
    {
     
      // f
      Teuchos::RCP<Epetra_Vector> f = network_outargs.get_f();
      if (f != Teuchos::null) {
	g[0]->Print(std::cout << "g[0] = " << std::endl);
	g[1]->Print(std::cout << "g[1] = " << std::endl);
	g[2]->Print(std::cout << "g[2] = " << std::endl);
	f->PutScalar(0.0);
	for (int i=0; i<n; i++) {
	  (*f)[i]     = (*g[0])[i] - (*g[1])[i];
	  (*f)[i+n]   = (*g[2])[i] - (*g[1])[i+n];
	  (*f)[i+2*n] = (*p[0])[i] + (*p[1])[i];
	  (*f)[i+3*n] = (*p[2])[i] + (*p[1])[i+n];
	}
      }

      // W
      Teuchos::RCP<Epetra_Operator> W = network_outargs.get_W();
      if (W != Teuchos::null) {
	dgdp[0]->Print(std::cout << "dgdp[0] = " << std::endl);
	dgdp[1]->Print(std::cout << "dgdp[1] = " << std::endl);
	dgdp[2]->Print(std::cout << "dgdp[2] = " << std::endl);
	Teuchos::RCP<Epetra_CrsMatrix> W_crs = 
	  Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W, true);
	W_crs->PutScalar(0.0);
	int row, col;
	double val;

	// Block row 1
	for (int i=0; i<n; i++) {
	  row = i; 
	  
	  // (1,1) block
	  for (int j=0; j<n; j++) {
	    col = j; 
	    val = (*dgdp[0])[j][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	  // (1,2) block
	  for (int j=0; j<n; j++) {
	    col = n+j; 
	    val = -(*dgdp[1])[j][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	  // (1,3) block
	  for (int j=0; j<n; j++) {
	    col = 2*n+j; 
	    val = -(*dgdp[1])[j+n][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	  // (1,4) block
	  for (int j=0; j<n; j++) {
	    col = 3*n+j; 
	    val = 0.0;
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	}

	// Block row 2
	for (int i=0; i<n; i++) {
	  row = i+n; 
	  
	  // (2,1) block
	  for (int j=0; j<n; j++) {
	    col = j; 
	    val = 0.0;
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	  // (2,2) block
	  for (int j=0; j<n; j++) {
	    col = n+j; 
	    val = -(*dgdp[1])[j][i+n];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	  // (2,3) block
	  for (int j=0; j<n; j++) {
	    col = 2*n+j; 
	    val = -(*dgdp[1])[j+n][i+n];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	  // (2,4) block
	  for (int j=0; j<n; j++) {
	    col = 3*n+j; 
	    val = (*dgdp[2])[j][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	}

	// Block row 3
	for (int i=0; i<n; i++) {
	  row = 2*n+i; 
	  
	  // (3,1) block
	  col = 0+i; 
	  val = 1.0;
	  W_crs->ReplaceGlobalValues(row, 1, &val, &col);

	  // (3,2) block
	  col = n+i; 
	  val = 1.0;
	  W_crs->ReplaceGlobalValues(row, 1, &val, &col);

	  // (3,3) block -- zero

	  // (3,4) block -- zero
	}

	// Block row 4
	for (int i=0; i<n; i++) {
	  row = 3*n+i; 
	  
	  // (4,1) block -- zero

	  // (4,2) block -- zero

	  // (4,3) block
	  col = 2*n+i; 
	  val = 1.0;
	  W_crs->ReplaceGlobalValues(row, 1, &val, &col);

	  // (4,4) block
	  col = 3*n+i; 
	  val = 1.0;
	  W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	}

	//W_crs->Print(std::cout << "W_crs =" << std::endl);
      }
      
      /*
      // f_sg
      if (network_outargs.supports(EpetraExt::ModelEvaluator::OUT_ARG_f_sg)) {
	EpetraExt::ModelEvaluator::OutArgs::sg_vector_t f_sg = 
	  network_outargs.get_f_sg();
	if (f_sg != Teuchos::null) {
	  f_sg->init(0.0);
	  for (int block=0; block<f_sg->size(); block++) {
	    for (int i=0; i<n_p[0]; i++)
	  (*f_sg)[block][i] = 
	    (*p_sg[0])[block][i] - (*g_sg[1])[block][i];
	    for (int i=0; i<n_p[1]; i++)
	      (*f_sg)[block][i+n_p[0]] = 
		(*p_sg[1])[block][i] - (*g_sg[0])[block][i];
	  }
	}
      }
      
      // W_sg
      if (network_outargs.supports(EpetraExt::ModelEvaluator::OUT_ARG_W_sg)) {
	EpetraExt::ModelEvaluator::OutArgs::sg_operator_t W_sg = 
	  network_outargs.get_W_sg();
	if (W_sg != Teuchos::null) {
	  for (int block=0; block<W_sg->size(); block++) {
	    Teuchos::RCP<Epetra_CrsMatrix> W_crs = 
	      Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_sg->getCoeffPtr(block), 
							  true);
	    W_crs->PutScalar(0.0);
	    int row, col;
	    double val;
	    for (int i=0; i<n_p[0]; i++) {
	      row = i; 
	      
	      // Diagonal part
	      if (block == 0) {
		col = row; 
		val = 1.0;
		W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	      }
	      
	      // dg_2/dp_2 part
	      for (int j=0; j<n_p[1]; j++) {
		col = n_p[0]+j; 
		if (dgdp_layout[1] == EpetraExt::ModelEvaluator::DERIV_MV_BY_COL)
		  val = -(*dgdp_sg[1])[block][j][i];
		else
		  val = -(*dgdp_sg[1])[block][i][j];
		W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	      }
	    }
	    for (int i=0; i<n_p[1]; i++) {
	      row = n_p[0] + i; 
	      
	      // Diagonal part
	      if (block == 0) {
		col = row; 
		val = 1.0;
		W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	      }
	      
	      // dg_1/dp_1 part
	      for (int j=0; j<n_p[0]; j++) {
		col = j; 
		if (dgdp_layout[0] == EpetraExt::ModelEvaluator::DERIV_MV_BY_COL)
		  val = -(*dgdp_sg[0])[block][j][i];
		else
		  val = -(*dgdp_sg[0])[block][i][j];
		W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	      }
	    }
	  }
	}
      }
      */
    }

protected:
  int n;
};


int main(int argc, char *argv[]) {

  int status=0; // 0 = pass, failures are incremented
  bool success = true;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  
  //***********************************************************
  // Command-line argument for input file
  //***********************************************************

  std::string xmlfilename_coupled;
  if(argc > 1){
    if(!strcmp(argv[1],"--help")){
      std::cout << "albany [inputfile.xml]" << std::endl;
      std::exit(1);
    }
    else
      xmlfilename_coupled = argv[1];
  }
  else
    xmlfilename_coupled = "input.xml";

  try {

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    //***********************************************************
    // Set up coupled model
    //***********************************************************

    Albany::SolverFactory coupled_slvrfctry(xmlfilename_coupled, 
					    Albany_MPI_COMM_WORLD);
    RCP<Epetra_Comm> coupledComm = 
      Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
    Teuchos::ParameterList& coupledParams = coupled_slvrfctry.getParameters();
    Teuchos::ParameterList& coupledSystemParams = 
      coupledParams.sublist("Coupled System");
    Teuchos::RCP< Teuchos::ParameterList> coupledPiroParams = 
      Teuchos::rcp(&(coupledParams.sublist("Piro")),false);
    Teuchos::Array<std::string> model_filenames =
      coupledSystemParams.get<Teuchos::Array<std::string> >("Model XML Files");
    int num_models = model_filenames.size();
    Teuchos::Array< RCP<Albany::Application> > apps(num_models);
    Teuchos::Array< RCP<EpetraExt::ModelEvaluator> > models(num_models);
    Teuchos::Array< RCP<Teuchos::ParameterList> > piroParams(num_models);

    // Set up each model
    for (int m=0; m<num_models; m++) {
      Albany::SolverFactory slvrfctry(model_filenames[m], 
				      Albany_MPI_COMM_WORLD);
      RCP<Epetra_Comm> appComm = 
	Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
      models[m] = slvrfctry.createAlbanyAppAndModel(apps[m], appComm);
      Teuchos::ParameterList& appParams = slvrfctry.getParameters();
      piroParams[m] = Teuchos::rcp(&(appParams.sublist("Piro")),false);
    }
    
    // Setup network model
    std::string network_name = 
      coupledSystemParams.get("Network Model", "Param To Response");
    RCP<Piro::Epetra::AbstractNetworkModel> network_model;
    if (network_name == "Param To Response")
      network_model = rcp(new Piro::Epetra::ParamToResponseNetworkModel);
    else if (network_name == "Reactor Network")
      network_model = rcp(new ReactorNetworkModel(1));
    else
      TEUCHOS_TEST_FOR_EXCEPTION(
	true, std::logic_error, "Invalid network model name " << network_name);
    RCP<EpetraExt::ModelEvaluator> coupledModel =
      rcp(new Piro::Epetra::NECoupledModelEvaluator(models, piroParams,
						    network_model,
						    coupledPiroParams, 
						    coupledComm));
    RCP<EpetraExt::ModelEvaluator> coupledSolver =
      Piro::Epetra::Factory::createSolver(coupledPiroParams, coupledModel);
    
    // Solve coupled system
    EpetraExt::ModelEvaluator::InArgs inArgs = coupledSolver->createInArgs();
    EpetraExt::ModelEvaluator::OutArgs outArgs = coupledSolver->createOutArgs();
    for (int i=0; i<inArgs.Np(); i++)
      inArgs.set_p(i, coupledSolver->get_p_init(i));
    for (int i=0; i<outArgs.Ng(); i++) {
      RCP<Epetra_Vector> g = 
	rcp(new Epetra_Vector(*(coupledSolver->get_g_map(i))));
      outArgs.set_g(i, g);
    }
    coupledSolver->evalModel(inArgs, outArgs);

    // "observe solution" -- need to integrate this with the solvers
    for (int m=0; m<num_models; m++) {
      Albany_NOXObserver observer(apps[m]);
      observer.observeSolution(*(outArgs.get_g(m)));
    }
    
    // Print results
    RCP<Epetra_Vector> x_final = outArgs.get_g(outArgs.Ng()-1);
    RCP<Epetra_Vector> x_final_local = x_final;
    if (x_final->Map().DistributedGlobal()) {
      Epetra_LocalMap local_map(x_final->GlobalLength(), 0, 
				x_final->Map().Comm());
      x_final_local = rcp(new Epetra_Vector(local_map));
      Epetra_Import importer(local_map, x_final->Map());
      x_final_local->Import(*x_final, importer, Insert);
    }
    *out << std::endl
	 << "Final value of coupling variables:" << std::endl
	 << *x_final_local << std::endl;

    status += coupled_slvrfctry.checkTestResults(0, 0, x_final_local.get(), 
						 NULL);
    *out << "\nNumber of Failed Comparisons: " << status << endl;
  }

  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status+=10000;

  Teuchos::TimeMonitor::summarize(*out,false,true,false/*zero timers*/);
  return status;
}
