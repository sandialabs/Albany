//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_GenEigensolver.hpp"

//#include "Stokhos.hpp"
//#include "Stokhos_Epetra.hpp"
//#include "Sacado_PCE_OrthogPoly.hpp"


#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"

//needed?
//#include "Teuchos_RCP.hpp"
//#include "Teuchos_VerboseObject.hpp"
//#include "Teuchos_FancyOStream.hpp"

//#include "Albany_ModelFactory.hpp"
//#include "Albany_Utils.hpp"
//#include "Albany_SolverFactory.hpp"
//#include "Albany_StateInfoStruct.hpp"
#include "Albany_EigendataInfoStruct.hpp"
#include "Epetra_Import.h"

#include "AnasaziConfigDefs.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziLOBPCGSolMgr.hpp"
#include "AnasaziBasicOutputManager.hpp"
#include "AnasaziEpetraAdapter.hpp"
#include "Epetra_CrsMatrix.h"



QCAD::GenEigensolver::
GenEigensolver(const Teuchos::RCP<Teuchos::ParameterList>& eigensolveParams,
	       const Teuchos::RCP<EpetraExt::ModelEvaluator>& model,
	       const Teuchos::RCP<Albany::StateManager>& observer,
	       Teuchos::RCP<const Epetra_Comm> comm)
{
  using std::string;
  
  // make a copy of the appParams, since we modify them below (e.g. discretization list)
  Teuchos::RCP<Teuchos::ParameterList> myParams = Teuchos::rcp( new Teuchos::ParameterList(*eigensolveParams) );

  this->model = model;
  this->observer = observer;

  EpetraExt::ModelEvaluator::InArgs model_inArgs = model->createInArgs();
  EpetraExt::ModelEvaluator::OutArgs model_outArgs = model->createOutArgs();
  model_num_p = model_inArgs.Np();   // Number of *vectors* of parameters
  model_num_g = model_outArgs.Ng();  // Number of *vectors* of responses

  which = "SM"; //always get smallest eigenvalues
  bHermitian = myParams->get<bool>("Symmetric",true);
  nev = myParams->get<int>("Num Eigenvalues",10);
  blockSize = myParams->get<int>("Block Size",5);
  maxIters = myParams->get<int>("Maximum Iterations",500);
  conv_tol = myParams->get<double>("Convergece Tolerance",1.0e-8);

  myComm = comm;
}

QCAD::GenEigensolver::~GenEigensolver()
{
}


Teuchos::RCP<const Epetra_Map> QCAD::GenEigensolver::get_x_map() const
{
  Teuchos::RCP<const Epetra_Map> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Map> QCAD::GenEigensolver::get_f_map() const
{
  Teuchos::RCP<const Epetra_Map> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Map> QCAD::GenEigensolver::get_p_map(int l) const
{
  return model->get_p_map(l);
}

Teuchos::RCP<const Epetra_Map> QCAD::GenEigensolver::get_g_map(int j) const
{
  if (j == model_num_g) return model->get_x_map(); //last response vector is solution (same map as x)
  else return model->get_g_map(j);
}

Teuchos::RCP<const Epetra_Vector> QCAD::GenEigensolver::get_x_init() const
{
  Teuchos::RCP<const Epetra_Vector> neverused;
  return neverused;  
}

Teuchos::RCP<const Epetra_Vector> QCAD::GenEigensolver::get_x_dot_init() const
{
  Teuchos::RCP<const Epetra_Vector> neverused;
  return neverused;  
}


Teuchos::RCP<const Epetra_Vector> QCAD::GenEigensolver::get_p_init(int l) const
{
  return model->get_p_init(l);
}


EpetraExt::ModelEvaluator::InArgs QCAD::GenEigensolver::createInArgs() const
{
  InArgsSetup inArgs;
  inArgs.setModelEvalDescription("QCAD Generalized Eigensolver Model Evaluator");
  inArgs.set_Np(model_num_p);
  return inArgs;
}

EpetraExt::ModelEvaluator::OutArgs QCAD::GenEigensolver::createOutArgs() const
{
  //Based on Piro_Epetra_NOXSolver.cpp implementation
  OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("QCAD Generalized Eigensolver Model Evaluator");

  // Ng is 1 bigger then model's Ng so that the solution vector can be an outarg
  outArgs.set_Np_Ng(model_num_p, model_num_g+1);

  //Derivative info 
  EpetraExt::ModelEvaluator::OutArgs model_outArgs = model->createOutArgs();
  for (int i=0; i<model_num_g; i++) {
    for (int j=0; j<model_num_p; j++)
      outArgs.setSupports(OUT_ARG_DgDp, i, j, model_outArgs.supports(OUT_ARG_DgDp, i, j));
  }

  return outArgs;
}


void 
QCAD::GenEigensolver::evalModel(const InArgs& inArgs,
			const OutArgs& outArgs ) const
{  
  // type definitions
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;
  
  // Get the stiffness and mass matrices
  InArgs model_inArgs = model->createInArgs();
  OutArgs model_outArgs = model->createOutArgs();

  //input args
  model_inArgs.set_t(0.0);

  Teuchos::RCP<const Epetra_Vector> x = model->get_x_init();
  Teuchos::RCP<const Epetra_Vector> x_dot = model->get_x_dot_init();
  model_inArgs.set_x(x);
  model_inArgs.set_x_dot(x_dot);

  model_inArgs.set_alpha(0.0);
  model_inArgs.set_beta(1.0);

  for(int i=0; i<model_num_p; i++)
    model_inArgs.set_p(i, inArgs.get_p(i));
  
  //output args
  Teuchos::RCP<Epetra_CrsMatrix> K = 
    Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(model->create_W(), true);
  model_outArgs.set_W(K); 

  model->evalModel(model_inArgs, model_outArgs); //compute K matrix

  // reset alpha and beta to compute the mass matrix
  model_inArgs.set_alpha(1.0);
  model_inArgs.set_beta(0.0);
  Teuchos::RCP<Epetra_CrsMatrix> M = 
    Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(model->create_W(), true);
  model_outArgs.set_W(M); 

  model->evalModel(model_inArgs, model_outArgs); //compute M matrix

  Teuchos::RCP<Epetra_MultiVector> ivec = Teuchos::rcp( new Epetra_MultiVector(K->OperatorDomainMap(), blockSize) );
  ivec->Random();

  // Create the eigenproblem.
  Teuchos::RCP<Anasazi::BasicEigenproblem<double, MV, OP> > eigenProblem =
    Teuchos::rcp( new Anasazi::BasicEigenproblem<double, MV, OP>(K, M, ivec) );

  // Inform the eigenproblem that the operator A is symmetric
  eigenProblem->setHermitian(bHermitian);

  // Set the number of eigenvalues requested
  eigenProblem->setNEV( nev );

  // Inform the eigenproblem that you are finishing passing it information
  bool bSuccess = eigenProblem->setProblem();
  TEUCHOS_TEST_FOR_EXCEPTION(!bSuccess, Teuchos::Exceptions::InvalidParameter,
     "Anasazi::BasicEigenproblem::setProblem() returned an error.\n" << std::endl);

  // Create parameter list to pass into the solver manager
  //
  Teuchos::ParameterList eigenPL;
  eigenPL.set( "Which", which );
  eigenPL.set( "Block Size", blockSize );
  eigenPL.set( "Maximum Iterations", maxIters );
  eigenPL.set( "Convergence Tolerance", conv_tol );
  eigenPL.set( "Full Ortho", true );
  eigenPL.set( "Use Locking", true );
  eigenPL.set( "Verbosity", Anasazi::IterationDetails );

  // Create the solver manager
  Anasazi::LOBPCGSolMgr<double, MV, OP> eigenSolverMan(eigenProblem, eigenPL);

  // Solve the problem
  Anasazi::ReturnType returnCode = eigenSolverMan.solve();

  // Get the eigenvalues and eigenvectors from the eigenproblem
  Anasazi::Eigensolution<double,MV> sol = eigenProblem->getSolution();
  std::vector<Anasazi::Value<double> > evals = sol.Evals;
  Teuchos::RCP<MV> evecs = sol.Evecs;

  std::vector<double> evals_real(sol.numVecs);
  for(int i=0; i<sol.numVecs; i++) evals_real[i] = evals[i].realpart;

  // Compute residuals.
  std::vector<double> normR(sol.numVecs);
  if (sol.numVecs > 0) {
    Teuchos::SerialDenseMatrix<int,double> T(sol.numVecs, sol.numVecs);
    Epetra_MultiVector Kvec( K->OperatorDomainMap(), evecs->NumVectors() );
    Epetra_MultiVector Mvec( M->OperatorDomainMap(), evecs->NumVectors() );
    T.putScalar(0.0); 
    for (int i=0; i<sol.numVecs; i++) {
      T(i,i) = evals_real[i];
    }
    K->Apply( *evecs, Kvec );  
    M->Apply( *evecs, Mvec );  
    MVT::MvTimesMatAddMv( -1.0, Mvec, T, 1.0, Kvec );
    MVT::MvNorm( Kvec, normR );
  }

  // Print the results
  std::ostringstream os;
  os.setf(std::ios_base::right, std::ios_base::adjustfield);
  os<<"Solver manager returned " << (returnCode == Anasazi::Converged ? "converged." : "unconverged.") << std::endl;
  os<<std::endl;
  os<<"------------------------------------------------------"<<std::endl;
  os<<std::setw(16)<<"Eigenvalue"
    <<std::setw(18)<<"Direct Residual"
    <<std::endl;
  os<<"------------------------------------------------------"<<std::endl;
  for (int i=0; i<sol.numVecs; i++) {
    os<<std::setw(16)<<evals_real[i]
      <<std::setw(18)<<normR[i]/evals_real[i]
      <<std::endl;
  }
  os<<"------------------------------------------------------"<<std::endl;

  std::cout << Anasazi::Anasazi_Version() << std::endl << std::endl;
  std::cout << os.str();



  // Package the results in an eigendata structure and "observe" them 
  //   (put them into the user-supplied StateManager object)  (see Albany_SaveEigenData.cpp)
  Teuchos::RCP<Albany::EigendataStruct> eigenData = Teuchos::rcp( new Albany::EigendataStruct );
  eigenData->eigenvalueIm = Teuchos::null;  // eigenvalues are real
  eigenData->eigenvectorIm = Teuchos::null; // eigenvectors are real

  Teuchos::RCP<Albany::AbstractDiscretization> disc = 
    observer->getDiscretization();

  eigenData->eigenvalueRe = Teuchos::rcp( new std::vector<double>(evals_real) );
  for(int i=0; i<sol.numVecs; i++) (*(eigenData->eigenvalueRe))[i] *= -1; 
      //make eigenvals --> neg_eigenvals to mimic historic LOCA eigensolver (TODO: remove this and switch convention)

  if (sol.numVecs > 0) {
    // Store *overlapped* eigenvectors in EigendataStruct
    eigenData->eigenvectorRe = 
      Teuchos::rcp(new Epetra_MultiVector(*(disc->getOverlapMap()), sol.numVecs));

    // Importer for overlapped data
    Teuchos::RCP<Epetra_Import> importer =
      Teuchos::rcp(new Epetra_Import(*(disc->getOverlapMap()), *(disc->getMap())));

    // Overlapped eigenstate vectors
    for(int i=0; i<sol.numVecs; i++)
      (*(eigenData->eigenvectorRe))(i)->Import( *((*evecs)(i)), *importer, Insert );
  }

  observer->setEigenData(eigenData);
  
}

/*Teuchos::RCP<const Teuchos::ParameterList>
QCAD::GenEigensolver::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList("ValidPoissonSchrodingerProblemParams");

  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0,
                    "Flag to select output of Phalanx Graph and level of detail");

  validPL->set<double>("Length Unit In Meters",1e-6,"Length unit in meters");
  validPL->set<double>("Energy Unit In Electron Volts",1.0,"Energy (voltage) unit in electron volts");
  validPL->set<double>("Temperature",300,"Temperature in Kelvin");
  validPL->set<std::string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
  validPL->set<int>("Number of Eigenvalues",0,"The number of eigenvalue-eigenvector pairs");
  validPL->set<bool>("Verbose Output",false,"Enable detailed output mode");

  validPL->set<bool>("Include exchange-correlation potential",false,"Include exchange-correlation potential in poisson source term");
  validPL->set<bool>("Only solve schrodinger in quantum blocks",true,"Limit schrodinger solution to elements blocks labeled as quantum in the materials DB");

  validPL->sublist("Poisson Problem", false, "");
  validPL->sublist("Schrodinger Problem", false, "");

  // Candidates for deprecation. Pertain to the solution rather than the problem definition.
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  
  return validPL;
}
*/
