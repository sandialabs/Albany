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
#include "Albany_StateInfoStruct.hpp"
#include "Albany_EigendataInfoStruct.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

void CreateSolver(const std::string& solverType, char* xmlfilename, 
		  Teuchos::RCP<Albany::Application>& albApp, 
		  Teuchos::RCP<EpetraExt::ModelEvaluator>& App, 
		  EpetraExt::ModelEvaluator::InArgs& params_in, 
		  EpetraExt::ModelEvaluator::OutArgs& responses_out);

void SolveModel(Teuchos::RCP<Albany::Application>& albApp, Teuchos::RCP<EpetraExt::ModelEvaluator>& App, 
		EpetraExt::ModelEvaluator::InArgs params_in, EpetraExt::ModelEvaluator::OutArgs responses_out,
    		Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates,
		Teuchos::RCP<Albany::EigendataStruct>& pInitialEData, 
		Teuchos::RCP<Albany::EigendataStruct>& pFinalEData);


void CopyStateToContainer(Albany::StateArrays& src,
			  std::string stateNameToCopy,
			  std::vector<Intrepid::FieldContainer<RealType> >& dest);

void CopyContainerToState(std::vector<Intrepid::FieldContainer<RealType> >& src,
			  Albany::StateArrays& dest,
			  std::string stateNameOfCopy);

void CopyState(Albany::StateArrays& src, Albany::StateArrays& dest,  std::string stateNameToCopy);

void AddStateToState(Albany::StateArrays& src, std::string srcStateNameToAdd, 
		     Albany::StateArrays& dest, std::string destStateNameToAddTo);

void SubtractStateFromState(Albany::StateArrays& src, std::string srcStateNameToSubtract,
			    Albany::StateArrays& dest, std::string destStateNameToSubtractFrom);

bool checkConvergence(Albany::StateArrays& states, 
		      std::vector<Intrepid::FieldContainer<RealType> >& prevState,
		      std::string stateName, double tol);

double GetEigensolverShift(Albany::StateArrays& states, const std::string& stateNameToBaseShiftOn);



#include "Teuchos_ParameterList.hpp"
#include "Piro_Epetra_LOCASolver.hpp"
void ResetEigensolverShift(Teuchos::RCP<EpetraExt::ModelEvaluator>& Solver, double newShift);
Teuchos::RCP<Teuchos::ParameterList> eigList;


int main(int argc, char *argv[]) {

  int status=0;
  bool success = true;
  int maxIter = 0;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv);

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  //Field Containers to store states used in Poisson-Schrodinger loop
  std::vector<Intrepid::FieldContainer<RealType> > prevElectricPotential;
  std::vector<Intrepid::FieldContainer<RealType> > tmpContainer;

  // Command-line argument for input file
  char * PoissonXmlFilename=0;
  char * SchrodingerXmlFilename=0;
  if(argc>3){
    if(!strcmp(argv[1],"--help")){
      printf("albanyQCAD [PoissonInputfile.xml] [SchrodingerInputfile.xml] [maxiter]\n");
      exit(1);
    }
    else {

      PoissonXmlFilename=argv[1];
      SchrodingerXmlFilename=argv[2];
      maxIter = atoi(argv[3]);
    }
  }
  else {
    printf("albanyQCAD [PoissonInputfile.xml] [SchrodingerInputfile.xml] [maxiter]\n");
    exit(1);
  }
  
  try {
    RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

    RCP<Albany::Application> initPoissonApp, poissonApp, dummyPoissonApp, schrodingerApp;
    RCP<EpetraExt::ModelEvaluator> initPoissonSolver, poissonSolver, dummyPoissonSolver, schrodingerSolver;

    EpetraExt::ModelEvaluator::InArgs initPoisson_params_in, 
                                      poisson_params_in,dummy_params_in,
                                      schrodinger_params_in;

    EpetraExt::ModelEvaluator::OutArgs initPoisson_responses_out, 
                                       poisson_responses_out,dummy_responses_out, 
                                       schrodinger_responses_out;

    RCP<Teuchos::Time> totalTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: ***Total Time***");
    RCP<Teuchos::Time> setupTime = 
      Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time");
    Teuchos::TimeMonitor totalTimer(*totalTime); //start timer
    Teuchos::TimeMonitor setupTimer(*setupTime); //start timer

    *out << "QCAD Solve: creating initial Poisson solver using input " << PoissonXmlFilename << endl;
    CreateSolver("initial poisson", PoissonXmlFilename, initPoissonApp, initPoissonSolver, 
      initPoisson_params_in, initPoisson_responses_out);

    *out << "QCAD Solve: creating Poisson solver using input " << PoissonXmlFilename << endl;
    CreateSolver("poisson", PoissonXmlFilename, poissonApp, poissonSolver, 
      poisson_params_in, poisson_responses_out);

    *out << "QCAD Solve: creating dummy Poisson solver using input " << PoissonXmlFilename << endl;
    CreateSolver("dummy poisson", PoissonXmlFilename, dummyPoissonApp, dummyPoissonSolver, 
      dummy_params_in, dummy_responses_out);

    *out << "QCAD Solve: creating Schrodinger solver using input " << SchrodingerXmlFilename << endl;
    CreateSolver("schrodinger", SchrodingerXmlFilename, schrodingerApp, schrodingerSolver, 
      schrodinger_params_in, schrodinger_responses_out);

    setupTimer.~TimeMonitor();

    //state variables
    Albany::StateArrays* pStatesToPass = NULL;
    Albany::StateArrays* pStatesFromDummy = NULL;
    Albany::StateArrays* pStatesToLoop = NULL; 
    Albany::StateArrays* pLastSavedPotential = new Albany::StateArrays;
    Teuchos::RCP<Albany::EigendataStruct> eigenDataToPass = Teuchos::null;
    Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;
    
    *out << "QCAD Solve: Initial Poisson solve (no quantum region) " << endl;
    SolveModel(initPoissonApp, initPoissonSolver, 
	       initPoisson_params_in, initPoisson_responses_out,
	       pStatesToPass, pStatesToLoop, eigenDataToPass, eigenDataNull);
    eigenDataNull = Teuchos::null;
    
    *out << "QCAD Solve: Beginning Poisson-Schrodinger solve loop" << endl;
    bool bConverged = false; 
    int iter = 0;
    
    // determine if using predictor-corrector method 
    bool bPredictorCorrector = false; 
    Albany::SolverFactory slvrfctry(PoissonXmlFilename, Albany_MPI_COMM_WORLD);
    Teuchos::ParameterList& appParams = slvrfctry.getParameters();
    bPredictorCorrector = appParams.sublist("Problem").sublist("Schrodinger Coupling").get<bool>("Use predictor-corrector method",false);
    
    while(!bConverged && iter < maxIter) 
    {
      iter++;
 
      double newShift = GetEigensolverShift(*pStatesToLoop, "Conduction Band");
      ResetEigensolverShift(schrodingerSolver, newShift);

      *out << "QCAD Solve: Schrodinger iteration " << iter << endl;
      SolveModel(schrodingerApp, schrodingerSolver,
        schrodinger_params_in, schrodinger_responses_out,
        pStatesToLoop, pStatesToPass, eigenDataNull, eigenDataToPass);

      CopyStateToContainer(*pStatesToLoop, "Saved Solution", tmpContainer);
      CopyContainerToState(tmpContainer, *pStatesToPass, "Previous Poisson Potential");

      *out << "QCAD Solve: Poisson iteration " << iter << endl;
      SolveModel(poissonApp, poissonSolver, 
        poisson_params_in, poisson_responses_out,
        pStatesToPass, pStatesToLoop, eigenDataToPass, eigenDataNull);

      if (!bPredictorCorrector)  // not use the predictor-corrector method
      {
        *out << "QCAD Solve: Poisson Dummy iteration " << iter << endl;
        SolveModel(dummyPoissonApp, dummyPoissonSolver, 
          dummy_params_in, dummy_responses_out,
          pStatesToPass, pStatesFromDummy, eigenDataToPass, eigenDataNull);

        AddStateToState(*pStatesFromDummy, "Electric Potential", *pStatesToLoop, "Conduction Band");
      }
      
      eigenDataNull = Teuchos::null;
      if(iter > 1) 
        bConverged = checkConvergence(*pStatesToLoop, prevElectricPotential, "Electric Potential", 1e-6);
      
      //CopyState(*pLastSavedPotential, *pStatesToLoop, "Electric Potential");
      CopyStateToContainer(*pStatesToLoop, "Electric Potential", prevElectricPotential);
    } 

    if(bConverged)
      *out << "QCAD Solve: Converged Poisson-Schrodinger solve loop after " << iter << " iterations." << endl;
    else
      *out << "QCAD Solve: Maximum iterations (" << maxIter << ") reached." << endl;

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


void CreateSolver(const std::string& solverType, char* xmlfilename, 
		  Teuchos::RCP<Albany::Application>& albApp, Teuchos::RCP<EpetraExt::ModelEvaluator>& App, 
		  EpetraExt::ModelEvaluator::InArgs& params_in, EpetraExt::ModelEvaluator::OutArgs& responses_out)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  //! Create solver factory, which reads xml input filen
  Albany::SolverFactory slvrfctry(xmlfilename, Albany_MPI_COMM_WORLD);
    
  //! Process input parameters based on solver type before creating solver & application
  Teuchos::ParameterList& appParams = slvrfctry.getParameters();

  if(solverType == "initial poisson") {
    //! Turn off schrodinger source
    appParams.sublist("Problem").sublist("Schrodinger Coupling").set<bool>("Schrodinger source in quantum blocks",false);

    //! Rename output file
    std::string exoName= "init" + appParams.sublist("Discretization").get<std::string>("Exodus Output File Name");
    appParams.sublist("Discretization").set("Exodus Output File Name", exoName);
  }

  else if(solverType == "dummy poisson") {
    //! Rename materials file
    std::string mtrlName= "dummy_" + appParams.sublist("Problem").get<std::string>("MaterialDB Filename");
    appParams.sublist("Problem").set("MaterialDB Filename", mtrlName);

    //! Rename output file
    std::string exoName= "dummy" + appParams.sublist("Discretization").get<std::string>("Exodus Output File Name");
    appParams.sublist("Discretization").set("Exodus Output File Name", exoName);
 
    //! Replace Dirichlet BCs and Parameters sublists with dummy versions
    appParams.sublist("Problem").sublist("Dirichlet BCs") = 
      appParams.sublist("Problem").sublist("Dummy Dirichlet BCs");
    appParams.sublist("Problem").sublist("Parameters") = 
      appParams.sublist("Problem").sublist("Dummy Parameters");
  }

  //! Create solver and application objects via solver factory
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
		Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates,
		Teuchos::RCP<Albany::EigendataStruct>& pInitialEData, Teuchos::RCP<Albany::EigendataStruct>& pFinalEData)
{
  if(pInitialStates) albApp->getStateMgr().importStateData( *pInitialStates );
  if(pInitialEData != Teuchos::null) albApp->getStateMgr().setEigenData(pInitialEData);
  App->evalModel(params_in, responses_out);
  pFinalStates = &(albApp->getStateMgr().getStateArrays());
  pFinalEData = albApp->getStateMgr().getEigenData();
}


void CopyStateToContainer(Albany::StateArrays& src,
			  std::string stateNameToCopy,
			  std::vector<Intrepid::FieldContainer<RealType> >& dest)
{
  int numWorksets = src.size();
  std::vector<int> dims;

  //allocate destination container if necessary
  if(dest.size() != (unsigned int)numWorksets) {
    dest.resize(numWorksets);    
    for (int ws = 0; ws < numWorksets; ws++) {
      src[ws][stateNameToCopy].dimensions(dims);
      dest[ws].resize(dims);
    }
  }

  for (int ws = 0; ws < numWorksets; ws++)
  {
    src[ws][stateNameToCopy].dimensions(dims);
    TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++)
      for(int qp=0; qp < dims[1]; qp++)
        dest[ws](cell,qp) = src[ws][stateNameToCopy](cell,qp);
  }
}


//Note: state must be allocated already
void CopyContainerToState(std::vector<Intrepid::FieldContainer<RealType> >& src,
			  Albany::StateArrays& dest,
			  std::string stateNameOfCopy)
{
  int numWorksets = src.size();
  std::vector<int> dims;

  for (int ws = 0; ws < numWorksets; ws++)
  {
    dest[ws][stateNameOfCopy].dimensions(dims);
    TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++)
      for(int qp=0; qp < dims[1]; qp++)
        dest[ws][stateNameOfCopy](cell,qp) = src[ws](cell,qp);
  }
}


//Note: assumes src and dest have allocated states of <stateNameToCopy>
void CopyState(Albany::StateArrays& src,
	       Albany::StateArrays& dest,
	       std::string stateNameToCopy)
{
  int numWorksets = src.size();
  int totalSize;

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][stateNameToCopy].size();
    for(int i=0; i<totalSize; ++i)
      dest[ws][stateNameToCopy][i] = src[ws][stateNameToCopy][i];
  }
}


void AddStateToState(Albany::StateArrays& src,
		     std::string srcStateNameToAdd, 
		     Albany::StateArrays& dest,
		     std::string destStateNameToAddTo)
{
  int totalSize, numWorksets = src.size();
  TEST_FOR_EXCEPT( numWorksets != (int)dest.size() );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][srcStateNameToAdd].size();
    
    for(int i=0; i<totalSize; ++i)
      dest[ws][destStateNameToAddTo][i] += src[ws][srcStateNameToAdd][i];
  }
}


void SubtractStateFromState(Albany::StateArrays& src, 
			    std::string srcStateNameToSubtract,
			    Albany::StateArrays& dest,
			    std::string destStateNameToSubtractFrom)
{
  int totalSize, numWorksets = src.size();
  TEST_FOR_EXCEPT( numWorksets != (int)dest.size() );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][srcStateNameToSubtract].size();
    
    for(int i=0; i<totalSize; ++i)
      dest[ws][destStateNameToSubtractFrom][i] -= src[ws][srcStateNameToSubtract][i];
  }
}


bool checkConvergence(Albany::StateArrays& states, 
		      std::vector<Intrepid::FieldContainer<RealType> >& prevState,
		      std::string stateName, double tol)
{
  int totalSize, numWorksets = states.size();
  std::vector<int> dims;

  TEST_FOR_EXCEPT( ! (numWorksets == (int)prevState.size()) );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    states[ws][stateName].dimensions(dims);
    TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++) 
    {
      for(int qp=0; qp < dims[1]; qp++) 
      {
        // std::cout << "prevState = " << prevState[ws](cell,qp) << std::endl;
        // std::cout << "currState = " << states[ws][stateName](cell,qp) << std::endl;
        if( fabs( states[ws][stateName](cell,qp) - prevState[ws](cell,qp) ) > tol )
        {
          // std::cout << "diff = " << fabs( states[ws][stateName](cell,qp) - prevState[ws](cell,qp) ) << std::endl;
          return false;
        }  
      }
    }
  }
  return true;
}


void ResetEigensolverShift(Teuchos::RCP<EpetraExt::ModelEvaluator>& Solver, double newShift) 
{
  Teuchos::RCP<Piro::Epetra::LOCASolver> pels = Teuchos::rcp_dynamic_cast<Piro::Epetra::LOCASolver>(Solver);
  TEST_FOR_EXCEPT(pels == Teuchos::null);

  Teuchos::RCP<LOCA::Stepper> stepper =  pels->getLOCAStepperNonConst();
  const Teuchos::ParameterList& oldEigList = stepper->getList()->sublist("LOCA").sublist("Stepper").sublist("Eigensolver");
  //Teuchos::RCP<Teuchos::ParameterList> eigList =
  eigList =
    Teuchos::rcp(new Teuchos::ParameterList(oldEigList));

  eigList->set("Shift",newShift);

  //cout << " OLD Eigensolver list  " << oldEigList << endl;
  //cout << " NEW Eigensolver list  " << *eigList << endl;
  std::cout << "DEBUG: new eigensolver shift = " << newShift << std::endl;
  stepper->eigensolverReset(eigList);
}


double GetEigensolverShift(Albany::StateArrays& states,
			   const std::string& stateNameToBaseShiftOn)
{
  int numWorksets = states.size();
  std::vector<PHX::DataLayout::size_type> dims;
  const std::string& name = stateNameToBaseShiftOn;

  double val;
  double minVal, maxVal;
  minVal = +1e10; maxVal = -1e10;

  for (int ws = 0; ws < numWorksets; ws++)
  {
    states[ws][name].dimensions(dims);
    
    int size = dims.size();
    TEST_FOR_EXCEPTION(size != 2, std::logic_error, "Unimplemented number of dimensions");
    int cells = dims[0];
    int qps = dims[1];

    for (int cell = 0; cell < cells; ++cell)  {
      for (int qp = 0; qp < qps; ++qp) {
        val = states[ws][name](cell, qp);
        if(val < minVal) minVal = val;
        if(val > maxVal) maxVal = val;
      }
    }
  }

  //set shift to be slightly (5% of range) below minimum value
  double shift = -(minVal - 0.05*(maxVal-minVal)); //minus sign b/c negative eigenvalue convention
  return shift;
}
  



//UNUSED - but keeping around for reference for a little while
/*void CopyState(Albany::StateArrays& dest, 
	       Albany::StateArrays& src,
	       std::string stateNameToCopy)
{
  int numWorksets = src.size();
  int totalSize;
  std::vector<int> dims;

  //allocate destination if necessary
  if(dest.size() != (unsigned int)numWorksets) {
    dest.resize(numWorksets);    
    for (int ws = 0; ws < numWorksets; ws++) {
      src[ws][stateNameToCopy].dimensions(dims);
      totalSize = src[ws][stateNameToCopy].size();
      double* pData = new double[totalSize];

      TEST_FOR_EXCEPT( dims.size() != 2 );
      dest[ws][stateNameToCopy].assign<shards::ArrayDimension,shards::ArrayDimension>(pData, dims[0], dims[1]);
    }
  }

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][stateNameToCopy].size();
    
    for(int i=0; i<totalSize; ++i)
      dest[ws][stateNameToCopy][i] = src[ws][stateNameToCopy][i];
  }
}
*/
