//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_Adaptation.hpp"

#include "Albany_STKDiscretization.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Albany {

Adaptation::Adaptation(const Teuchos::RCP<Teuchos::ParameterList>& appParams){
#if 0
       const Teuchos::RCP<const Epetra_Comm>& comm) :
  maxIter(0),CONVERGE_TOL(1e-6)
{
  using std::string;

  num_p = 1; // Only use first parameter (p)
  num_g = 1; // First response vector (but really 2 vectors b/c solution is 2nd vector)

  // Get sub-problem input xml files from problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");

  problemName = problemParams.get<string>("Name");
  std::size_t nProblems = problemParams.get<int>("Number of Problems");

  //validate problemParams here?

  std::map<std::string, std::string> inputFilenames;

  // Collect sub-problem input file names
  for(std::size_t i=0; i<nProblems; i++) {
    Teuchos::ParameterList& subProblemParams = problemParams.sublist(Albany::strint("Subproblem",i));
    string subName = subProblemParams.get<string>("Name");
    string inputFile = subProblemParams.get<string>("Input Filename");
    inputFilenames[subName] = inputFile;
  }

  // Check if "verbose" mode is enabled
  bVerbose = (comm->MyPID() == 0) && problemParams.get<bool>("Verbose Output", false);

  // Get problem parameters specific to certain problems
  if( problemName == "Poisson Schrodinger" ||
      problemName == "Poisson CI") {
    maxIter = problemParams.get<int>("Maximum Iterations", 100);
    iterationMethod = problemParams.get<string>("Iteration Method", "Picard");
    shiftPercentBelowMin = problemParams.get<double>("Eigensolver Percent Shift Below Potential Min", 1.0);
    CONVERGE_TOL = problemParams.get<double>("Convergence Tolerance", 1e-6);
  }

  // Create Solver(s) based on problem name

  if( problemName == "Poisson" ) {
    subSolvers["Poisson"] = CreateSubSolver(inputFilenames["Poisson"], "none", *comm);
  }

  else if( problemName == "Poisson Schrodinger" ) {
    subSolvers["InitPoisson"] = CreateSubSolver(inputFilenames["Poisson"], "initial poisson", *comm);
    subSolvers["Poisson"]     = CreateSubSolver(inputFilenames["Poisson"], "none", *comm);
    subSolvers["Schrodinger"] = CreateSubSolver(inputFilenames["Schrodinger"], "none", *comm);
  }

  else if( problemName == "Poisson CI" ) {
    subSolvers["InitPoisson"]    = CreateSubSolver(inputFilenames["Poisson"], "initial poisson", *comm);
    subSolvers["Poisson"]        = CreateSubSolver(inputFilenames["Poisson"], "none", *comm);
    subSolvers["DeltaPoisson"]   = CreateSubSolver(inputFilenames["Poisson"], "Delta poisson", *comm);
    subSolvers["CoulombPoisson"] = CreateSubSolver(inputFilenames["Poisson"], "Coulomb poisson", *comm);
    subSolvers["CIPoisson"]      = CreateSubSolver(inputFilenames["Poisson"], "CI poisson", *comm);
    subSolvers["Schrodinger"]    = CreateSubSolver(inputFilenames["Schrodinger"], "none", *comm);

  }

  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in QCAD::Solver constructor:  " <<
				  "Invalid problem name " << problemName << std::endl);

  //Determine whether we should support DgDp (all sub-solvers must support DpDg for QCAD::Solver to)
  bSupportDpDg = true;  
  std::map<std::string, SolverSubSolver>::const_iterator it;
  for(it = subSolvers.begin(); it != subSolvers.end(); ++it) {
    EpetraExt::ModelEvaluator::OutArgs model_outargs = (it->second.model)->createOutArgs();
    if( model_outargs.supports(OUT_ARG_DgDp, 0, 0).none() ) { //just test if p=0, g=0 DgDp is supported
      bSupportDpDg = false;
      break;
    }
  }

  //Save comm for evaluation
  solverComm = comm;


  //Setup Parameter and responses maps
  
  // input file can have 
  //    <Parameter name="Parameter 0" type="string" value="Poisson[0]" />
  //    <Parameter name="Parameter 1" type="string" value="Poisson[1:3]" />
  //
  //    <Parameter name="Response 0" type="string" value="Poisson[0] # charge" />
  //    <Parameter name="Response 0" type="string" value="Schrodinger[1,3]" />
  //    <Parameter name="Response 0" type="string" value="=dist(Poisson[1:4],Poisson[4:7]) # distance example" />

  Teuchos::ParameterList& paramList = problemParams.sublist("Parameters");
  setupParameterMapping(paramList);

  Teuchos::ParameterList& responseList = problemParams.sublist("Response Functions");
  setupResponseMapping(responseList);

  // Create Epetra map for parameter vector (only one since num_p always == 1)
  epetra_param_map = Teuchos::rcp(new Epetra_LocalMap(nParameters, 0, *comm));

  // Create Epetra map for (first) response vector
  epetra_response_map = Teuchos::rcp(new Epetra_LocalMap(nResponseDoubles, 0, *comm));
     //ANDY: if (nResponseDoubles > 0) needed ??

  // Create Epetra map for solution vector (second response vector).  Assume 
  //  each subSolver has the same map, so just get the first one.
  Teuchos::RCP<const Epetra_Map> sub_x_map = (subSolvers.begin()->second).app->getMap();
  TEUCHOS_TEST_FOR_EXCEPT( sub_x_map == Teuchos::null );
  epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));
#endif
}


/*
Adaptation::Adaptation(const Teuchos::RCP<AbstractDiscretization> &disc) :
   stkDisc_(Teuchos::rcp_dynamic_cast<STKDiscretization>(disc)),
   exoOutTime_(Teuchos::TimeMonitor::getNewTimer("Albany: Output to Exodus"))
{
  // Nothing to oo
}
*/

void Adaptation::writeSolution(double stamp, const Epetra_Vector &solution)
{
//   Teuchos::TimeMonitor exoOutTimer(*exoOutTime_);
   stkDisc_->outputToExodus(solution, stamp);
}

}
