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


#include "QCAD_Solver.hpp"
#include "Piro_Epetra_LOCASolver.hpp"
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Sacado_PCE_OrthogPoly.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_EigendataInfoStruct.hpp"



QCAD::SolverSubSolver CreateSubSolver(const std::string xmlfilename, 
				const std::string& xmlPreprocessType, const Epetra_Comm& comm);
void preprocessParams(Teuchos::ParameterList& params, std::string preprocessType);

void SolveModel(const QCAD::SolverSubSolver& ss);
void SolveModel(const QCAD::SolverSubSolver& ss, 
		Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates);
void SolveModel(const QCAD::SolverSubSolver& ss, 
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

double getMaxDifference(Albany::StateArrays& states, 
			std::vector<Intrepid::FieldContainer<RealType> >& prevState,
			std::string stateName);

void ResetEigensolverShift(const Teuchos::RCP<EpetraExt::ModelEvaluator>& Solver, double newShift,
			   Teuchos::RCP<Teuchos::ParameterList>& eigList);
double GetEigensolverShift(const QCAD::SolverSubSolver& ss, int minPotentialResponseIndex);


//String processing helper functions
std::vector<std::string> string_split(const std::string& s, char delim, bool bProtect=false);
std::string string_remove_whitespace(const std::string& s);
std::vector<std::string> string_parse_function(const std::string& s);
std::map<std::string,std::string> string_parse_arrayref(const std::string& s);
std::vector<int> string_expand_compoundindex(const std::string& indexStr, int min_index, int max_index);



QCAD::Solver::
Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
       const Teuchos::RCP<const Epetra_Comm>& comm) :
  maxIter(0)
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
  }

  // Create Solver(s) based on problem name

  if( problemName == "Poisson" ) {
    subSolvers["Poisson"] = CreateSubSolver(inputFilenames["Poisson"], "none", *comm);
  }

  else if( problemName == "Poisson Schrodinger" ) {
    subSolvers["InitPoisson"] = CreateSubSolver(inputFilenames["Poisson"], "initial poisson", *comm);
    subSolvers["Poisson"] = CreateSubSolver(inputFilenames["Poisson"], "none", *comm);
    subSolvers["Schrodinger"] = CreateSubSolver(inputFilenames["Schrodinger"], "none", *comm);
  }

  else if( problemName == "Poisson CI" ) {
    subSolvers["InitPoisson"] = CreateSubSolver(inputFilenames["Poisson"], "initial poisson", *comm);
    subSolvers["Poisson"] = CreateSubSolver(inputFilenames["Poisson"], "none", *comm);
    subSolvers["DummyPoisson"] = CreateSubSolver(inputFilenames["Poisson"], "dummy poisson", *comm);
    subSolvers["Schrodinger"] = CreateSubSolver(inputFilenames["Schrodinger"], "none", *comm);
  }

  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in QCAD::Solver constructor:  " <<
				  "Invalid problem name " << problemName << std::endl);

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
}

QCAD::Solver::~Solver()
{
}


Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_x_map() const
{
  Teuchos::RCP<const Epetra_Map> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_f_map() const
{
  Teuchos::RCP<const Epetra_Map> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_p_map(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= num_p || l < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::Solver::get_p_map():  " <<
                     "Invalid parameter index l = " <<
                     l << std::endl);

  return epetra_param_map;  //no index because num_p == 1 so l must be zero
}

Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_g_map(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(j > num_g || j < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::Solver::get_g_map():  " <<
                     "Invalid response index j = " <<
                     j << std::endl);

  if      (j < num_g) return epetra_response_map;  //no index because num_g == 1 so j must be zero
  else if (j == num_g) return epetra_x_map;
  return Teuchos::null;
}

Teuchos::RCP<const Epetra_Vector> QCAD::Solver::get_x_init() const
{
  Teuchos::RCP<const Epetra_Vector> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Vector> QCAD::Solver::get_p_init(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= num_p || l < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::Solver::get_p_init():  " <<
                     "Invalid parameter index l = " <<
                     l << std::endl);

  Teuchos::RCP<Epetra_Vector> p_init = 
    Teuchos::rcp(new Epetra_Vector(*(epetra_param_map)));

  // Take initial value from the first (if multiple) parameter 
  //    fns for each given parameter
  for(std::size_t i=0; i<nParameters; i++) {
    (*p_init)[i] = paramFnVecs[i][0]->getInitialParam(subSolvers);
  }

  return p_init;
}

EpetraExt::ModelEvaluator::InArgs QCAD::Solver::createInArgs() const
{
  EpetraExt::ModelEvaluator::InArgsSetup inArgs;
  inArgs.setModelEvalDescription("QCAD Solver Model Evaluator Description");
  inArgs.set_Np(num_p);
  return inArgs;
}

EpetraExt::ModelEvaluator::OutArgs QCAD::Solver::createOutArgs() const
{
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("QCAD Solver Model Evaluator Description");
  // Ng is 1 bigger then model-Ng so that the solution vector can be an outarg
  outArgs.set_Np_Ng(num_p, num_g+1);

  //Derivative info - todo later. Erik talk w/ ANDY
  /*EpetraExt::ModelEvaluator::OutArgs model_outargs = model->createOutArgs();
  for (int i=0; i<num_g; i++) {
    for (int j=0; j<num_p; j++) {
      
      //TODO - check if param j and response i have simple relationships to 
      // the *same* underlying albany app, in which case derivative info is available...
      if (!model_outargs.supports(OUT_ARG_DgDp, i, j).none())
	outArgs.setSupports(OUT_ARG_DgDp, i, j, 
			    DerivativeSupport(DERIV_MV_BY_COL));
  */

  return outArgs;
}

void 
QCAD::Solver::evalModel(const InArgs& inArgs,
			const OutArgs& outArgs ) const
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // update sub-solver parameters using the main solver's parameter values
  Teuchos::RCP<const Epetra_Vector> p = inArgs.get_p(0); //only use *first* param vector
  std::vector<Teuchos::RCP<QCAD::SolverParamFn> >::const_iterator pit; //XXX const??
  for(std::size_t i=0; i<nParameters; i++) {
    for(pit = paramFnVecs[i].begin(); pit != paramFnVecs[i].end(); pit++) {
      (*pit)->fillSubSolverParams((*p)[i], subSolvers);
    }
  }

  if(bVerbose) {
    *out << "BEGIN QCAD Solver Parameters:" << endl;
    for(std::size_t i=0; i<nParameters; i++)
      *out << "  Parameter " << i << " = " << (*p)[i] << endl;
    *out << "END QCAD Solver Parameters" << endl;
  }
   
  if( problemName == "Poisson" ) {
      if(bVerbose) *out << "QCAD Solve: Simple Poisson solve" << endl;
      SolveModel(getSubSolver("Poisson"));
  }

  else if( problemName == "Poisson Schrodinger" )
    evalPoissonSchrodingerModel(inArgs, outArgs);

  else if( problemName == "Poisson CI" )
    evalPoissonCIModel(inArgs, outArgs);


  // update main solver's responses using sub-solver response values
  Teuchos::RCP<Epetra_Vector> g = outArgs.get_g(0); //only use *first* response vector
  std::vector<Teuchos::RCP<QCAD::SolverResponseFn> >::const_iterator rit;
  int offset = 0;

  for(rit = responseFns.begin(); rit != responseFns.end(); rit++) {
    (*rit)->fillSolverResponses( *g, offset, subSolvers);
    offset += (*rit)->getNumDoubles();
  }

  if(bVerbose) {
    *out << "BEGIN QCAD Solver Responses:" << endl;
    for(int i=0; i< g->MyLength(); i++)
      *out << "  Response " << i << " = " << (*g)[i] << endl;
    *out << "END QCAD Solver Responses" << endl;
  }
}


void 
QCAD::Solver::evalPoissonSchrodingerModel(const InArgs& inArgs,
					  const OutArgs& outArgs ) const
{
  const double CONVERGE_TOL = 1e-5;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  //state variables
  Albany::StateArrays* pStatesToPass = NULL;
  Albany::StateArrays* pStatesToLoop = NULL; 
  Teuchos::RCP<Albany::EigendataStruct> eigenDataToPass = Teuchos::null;
  Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;

  //Field Containers to store states used in Poisson-Schrodinger loop
  std::vector<Intrepid::FieldContainer<RealType> > prevElectricPotential;
  std::vector<Intrepid::FieldContainer<RealType> > tmpContainer;
  
  if(bVerbose) *out << "QCAD Solve: Initial Poisson solve (no quantum region) " << endl;
  SolveModel(getSubSolver("InitPoisson"), pStatesToPass, pStatesToLoop);
    
  if(bVerbose) *out << "QCAD Solve: Beginning Poisson-Schrodinger solve loop" << endl;
  bool bConverged = false; 
  std::size_t iter = 0;
  double newShift;
    
  // determine if using predictor-corrector method
  //bool bPredictorCorrector = (iterationMethod == "Predictor Corrector");

  Teuchos::RCP<Teuchos::ParameterList> eigList; //used to hold memory I think - maybe unneeded?

  while(!bConverged && iter < maxIter) 
  {
    iter++;

    if (iter == 1) 
      newShift = GetEigensolverShift(getSubSolver("InitPoisson"), 0);
    else
      newShift = GetEigensolverShift(getSubSolver("Poisson"), 0);
    ResetEigensolverShift(getSubSolver("Schrodinger").model, newShift, eigList);

    // Schrodinger Solve -> eigenstates
    if(bVerbose) *out << "QCAD Solve: Schrodinger iteration " << iter << endl;
    SolveModel(getSubSolver("Schrodinger"), pStatesToLoop, pStatesToPass,
	       eigenDataNull, eigenDataToPass);

    // Save solution for predictory-corrector outer iterations      
    CopyStateToContainer(*pStatesToLoop, "Saved Solution", tmpContainer);
    CopyContainerToState(tmpContainer, *pStatesToPass, "Previous Poisson Potential");

    // Poisson Solve
    if(bVerbose) *out << "QCAD Solve: Poisson iteration " << iter << endl;
    SolveModel(getSubSolver("Poisson"), pStatesToPass, pStatesToLoop,
	       eigenDataToPass, eigenDataNull);

    eigenDataNull = Teuchos::null;

    if(iter > 1) {
      double local_maxDiff = getMaxDifference(*pStatesToLoop, prevElectricPotential, "Electric Potential");
      double global_maxDiff;
      solverComm->MaxAll(&local_maxDiff, &global_maxDiff, 1);
      bConverged = (global_maxDiff < CONVERGE_TOL);
      if(bVerbose) *out << "QCAD Solve: Electric Potential max diff=" 
			<< global_maxDiff << " (tol=" << CONVERGE_TOL << ")" << std::endl;
    }
      
    CopyStateToContainer(*pStatesToLoop, "Electric Potential", prevElectricPotential);
  } 

  if(bVerbose) {
    if(bConverged)
      *out << "QCAD Solve: Converged Poisson-Schrodinger solve loop after " << iter << " iterations." << endl;
    else
      *out << "QCAD Solve: Maximum iterations (" << maxIter << ") reached." << endl;
  }
}


void 
QCAD::Solver::evalPoissonCIModel(const InArgs& inArgs,
				 const OutArgs& outArgs ) const
{
  const double CONVERGE_TOL = 1e-5;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  //state variables
  Albany::StateArrays* pStatesToPass = NULL;
  Albany::StateArrays* pStatesFromDummy = NULL;
  Albany::StateArrays* pStatesToLoop = NULL; 
  Teuchos::RCP<Albany::EigendataStruct> eigenDataToPass = Teuchos::null;
  Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;

  //Field Containers to store states used in Poisson-Schrodinger loop
  std::vector<Intrepid::FieldContainer<RealType> > prevElectricPotential;
  std::vector<Intrepid::FieldContainer<RealType> > tmpContainer;
 
  if(bVerbose) *out << "QCAD Solve: Initial Poisson solve (no quantum region) " << endl;
  SolveModel(getSubSolver("InitPoisson"), pStatesToPass, pStatesToLoop);
    
  if(bVerbose) *out << "QCAD Solve: Beginning Poisson-CI solve loop" << endl;
  bool bConverged = false;
  std::size_t iter = 0;
  double newShift;

  Teuchos::RCP<Teuchos::ParameterList> eigList; //used to hold memory I think - maybe unneeded?
    
  while(!bConverged && iter < maxIter) 
  {
    iter++;
 
    if (iter == 1) 
      newShift = GetEigensolverShift(getSubSolver("InitPoisson"), 0);
    else
      newShift = GetEigensolverShift(getSubSolver("Poisson"), 0);
    ResetEigensolverShift(getSubSolver("Schrodinger").model, newShift, eigList);

    // Schrodinger Solve -> eigenstates
    if(bVerbose) *out << "QCAD Solve: Schrodinger iteration " << iter << endl;
    SolveModel(getSubSolver("Schrodinger"), pStatesToLoop, pStatesToPass,
	       eigenDataNull, eigenDataToPass);
     
    // Save solution for predictory-corrector outer iterations
    CopyStateToContainer(*pStatesToLoop, "Saved Solution", tmpContainer);
    CopyContainerToState(tmpContainer, *pStatesToPass, "Previous Poisson Potential");
      
    // Construct CI matrices:
    // For N eigenvectors:
    //  1) a NxN matrix of the single particle hamiltonian: H1P. Derivation:
    //    H1P = diag(E) - delta, where delta_ij = int( [i(r)j(r)] F(r) dr)
    //    - no actual poisson solve needed, but need framework to integrate?
    //    - could we save a state containing weights and then integrate outside of NOX?
    //
    //  2) a matrix of all pair integrals.  Derivation:
    //    <12|1/r|34> = int( 1(r1) 2(r2) 1/(r1-r2) 3(r1) 4(r2) dr1 dr2)
    //                = int( 1(r1) 3(r1) [ int( 2(r2) 1/(r1-r2) 4(r2) dr2) ] dr1 )
    //                = int( 1(r1) 3(r1) [ soln of Poisson, rho(r1), with src = 2(r) 4(r) ] dr1 )
    //    - so use dummy poisson solve which has i(r)j(r) as only RHS term, for each pair <ij>,
    //       and as output of each Solve integrate wrt each other potential pair 1(r) 3(r)
    //       to generate all the elements.


    // Poisson Solve
    if(bVerbose) *out << "QCAD Solve: Poisson iteration " << iter << endl;
    SolveModel(getSubSolver("Poisson"), pStatesToPass, pStatesToLoop,
	       eigenDataToPass, eigenDataNull);

    // Dummy Solve (needed?)
    if(bVerbose) *out << "QCAD Solve: Poisson Dummy iteration " << iter << endl;
    SolveModel(getSubSolver("DummyPoisson"), pStatesToPass, pStatesFromDummy,
	       eigenDataToPass, eigenDataNull);
    AddStateToState(*pStatesFromDummy, "Electric Potential", *pStatesToLoop, "Conduction Band");
      
    eigenDataNull = Teuchos::null;

    if(iter > 1) {
      double local_maxDiff = getMaxDifference(*pStatesToLoop, prevElectricPotential, "Electric Potential");
      double global_maxDiff;
      solverComm->MaxAll(&local_maxDiff, &global_maxDiff, 1);
      bConverged = (global_maxDiff < CONVERGE_TOL);

      if(bVerbose) *out << "QCAD Solve: Electric Potential max diff=" 
			<< global_maxDiff << " (tol=" << CONVERGE_TOL << ")" << std::endl;
    }
      
    CopyStateToContainer(*pStatesToLoop, "Electric Potential", prevElectricPotential);
  } 

  if(bVerbose) {
    if(bConverged)
      *out << "QCAD Solve: Converged Poisson-CI solve loop after " << iter << " iterations." << endl;
    else
      *out << "QCAD Solve: Maximum iterations (" << maxIter << ") reached." << endl;
  }
}



void QCAD::Solver::setupParameterMapping(const Teuchos::ParameterList& list)
{
  std::string s;
  std::vector<std::string> fnStrings;

  //default work-around b/c of const
  if( list.isType<int>("Number") )
    nParameters = list.get<int>("Number");
  else nParameters = 0;

  for(std::size_t i=0; i<nParameters; i++) {
    s = list.get<std::string>(Albany::strint("Parameter",i));
    s = string_remove_whitespace(s);
    fnStrings = string_split(s,';',true);
    
    std::vector<Teuchos::RCP<QCAD::SolverParamFn> > fnVec;
    for(std::size_t j=0; j<fnStrings.size(); j++)
      fnVec.push_back( Teuchos::rcp(new QCAD::SolverParamFn(fnStrings[j], subSolvers)) );

    paramFnVecs.push_back( fnVec );
  }
}


void QCAD::Solver::setupResponseMapping(const Teuchos::ParameterList& list)
{
  int number;

  //default work-around b/c of const
  if( list.isType<int>("Number") )
    number = list.get<int>("Number");
  else number = 0;

  std::string s;
  Teuchos::RCP<QCAD::SolverResponseFn> fn;

  nResponseDoubles = 0;
  for(int i=0; i<number; i++) {
    s = list.get<std::string>(Albany::strint("Response",i));
    fn = Teuchos::rcp(new QCAD::SolverResponseFn(s, subSolvers));
    nResponseDoubles += fn->getNumDoubles();
    responseFns.push_back( fn );
  }
}

const QCAD::SolverSubSolver& QCAD::Solver::getSubSolver(const std::string& name) const
{
  return subSolvers.find(name)->second;
}

  

// Parameter Function object

// Function string can be of form:
// "fn1(a,b)>fn2(c,d) ... >SolverName[X:Y]  OR
// "SolverName[X:Y]"
QCAD::SolverParamFn::SolverParamFn(const std::string& fnString, 
			     const std::map<std::string, QCAD::SolverSubSolver>& subSolvers)
{
  std::vector<std::string> fnsAndTarget = string_split(fnString,'>',true);
  std::vector<std::string>::const_iterator it;  
  std::map<std::string,std::string> target;

  if( fnsAndTarget.begin() != fnsAndTarget.end() ) {
    for(it=fnsAndTarget.begin(); it != fnsAndTarget.end()-1; it++) {
      filters.push_back( string_parse_function( *it ) );
    }
    it = fnsAndTarget.end()-1;
    target = string_parse_arrayref( *it );
    targetName = target["name"];
    
    const Epetra_Vector& solver_p = *((subSolvers.find(targetName)->second).params_in->get_p(0));
    targetIndices = string_expand_compoundindex(target["index"], 0, solver_p.MyLength());
  } 
}


void QCAD::SolverParamFn::fillSubSolverParams(double parameterValue, 
   const std::map<std::string, QCAD::SolverSubSolver>& subSolvers) const
{
  std::vector< std::vector<std::string> >::const_iterator fit;
  for(fit = filters.begin(); fit != filters.end(); ++fit) {

    //perform function operation
    std::string fnName = (*fit)[0];
    if( fnName == "scale" ) {
      TEUCHOS_TEST_FOR_EXCEPT( fit->size() != 1+1 ); // "scale" should have 1 parameter
      parameterValue *= atof( (*fit)[1].c_str() );
    }
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	      "Unknown function " << (*fit)[0] << " for given type." << std::endl);
  }

  Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> inArgs = (subSolvers.find(targetName)->second).params_in;
  Teuchos::RCP<const Epetra_Vector> p_ro = inArgs->get_p(0); //only use *first* param vector now
  Teuchos::RCP<Epetra_Vector> p = Teuchos::rcp( new Epetra_Vector( *p_ro ) );
  
  // copy parameterValue into sub-solver parameter vector where appropriate
  std::vector<int>::const_iterator it;
  for(it = targetIndices.begin(); it != targetIndices.end(); ++it)
    (*p)[ (*it) ] = parameterValue;

  inArgs->set_p(0, p);
}

double QCAD::SolverParamFn::
getInitialParam(const std::map<std::string, QCAD::SolverSubSolver>& subSolvers) const
{
  //get first target parameter's initial value
  double initVal;

  Teuchos::RCP<const Epetra_Vector> p_init = 
    (subSolvers.find(targetName)->second).model->get_p_init(0); //only one p vector used

  TEUCHOS_TEST_FOR_EXCEPT(targetIndices.size() == 0);
  initVal = (*p_init)[ targetIndices[0] ];

  std::vector< std::vector<std::string> >::const_iterator fit;
  for(fit = filters.end(); fit != filters.begin(); --fit) {

    //perform INVERSE function operation to back out initial value
    std::string fnName = (*fit)[0];
    if( fnName == "scale" ) {
      TEUCHOS_TEST_FOR_EXCEPT( fit->size() != 1+1 ); // "scale" should have 1 parameter
      initVal /= atof( (*fit)[1].c_str() );
    }
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	      "Unknown function " << (*fit)[0] << " for given type." << std::endl);
  }

  return initVal;
}



// Response Function object

// Function string can be of form:
// "fn1(a,SolverName[X:Y],b)  OR
// "SolverName[X:Y]"
QCAD::SolverResponseFn::SolverResponseFn(const std::string& fnString, 
			     const std::map<std::string, QCAD::SolverSubSolver>& subSolvers)
{
  std::vector<std::string> fnsAndTarget = string_split(fnString,'>',true);
  std::vector<std::string>::const_iterator it;  
  std::map<std::string,std::string> arrayRef;
  ArrayRef ar;
  int nParams = 0;    

  //Case: no function name given
  if( fnString.find_first_of('(') == string::npos ) { 
    fnName = "nop";
    arrayRef = string_parse_arrayref( fnString );
    ar.name = arrayRef["name"];

    const Epetra_Vector& solver_p = *((subSolvers.find(ar.name)->second).params_in->get_p(0));
    ar.indices = string_expand_compoundindex(arrayRef["index"], 0, solver_p.MyLength());
    params.push_back(ar);
    nParams += ar.indices.size();
  }

  //Case: function name given
  else {
    std::vector<std::string> fnAndParams = string_parse_function( fnString );
    fnName = fnAndParams[0];

    for(std::size_t i=1; i<fnAndParams.size(); i++) {

      //if contains [ treat as solver array reference, otherwise as a string param
      if( fnAndParams[i].find_first_of('[') == string::npos ) {
	ar.name = fnAndParams[i];  
	ar.indices.clear();
	params.push_back(ar);
	nParams += 1;
      }
      else {
	arrayRef = string_parse_arrayref( fnAndParams[i] );
	ar.name = arrayRef["name"];

	const Epetra_Vector& solver_p = *((subSolvers.find(ar.name)->second).params_in->get_p(0));
	ar.indices = string_expand_compoundindex(arrayRef["index"], 0, solver_p.MyLength());

	params.push_back(ar);
	nParams += ar.indices.size();
      }      
    }
  }

  // validate: check number of params and set numDoubles
  if( fnName == "min" || fnName == "max") {
    TEUCHOS_TEST_FOR_EXCEPT(nParams != 2);
    numDoubles = 1;
  }
  else if( fnName == "dist") {
    TEUCHOS_TEST_FOR_EXCEPT( !(nParams == 2 || nParams == 4 || nParams == 6));
    numDoubles = 1;
  }
  else if( fnName == "scale") {
    TEUCHOS_TEST_FOR_EXCEPT(nParams != 2);
    numDoubles = 1;
  }
  else if( fnName == "divide") {
    TEUCHOS_TEST_FOR_EXCEPT(nParams != 2);
    numDoubles = 1;
  }
  else if( fnName == "nop") {
    numDoubles = nParams; 
  }
  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
     "Unknown function " << fnName << " for QCAD solver response." << std::endl);

}


void QCAD::SolverResponseFn::fillSolverResponses(Epetra_Vector& g, int offset,
	   const std::map<std::string, QCAD::SolverSubSolver>& subSolvers) const
{
  //Collect parameters
  std::vector< double > pvals; //paramter values
  std::vector< ArrayRef >::const_iterator pit;
  for(pit = params.begin(); pit != params.end(); ++pit) {
    
    if( pit->indices.size() == 0 )
      pvals.push_back( atof(pit->name.c_str()) ); //if no indices, "array name" is a double param value

    else {
      std::vector<int>::const_iterator it;
      Teuchos::RCP<Epetra_Vector> sub_g = 
	(subSolvers.find(pit->name)->second).responses_out->get_g(0); // only use first g vector
    
      for(it = pit->indices.begin(); it != pit->indices.end(); ++it)
	pvals.push_back( (*sub_g)[ *it ]);
    }
  }
  
  std::size_t nParams = pvals.size();

  // minimum
  if( fnName == "min" ) {
    g[offset] = (pvals[0] < pvals[1]) ? pvals[0] : pvals[1];
  }

  // maximum
  else if(fnName == "max") {
    g[offset] = (pvals[0] >= pvals[1]) ? pvals[0] : pvals[1];
  }

  // distance btwn 1D, 2D or 3D points (params ordered as (x1,y1,z1,x2,y2,z2) )
  else if( fnName == "dist") { 
    if(nParams == 2) 
      g[offset] = abs(pvals[0]-pvals[1]);
    else if(nParams == 4) 
      g[offset] = sqrt( pow(pvals[0]-pvals[2],2) + pow(pvals[1]-pvals[3],2));
    else if(nParams == 6) 
      g[offset] = sqrt( pow(pvals[0]-pvals[3],2) + 
			pow(pvals[1]-pvals[4],2) + pow(pvals[2]-pvals[5],2) );
  }

  // multiplicative scaling
  else if( fnName == "scale") {
    g[offset] = pvals[0] * pvals[1];
  }

  // multiplicative scaling
  else if( fnName == "divide") {
    g[offset] = pvals[0] / pvals[1];
  }

  // no op (but can pass through multiple doubles)
  else if( fnName == "nop") {
    for(std::size_t i=0; i < nParams; i++)
      g[offset+i] = pvals[i];
  }

  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			  "Unknown function " << fnName << " for QCAD solver response." << std::endl);
}





// Helper Functions

QCAD::SolverSubSolver CreateSubSolver(const std::string xmlfilename, 
				const std::string& xmlPreprocessType, const Epetra_Comm& comm)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  QCAD::SolverSubSolver ret; //value to return

  const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(comm);

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "QCAD Solver: creating solver from input " << xmlfilename 
       << " after preprocessing as " << xmlPreprocessType << std::endl;
 
  //! Create solver factory, which reads xml input filen
  Albany::SolverFactory slvrfctry(xmlfilename.c_str(), mpiComm);
    
  //! Process input parameters based on solver type before creating solver & application
  Teuchos::ParameterList& appParams = slvrfctry.getParameters();

  preprocessParams(appParams, xmlPreprocessType);

  //! Create solver and application objects via solver factory
  RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(mpiComm);
  ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, appComm, appComm);

  ret.params_in = rcp(new EpetraExt::ModelEvaluator::InArgs);
  ret.responses_out = rcp(new EpetraExt::ModelEvaluator::OutArgs);  

  *(ret.params_in) = ret.model->createInArgs();
  *(ret.responses_out) = ret.model->createOutArgs();
  int num_p = ret.params_in->Np();     // Number of *vectors* of parameters
  int num_g = ret.responses_out->Ng(); // Number of *vectors* of responses
  RCP<Epetra_Vector> p1;
  RCP<Epetra_Vector> g1;
  
  if (num_p > 0)
    p1 = rcp(new Epetra_Vector(*(ret.model->get_p_init(0))));
  if (num_g > 1)
    g1 = rcp(new Epetra_Vector(*(ret.model->get_g_map(0))));
  RCP<Epetra_Vector> xfinal =
    rcp(new Epetra_Vector(*(ret.model->get_g_map(num_g-1)),true) );
  
  // Sensitivity Analysis stuff
  bool supportsSensitivities = false;
  RCP<Epetra_MultiVector> dgdp;
  
  if (num_p>0 && num_g>1) {
    supportsSensitivities =
      !ret.responses_out->supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, 0, 0).none();
    
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
  
  if (num_p > 0)  ret.params_in->set_p(0,p1);
  if (num_g > 1)  ret.responses_out->set_g(0,g1);
  ret.responses_out->set_g(num_g-1,xfinal);
  
  if (supportsSensitivities) ret.responses_out->set_DgDp(0,0,dgdp);
  
  return ret;
}


void preprocessParams(Teuchos::ParameterList& params, std::string preprocessType)
{
  if(preprocessType == "initial poisson") {
    //! Turn off schrodinger source
    params.sublist("Problem").sublist("Schrodinger Coupling").set<bool>("Schrodinger source in quantum blocks",false);

    //! Rename output file
    if (params.sublist("Discretization").isParameter("Exodus Output File Name"))
    {
      std::string exoName= "init" + params.sublist("Discretization").get<std::string>("Exodus Output File Name");
      params.sublist("Discretization").set("Exodus Output File Name", exoName);
    }
    else if (params.sublist("Discretization").isParameter("1D Output File Name"))
    {
      std::string exoName= "init" + params.sublist("Discretization").get<std::string>("1D Output File Name");
      params.sublist("Discretization").set("1D Output File Name", exoName);
    }
    
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			  "Unknown function Discretization Parameter" << std::endl);

    // temporary set Restart Index = 1 for initial poisson
    if (params.sublist("Discretization").isParameter("Restart Index"))
    {
      params.sublist("Discretization").set("Restart Index", 1); 
    }

  }

  else if(preprocessType == "dummy poisson") {
    //! Rename materials file
    std::string mtrlName= "dummy_" + params.sublist("Problem").get<std::string>("MaterialDB Filename");
    params.sublist("Problem").set("MaterialDB Filename", mtrlName);

    //! Rename output file
    std::string exoName= "dummy" + params.sublist("Discretization").get<std::string>("Exodus Output File Name");
    params.sublist("Discretization").set("Exodus Output File Name", exoName);
 
    //! Replace Dirichlet BCs and Parameters sublists with dummy versions
    params.sublist("Problem").sublist("Dirichlet BCs") = 
      params.sublist("Problem").sublist("Dummy Dirichlet BCs");
    params.sublist("Problem").sublist("Parameters") = 
      params.sublist("Problem").sublist("Dummy Parameters");
  }
}


void SolveModel(const QCAD::SolverSubSolver& ss)
{
  ss.model->evalModel( (*ss.params_in), (*ss.responses_out) );
}

void SolveModel(const QCAD::SolverSubSolver& ss, 
		Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates)
{
  if(pInitialStates != NULL) 
    ss.app->getStateMgr().importStateData( *pInitialStates );

  ss.model->evalModel( (*ss.params_in), (*ss.responses_out) );

  pFinalStates = &(ss.app->getStateMgr().getStateArrays());
}

void SolveModel(const QCAD::SolverSubSolver& ss, 
		Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates,
		Teuchos::RCP<Albany::EigendataStruct>& pInitialEData, 
		Teuchos::RCP<Albany::EigendataStruct>& pFinalEData)
{
  if(pInitialStates != NULL) 
    ss.app->getStateMgr().importStateData( *pInitialStates );

  if(pInitialEData != Teuchos::null) 
    ss.app->getStateMgr().setEigenData(pInitialEData);

  ss.model->evalModel( (*ss.params_in), (*ss.responses_out) );

  pFinalStates = &(ss.app->getStateMgr().getStateArrays());
  pFinalEData = ss.app->getStateMgr().getEigenData();
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
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 2 );
    
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
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++) {
      for(int qp=0; qp < dims[1]; qp++) {
	TEUCHOS_TEST_FOR_EXCEPT( isnan(src[ws](cell,qp)) );
        dest[ws][stateNameOfCopy](cell,qp) = src[ws](cell,qp);
      }
    }
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
  TEUCHOS_TEST_FOR_EXCEPT( numWorksets != (int)dest.size() );

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
  TEUCHOS_TEST_FOR_EXCEPT( numWorksets != (int)dest.size() );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][srcStateNameToSubtract].size();
    
    for(int i=0; i<totalSize; ++i)
      dest[ws][destStateNameToSubtractFrom][i] -= src[ws][srcStateNameToSubtract][i];
  }
}

double getMaxDifference(Albany::StateArrays& states, 
		      std::vector<Intrepid::FieldContainer<RealType> >& prevState,
		      std::string stateName)
{
  double maxDiff = 0.0;
  int numWorksets = states.size();
  std::vector<int> dims;

  TEUCHOS_TEST_FOR_EXCEPT( ! (numWorksets == (int)prevState.size()) );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    states[ws][stateName].dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++) 
    {
      for(int qp=0; qp < dims[1]; qp++) 
      {
        // std::cout << "prevState = " << prevState[ws](cell,qp) << std::endl;
        // std::cout << "currState = " << states[ws][stateName](cell,qp) << std::endl;
        if( fabs( states[ws][stateName](cell,qp) - prevState[ws](cell,qp) ) > maxDiff ) 
	  maxDiff = fabs( states[ws][stateName](cell,qp) - prevState[ws](cell,qp) );
      }
    }
  }
  return maxDiff;
}


void ResetEigensolverShift(const Teuchos::RCP<EpetraExt::ModelEvaluator>& Solver, double newShift, 
			   Teuchos::RCP<Teuchos::ParameterList>& eigList) 
{
  Teuchos::RCP<Piro::Epetra::LOCASolver> pels = Teuchos::rcp_dynamic_cast<Piro::Epetra::LOCASolver>(Solver);
  TEUCHOS_TEST_FOR_EXCEPT(pels == Teuchos::null);

  Teuchos::RCP<LOCA::Stepper> stepper =  pels->getLOCAStepperNonConst();
  const Teuchos::ParameterList& oldEigList = stepper->getList()->sublist("LOCA").sublist(
							 "Stepper").sublist("Eigensolver");

  eigList = Teuchos::rcp(new Teuchos::ParameterList(oldEigList));
  eigList->set("Shift",newShift);

  //cout << " OLD Eigensolver list  " << oldEigList << endl;
  //cout << " NEW Eigensolver list  " << *eigList << endl;
  std::cout << "QCAD Solver: setting eigensolver shift = " 
	    << std::setprecision(5) << newShift << std::endl;

  stepper->eigensolverReset(eigList);
}


double GetEigensolverShift(const QCAD::SolverSubSolver& ss, 
			   int minPotentialResponseIndex)
{
  int Ng = ss.responses_out->Ng();
  TEUCHOS_TEST_FOR_EXCEPT( Ng <= 0 );

  Teuchos::RCP<Epetra_Vector> gVector = ss.responses_out->get_g(0);
  
  TEUCHOS_TEST_FOR_EXCEPT( gVector->GlobalLength() <= minPotentialResponseIndex);
  double minVal = (*gVector)[minPotentialResponseIndex];

  //set shift to be slightly (5% of range) below minimum value
  // double shift = -(minVal - 0.05*(maxVal-minVal)); //minus sign b/c negative eigenvalue convention
  
  double shift = -minVal*1.1;  // 10% below minimum value
  return shift;
}
  



// String functions
std::vector<std::string> string_split(const std::string& s, char delim, bool bProtect)
{
  int last = 0;
  int parenLevel=0, bracketLevel=0, braceLevel=0;

  std::vector<string> ret(1);
  for(std::size_t i=0; i<s.size(); i++) {
    if(s[i] == delim && parenLevel==0 && bracketLevel==0 && braceLevel==0) {
      ret.push_back(""); 
      last++; 
    }
    else ret[last] += s[i];

    if(bProtect) {
      if(     s[i] == '(') parenLevel++;
      else if(s[i] == ')') parenLevel--;
      else if(s[i] == '[') bracketLevel++;
      else if(s[i] == ']') bracketLevel--;
      else if(s[i] == '{') braceLevel++;
      else if(s[i] == '}') braceLevel--;
    }
  }
  return ret;
}

std::string string_remove_whitespace(const std::string& s)
{
  std::string ret;
  for(std::size_t i=0; i<s.size(); i++) {
    if(s[i] == ' ') continue;
    ret += s[i];
  }
  return ret;
}

//given s="MyFunction(arg1,arg2,arg3)", 
// returns vector { "MyFunction", "arg1", "arg2", "arg3" }
std::vector<std::string> string_parse_function(const std::string& s)
{
  std::vector<string> ret;
  std::string fnName, fnArgString;
  std::size_t firstOpenParen, lastCloseParen;

  firstOpenParen = s.find_first_of('(');
  lastCloseParen = s.find_last_of(')');

  if(firstOpenParen == string::npos || lastCloseParen == string::npos) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Malformed function string: " << s << std::endl);
  }

  fnName = s.substr(0,firstOpenParen);
  fnArgString = s.substr(firstOpenParen+1,lastCloseParen-firstOpenParen-1);

  ret = string_split(fnArgString,',',true);
  ret.insert(ret.begin(),fnName);  //place function name at beginning of vector returned
  return ret;
}

std::map<std::string,std::string> string_parse_arrayref(const std::string& s)
{
  std::map<std::string,std::string> ret;
  std::string arName, indexString;
  std::size_t firstOpenBracket, lastCloseBracket;

  firstOpenBracket = s.find_first_of('[');
  lastCloseBracket = s.find_last_of(']');

  if(firstOpenBracket == string::npos || lastCloseBracket == string::npos) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Malformed array string: " << s << std::endl);
  }

  arName = s.substr(0,firstOpenBracket);
  indexString = s.substr(firstOpenBracket+1,lastCloseBracket-firstOpenBracket-1);

  ret["name"] = arName;
  ret["index"] = indexString;
  return ret;
}

std::vector<int> string_expand_compoundindex(const std::string& indexStr, int min_index, int max_index)
{
  std::vector<int> ret;
  std::vector<std::string> simpleRanges = string_split(indexStr,',',true);
  std::vector<std::string>::const_iterator it;

  for(it = simpleRanges.begin(); it != simpleRanges.end(); it++) {
    std::vector<std::string> endpts = string_split( (*it), ':', true);
    if(endpts.size() == 1)
      ret.push_back( atoi(endpts[0].c_str()) );
    else if(endpts.size() == 2) {
      int a=min_index ,b=max_index;
      if(endpts[0] != "") a = atoi(endpts[0].c_str());
      if(endpts[1] != "") b = atoi(endpts[1].c_str());
      for(int i=a; i<b; i++) ret.push_back(i);
    }
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Malformed array index: " << indexStr << std::endl);
  }
  return ret;
}
