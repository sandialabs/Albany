//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Solver.hpp"
#include "ATO_OptimizationProblem.hpp"

/* GAH FIXME - Silence warning:
TRILINOS_DIR/../../../include/pecos_global_defs.hpp:17:0: warning: 
        "BOOST_MATH_PROMOTE_DOUBLE_POLICY" redefined [enabled by default]
Please remove when issue is resolved
*/
#undef BOOST_MATH_PROMOTE_DOUBLE_POLICY

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"


//#define ATO_FILTER_ON
#undef ATO_FILTER_ON
// TEV: Following is for debugging filter operator.
#ifdef ATO_FILTER_ON
#include "EpetraExt_RowMatrixOut.h"
#endif //ATO_FILTER_ON

/******************************************************************************/
ATO::Solver::
Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
       const Teuchos::RCP<const Epetra_Comm>& comm,
       const Teuchos::RCP<const Epetra_Vector>& initial_guess)
: _solverComm(comm), _mainAppParams(appParams), filterOperator(Teuchos::null)
/******************************************************************************/
{
  zeroSet();


  ///*** PROCESS TOP LEVEL PROBLEM ***///

  // Validate Problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  _numPhysics = problemParams.get<int>("Number of Subproblems", 1);
  problemParams.validateParameters(*getValidProblemParameters(),0);

  // Parse and create optimizer
  Teuchos::ParameterList& optimizerParams = 
    problemParams.get<Teuchos::ParameterList>("Topological Optimization");
  ATO::OptimizerFactory optimizerFactory;
  _optimizer = optimizerFactory.create(optimizerParams);
  _optimizer->SetInterface(this);
  _optimizer->SetCommunicator(comm);

  // Parse and create aggregator
  Teuchos::ParameterList& aggregatorParams = 
    problemParams.get<Teuchos::ParameterList>("Objective Aggregator");
  ATO::AggregatorFactory aggregatorFactory;
  _aggregator = aggregatorFactory.create(aggregatorParams);

  // Parse topology info
  Teuchos::ParameterList& topoParams = problemParams.get<Teuchos::ParameterList>("Topology");
  _topoCentering = topoParams.get<std::string>("Centering");
  _topoName = topoParams.get<std::string>("Topology Name");
  _topoFilterRadius = topoParams.get<double>("Filter Radius",0.0);

  // Get and set the default Piro parameters from a file, if given
  std::string piroFilename  = problemParams.get<std::string>("Piro Defaults Filename", "");
  if(piroFilename.length() > 0) {
    const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(*comm);
    Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mpiComm);
    Teuchos::RCP<Teuchos::ParameterList> defaultPiroParams = 
      Teuchos::createParameterList("Default Piro Parameters");
    Teuchos::updateParametersFromXmlFileAndBroadcast(piroFilename, defaultPiroParams.ptr(), *tcomm);
    Teuchos::ParameterList& piroList = appParams->sublist("Piro", false);
    piroList.setParametersNotAlreadySet(*defaultPiroParams);
  }
  
  // set verbosity
  _is_verbose = (comm->MyPID() == 0) && problemParams.get<bool>("Verbose Output", false);




  ///*** PROCESS SUBPROBLEM(S) ***///
   
  _subProblemAppParams.resize(_numPhysics);
  _subProblems.resize(_numPhysics);
  for(int i=0; i<_numPhysics; i++){

    _subProblemAppParams[i] = createInputFile(appParams, i);
    _subProblems[i] = CreateSubSolver( _subProblemAppParams[i], *_solverComm);

    // ensure that all subproblems are topology based (i.e., optimizable)
    Teuchos::RCP<Albany::AbstractProblem> problem = _subProblems[i].app->getProblem();
    ATO::OptimizationProblem* atoProblem = 
      dynamic_cast<ATO::OptimizationProblem*>(problem.get());
    TEUCHOS_TEST_FOR_EXCEPTION( 
      atoProblem == NULL, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Requested subproblem does not support topologies." << std::endl);
  }

 
  // pass subProblems to the aggregator
  _aggregator->SetInputVariables(_subProblems);
  


  // store a pointer to the first problem as an ATO::OptimizationProblem for callbacks
  Teuchos::RCP<Albany::AbstractProblem> problem = _subProblems[0].app->getProblem();
  _atoProblem = dynamic_cast<ATO::OptimizationProblem*>(problem.get());
  _atoProblem->setDiscretization(_subProblems[0].app->getDiscretization());
  _atoProblem->setCommunicator(comm);
  _atoProblem->InitTopOpt();
  


  // get solution map from first subproblem
  const SolverSubSolver& sub = _subProblems[0];
  Teuchos::RCP<const Epetra_Map> sub_x_map = sub.app->getMap();
  TEUCHOS_TEST_FOR_EXCEPT( sub_x_map == Teuchos::null );
  _epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));

  // initialize/build the filter operator. this is built once.
#ifdef ATO_FILTER_ON
  buildFilterOperator(_subProblems[0].app);
#endif //ATO_FILTER_ON

  if( _topoCentering == "Node" ){
    // create overlap topo vector for output purposes
    Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
    Teuchos::RCP<const Epetra_BlockMap> 
      overlapNodeMap = stateMgr.getNodalDataBlock()->getOverlapMap();
    Teuchos::RCP<const Epetra_BlockMap> 
      localNodeMap = stateMgr.getNodalDataBlock()->getLocalMap();
    overlapTopoVec = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
    overlapdfdpVec = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
    dfdpVec  = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
    topoVec  = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
                                              
                                              //* target *//   //* source *//
    importer = Teuchos::rcp(new Epetra_Import(*overlapNodeMap, *localNodeMap));


    // create exporter (for integration type operations):
                                              //* source *//   //* target *//
    exporter = Teuchos::rcp(new Epetra_Export(*overlapNodeMap, *localNodeMap));
  }


#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif
}


/******************************************************************************/
void
ATO::Solver::zeroSet()
/******************************************************************************/
{
  // set parameters and responses
  _num_parameters = 0; //TEV: assume no parameters or responses for now...
  _num_responses  = 0; //TEV: assume no parameters or responses for now...
}

  
/******************************************************************************/
void
ATO::Solver::evalModel(const InArgs& inArgs,
                       const OutArgs& outArgs ) const
/******************************************************************************/
{


  if(_is_verbose){
    Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
    *out << "*** Performing Topology Optimization Loop ***" << std::endl;
  }

  _optimizer->Initialize();

  _optimizer->Optimize();
 
}



/******************************************************************************/
///*************** SOLVER - OPTIMIZER INTERFACE FUNCTIONS *******************///
/******************************************************************************/


/******************************************************************************/
void
ATO::Solver::ComputeObjective(const double* p, double& f, double* dfdp)
/******************************************************************************/
{
  for(int i=0; i<_numPhysics; i++){
    // copy data from p into each stateManager
    Albany::StateManager& stateMgr = _subProblems[i].app->getStateMgr();
    copyTopologyIntoStateMgr( p, stateMgr );

    // enforce PDE constraints
    _subProblems[i].model->evalModel((*_subProblems[i].params_in),
                                    (*_subProblems[i].responses_out));
  }

  _aggregator->Evaluate();
  
  // copy objective (f) and first derivative wrt the topology (dfdp) out 
  // of stateManager
  copyObjectiveFromStateMgr( f, dfdp );
  
}

/******************************************************************************/
void
ATO::Solver::copyTopologyIntoStateMgr( const double* p, Albany::StateManager& stateMgr )
/******************************************************************************/
{

  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  int numWorksets = dest.size();

  if( _topoCentering == "Element" ){
    int wsOffset = 0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topoName];
      int wsSize = wsTopo.size();
      for(int i=0; i<wsSize; i++)
        wsTopo(i) = p[wsOffset+i];
      wsOffset += wsSize;
    }
  } else 
  if( _topoCentering == "Node" ){

    // communicate boundary info
    int numLocalNodes = topoVec->MyLength();
    double* ltopo; topoVec->ExtractView(&ltopo);
    std::memcpy((void*)ltopo, (void*)p, numLocalNodes*sizeof(double));
    overlapTopoVec->Import(*topoVec, *importer, Insert);
    double* otopo; overlapTopoVec->ExtractView(&otopo);

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type&
      wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();
    Teuchos::RCP<const Epetra_BlockMap> 
      overlapNodeMap = stateMgr.getNodalDataBlock()->getOverlapMap();


    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topoName];
      int numCells = wsTopo.dimension(0);
      int numNodes = wsTopo.dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMap->LID(gid);
          wsTopo(cell,node) = otopo[lid];
        }
    }

    // save topology to nodal data for output sake
    Teuchos::RCP<Albany::NodeFieldContainer> 
      nodeContainer = stateMgr.getNodalDataBlock()->getNodeContainer();

    std::string nodal_topoName = _topoName+"_node";
    (*nodeContainer)[nodal_topoName]->saveField(overlapTopoVec,/*offset=*/0);

  }
}

/******************************************************************************/
void
ATO::Solver::copyObjectiveFromStateMgr( double& f, double* dfdp )
/******************************************************************************/
{
  // f and dfdp are stored in subProblem[0]
  Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& src = stateArrays.elemStateArrays;
  int numWorksets = src.size();

  std::string objName = _aggregator->getOutputObjectiveName();
  std::string derName = _aggregator->getOutputDerivativeName();

  Albany::MDArray& fSrc = src[0][objName];
  f = fSrc(0);
  fSrc(0) = 0.0;

  if( _topoCentering == "Element" ){
    int wsOffset = 0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& dfdpSrc = src[ws][derName];
      int wsSize = dfdpSrc.size();
      for(int i=0; i<wsSize; i++)
        dfdp[wsOffset+i] = dfdpSrc(i);
      wsOffset += wsSize;
    }
  } else
  if( _topoCentering == "Node" ){

    Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type&
      wsElNodeID = disc->getWsElNodeID();

    Teuchos::RCP<const Epetra_BlockMap> 
      overlapNodeMap = stateMgr.getNodalDataBlock()->getOverlapMap();

    dfdpVec->PutScalar(0.0);
    overlapdfdpVec->PutScalar(0.0);
    double* odfdp; overlapdfdpVec->ExtractView(&odfdp);

    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& dfdpSrc = src[ws][derName];
      int numCells = dfdpSrc.dimension(0);
      int numNodes = dfdpSrc.dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMap->LID(gid);
          odfdp[lid] += dfdpSrc(cell,node);
        }
    }
    // if no smoother is being used, values will not yet be consistent
    // accross processors, so communicate boundary info:
    bool smoother = false;
    if( !smoother ){
      dfdpVec->Export(*overlapdfdpVec, *exporter, Add);

      int numLocalNodes = dfdpVec->MyLength();
      double* lvec; dfdpVec->ExtractView(&lvec);
      std::memcpy((void*)dfdp, (void*)lvec, numLocalNodes*sizeof(double));

    }
  }
}
/******************************************************************************/
void
ATO::Solver::ComputeVolume(double& v)
/******************************************************************************/
{
  return _atoProblem->ComputeVolume(v);
}


/******************************************************************************/
void
ATO::Solver::ComputeVolume(const double* p, double& v, double* dvdp)
/******************************************************************************/
{
  return _atoProblem->ComputeVolume(p, v, dvdp);
}

/******************************************************************************/
void
ATO::Solver::ComputeConstraint(double* p, double& c, double* dcdp)
/******************************************************************************/
{
}

/******************************************************************************/
int
ATO::Solver::GetNumOptDofs()
/******************************************************************************/
{
  if( _topoCentering == "Element" ){
    Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
    Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
    Albany::StateArrayVec& dest = stateArrays.elemStateArrays;

    int numWorksets = dest.size();

    int numTotalElems = 0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topoName];
      int wsSize = wsTopo.size();
      numTotalElems += wsSize;
    }
    return numTotalElems;
    
  } else
  if( _topoCentering == "Node" ){
    return _subProblems[0].app->getDiscretization()->getNodeMap()->NumMyElements();
  }
}

/******************************************************************************/
///*********************** SETUP AND UTILITY FUNCTIONS **********************///
/******************************************************************************/


/******************************************************************************/
ATO::SolverSubSolver
ATO::Solver::CreateSubSolver( const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                              const Epetra_Comm& comm,
                              const Teuchos::RCP<const Epetra_Vector>& initial_guess) const
/******************************************************************************/
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  ATO::SolverSubSolver ret; //value to return

  const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(comm);

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "ATO Solver creating solver from " << appParams->name()
       << " parameter list" << std::endl;

  //! Create solver factory, which reads xml input filen
  Albany::SolverFactory slvrfctry(appParams, mpiComm);

  //! Create solver and application objects via solver factory
  RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(mpiComm);
  ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, appComm, appComm, initial_guess);


  ret.params_in = rcp(new EpetraExt::ModelEvaluator::InArgs);
  ret.responses_out = rcp(new EpetraExt::ModelEvaluator::OutArgs);

  *(ret.params_in) = ret.model->createInArgs();
  *(ret.responses_out) = ret.model->createOutArgs();
  int ss_num_p = ret.params_in->Np();     // Number of *vectors* of parameters
  int ss_num_g = ret.responses_out->Ng(); // Number of *vectors* of responses
  RCP<Epetra_Vector> p1;
  RCP<Epetra_Vector> g1;

  if (ss_num_p > 0)
    p1 = rcp(new Epetra_Vector(*(ret.model->get_p_init(0))));
  if (ss_num_g > 1)
    g1 = rcp(new Epetra_Vector(*(ret.model->get_g_map(0))));
  RCP<Epetra_Vector> xfinal =
    rcp(new Epetra_Vector(*(ret.model->get_g_map(ss_num_g-1)),true) );

  // Sensitivity Analysis stuff
  bool supportsSensitivities = false;
  RCP<Epetra_MultiVector> dgdp;

  if (ss_num_p>0 && ss_num_g>1) {
    supportsSensitivities =
      !ret.responses_out->supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, 0, 0).none();

    if (supportsSensitivities) {
      if (p1->GlobalLength() > 0)
        dgdp = rcp(new Epetra_MultiVector(g1->Map(), p1->GlobalLength() ));
      else
        supportsSensitivities = false;
    }
  }

  if (ss_num_p > 0)  ret.params_in->set_p(0,p1);
  if (ss_num_g > 1)  ret.responses_out->set_g(0,g1);
  ret.responses_out->set_g(ss_num_g-1,xfinal);

  if (supportsSensitivities) ret.responses_out->set_DgDp(0,0,dgdp);

  return ret;
}

/******************************************************************************/
Teuchos::RCP<Teuchos::ParameterList> 
ATO::Solver::createInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams, int physIndex) const
/******************************************************************************/
{   


  ///*** CREATE INPUT FILE FOR SUBPROBLEM: ***///
  

  // Get physics (pde) problem sublist, i.e., Physics Problem N, where N = physIndex.
  std::stringstream physStream;
  physStream << "Physics Problem " << physIndex;
  Teuchos::ParameterList& physics_subList = appParams->sublist("Problem").sublist(physStream.str(), false);

  // Create input parameter list for physics app which mimics a separate input file
  std::stringstream appStream;
  appStream << "Parameters for Subapplication " << physIndex;
  Teuchos::RCP<Teuchos::ParameterList> physics_appParams = Teuchos::createParameterList(appStream.str());

  // get reference to Problem ParameterList in new input file and initialize it 
  // from Parameters in Physics Problem N.
  Teuchos::ParameterList& physics_probParams = physics_appParams->sublist("Problem",false);
  physics_probParams.setParameters(physics_subList);

  // Add topology information
  Teuchos::ParameterList& topoParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Topology");
  physics_probParams.set<Teuchos::ParameterList>("Topology",topoParams);

  // Add aggregator information
  Teuchos::ParameterList& aggParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Objective Aggregator");
  physics_probParams.set<Teuchos::ParameterList>("Objective Aggregator",aggParams);

  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& physics_discList = physics_appParams->sublist("Discretization", false);
  physics_discList.setParameters(discList);

  // Piro sublist processing
  physics_appParams->set("Piro",appParams->sublist("Piro"));



  ///*** VERIFY SUBPROBLEM: ***///


  // extract physics and dimension of the subproblem
  Teuchos::ParameterList& subProblemParams = appParams->sublist("Problem").sublist(physStream.str());
  std::string problemName = subProblemParams.get<std::string>("Name");
  // "xD" where x = 1, 2, or 3
  std::string problemDimStr = problemName.substr( problemName.length()-2 );
  //remove " xD" where x = 1, 2, or 3
  std::string problemNameBase = problemName.substr( 0, problemName.length()-3 );
  
  //// check dimensions
  int numDimensions = 0;
  if(problemDimStr == "1D") numDimensions = 1;
  else if(problemDimStr == "2D") numDimensions = 2;
  else if(problemDimStr == "3D") numDimensions = 3;
  else TEUCHOS_TEST_FOR_EXCEPTION (
         true, Teuchos::Exceptions::InvalidParameter, std::endl 
         << "Error!  Cannot extract dimension from problem name: " << problemName << std::endl);
  TEUCHOS_TEST_FOR_EXCEPTION (
    numDimensions == 1, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  Topology optimization is not avaliable in 1D." << std::endl);

  //// See if requested physics work with ATO (add your physics here)
  std::vector<std::string> ATOablePhysics;
  ATOablePhysics.push_back( "LinearElasticity" );
  
  std::vector<std::string>::iterator it;
  it = std::find(ATOablePhysics.begin(), ATOablePhysics.end(), problemNameBase);
  TEUCHOS_TEST_FOR_EXCEPTION (
    it == ATOablePhysics.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  Invalid problem base name: " << problemNameBase << std::endl);
  
  
  return physics_appParams;

}

/******************************************************************************/
Teuchos::RCP<const Teuchos::ParameterList>
ATO::Solver::getValidProblemParameters() const
/******************************************************************************/
{

  Teuchos::RCP<Teuchos::ParameterList> validPL = 
    Teuchos::createParameterList("ValidTopologicalOptimizationProblemParams");

  // Basic set-up
  validPL->set<int>("Number of Subproblems", 1, "Number of PDE constraint problems");
  validPL->set<bool>("Verbose Output", false, "Enable detailed output mode");
  validPL->set<std::string>("Name", "", "String to designate Problem");

  // Specify physics problem(s)
  for(int i=0; i<_numPhysics; i++){
    std::stringstream physStream; physStream << "Physics Problem " << i;
    validPL->sublist(physStream.str(), false, "");
  }

  // Specify aggregator
  validPL->sublist("Objective Aggregator", false, "");

  // Specify optimizer
  validPL->sublist("Topological Optimization", false, "");

  // Specify responses
  validPL->sublist("Topology", false, "");

  // Physics solver options
  validPL->set<std::string>(
       "Piro Defaults Filename", "", 
       "An xml file containing a default Piro parameterlist and its sublists");

  // Candidate for deprecation.
  validPL->set<std::string>(
       "Solution Method", "Steady", 
       "Flag for Steady, Transient, or Continuation");

  return validPL;
}





/******************************************************************************/
///*************                   BOILERPLATE                  *************///
/******************************************************************************/



/******************************************************************************/
ATO::Solver::~Solver() { }
/******************************************************************************/


/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_x_map() const
/******************************************************************************/
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_f_map() const
/******************************************************************************/
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

/******************************************************************************/
EpetraExt::ModelEvaluator::InArgs 
ATO::Solver::createInArgs() const
/******************************************************************************/
{
  EpetraExt::ModelEvaluator::InArgsSetup inArgs;
  inArgs.setModelEvalDescription("ATO Solver Model Evaluator Description");
  inArgs.set_Np(_num_parameters);
  return inArgs;
}

/******************************************************************************/
EpetraExt::ModelEvaluator::OutArgs 
ATO::Solver::createOutArgs() const
/******************************************************************************/
{
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("ATO Solver Multipurpose Model Evaluator");
  outArgs.set_Np_Ng(_num_parameters, _num_responses+1);  //TODO: is the +1 necessary still??
  return outArgs;
}

/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_g_map(int j) const
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(j > _num_responses || j < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in ATO::Solver::get_g_map():  " <<
                     "Invalid response index j = " <<
                     j << std::endl);
  //TEV: Hardwired for now
  int _num_responses = 0;
  if      (j <  _num_responses) return _epetra_response_map;  //no index because num_g == 1 so j must be zero
  else if (j == _num_responses) return _epetra_x_map;
  return Teuchos::null;
}

/******************************************************************************/
ATO::SolverSubSolverData
ATO::Solver::CreateSubSolverData(const ATO::SolverSubSolver& sub) const
/******************************************************************************/
{
  ATO::SolverSubSolverData ret;
  if( sub.params_in->Np() > 0 && sub.responses_out->Ng() > 0 ) {
    ret.deriv_support = sub.model->createOutArgs().supports(OUT_ARG_DgDp, 0, 0);
  }
  else ret.deriv_support = EpetraExt::ModelEvaluator::DerivativeSupport();

  ret.Np = sub.params_in->Np();
  ret.pLength = std::vector<int>(ret.Np);
  for(int i=0; i<ret.Np; i++) {
    Teuchos::RCP<const Epetra_Vector> solver_p = sub.params_in->get_p(i);
    if(solver_p != Teuchos::null) ret.pLength[i] = solver_p->MyLength();  //uses local length (need to modify to work with distributed params)
    else ret.pLength[i] = 0;
  }

  ret.Ng = sub.responses_out->Ng();
  ret.gLength = std::vector<int>(ret.Ng);
  for(int i=0; i<ret.Ng; i++) {
    Teuchos::RCP<const Epetra_Vector> solver_g = sub.responses_out->get_g(i);
    if(solver_g != Teuchos::null) ret.gLength[i] = solver_g->MyLength(); //uses local length (need to modify to work with distributed responses)
    else ret.gLength[i] = 0;
  }

  if(ret.Np > 0) {
    Teuchos::RCP<const Epetra_Vector> p_init =
      sub.model->get_p_init(0); //only first p vector used - in the future could make ret.p_init an array of Np vectors
    if(p_init != Teuchos::null) ret.p_init = Teuchos::rcp(new const Epetra_Vector(*p_init)); //copy
    else ret.p_init = Teuchos::null;
  }
  else ret.p_init = Teuchos::null;

  return ret;
}

/******************************************************************************/
void
ATO::Solver::buildFilterOperator(const Teuchos::RCP<Albany::Application> app)
/******************************************************************************/
{


  if(_topoCentering == "Node") {

    Teuchos::RCP<Adapt::NodalDataBlock> node_data = app->getStateMgr().getNodalDataBlock();

    // create exporter
    Teuchos::RCP<const Epetra_Comm> comm             = app->getComm();
    Teuchos::RCP<const Epetra_BlockMap>  local_node_blockmap   = node_data->getLocalMap();
    Teuchos::RCP<const Epetra_BlockMap>  overlap_node_blockmap = node_data->getOverlapMap();
  
    // construct simple maps for node ids. 
    int num_global_elements = local_node_blockmap->NumGlobalElements();
    int num_my_elements     = local_node_blockmap->NumMyElements();
    int *global_node_ids    = new int[num_my_elements]; 
    local_node_blockmap->MyGlobalElements(global_node_ids);
    Epetra_Map local_node_map(num_global_elements,num_my_elements,global_node_ids,0,*comm);
    delete [] global_node_ids;
  
    num_global_elements = overlap_node_blockmap->NumGlobalElements();
    num_my_elements     = overlap_node_blockmap->NumMyElements();
    global_node_ids    = new int[num_my_elements]; 
    overlap_node_blockmap->MyGlobalElements(global_node_ids);
    Epetra_Map overlap_node_map(num_global_elements,num_my_elements,global_node_ids,0,*comm);
    delete [] global_node_ids;
  
    Epetra_Export exporter = Epetra_Export(overlap_node_map, local_node_map);
  
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type&
          wsElNodeID = app->getDiscretization()->getWsElNodeID();
  
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
      coords = app->getDiscretization()->getCoords();
  
    std::map< int, std::set<int> > neighbors;
  
    double filter_radius_sqrd = _topoFilterRadius*_topoFilterRadius;
    // awful n^2 search... all against all
    size_t dimension   = app->getDiscretization()->getNumDim();
    double *home_coord = new double[dimension];
    size_t num_worksets = coords.size();
    for (size_t home_ws=0; home_ws<num_worksets; home_ws++) {
      int home_num_cells = coords[home_ws].size();
      for (int home_cell=0; home_cell<home_num_cells; home_cell++) {
        size_t num_nodes = coords[home_ws][home_cell].size();
        for (int home_node=0; home_node<num_nodes; home_node++) {
          int home_node_gid = wsElNodeID[home_ws][home_cell][home_node];
          if(neighbors.find(home_node_gid)==neighbors.end()) {  // if this node was already accessed just skip
            for (int dim=0; dim<dimension; dim++)  {
              home_coord[dim] = coords[home_ws][home_cell][home_node][dim];
            }
            std::set<int> my_neighbors;
            for (size_t trial_ws=0; trial_ws<num_worksets; trial_ws++) {
              int trial_num_cells = coords[trial_ws].size();
              for (int trial_cell=0; trial_cell<trial_num_cells; trial_cell++) {
                size_t trial_num_nodes = coords[trial_ws][trial_cell].size();
                for (int trial_node=0; trial_node<trial_num_nodes; trial_node++) {
                  double tmp;
                  double delta_norm_sqr = 0.;
                  for (int dim=0; dim<dimension; dim++)  { //individual coordinates
                    tmp = home_coord[dim]-coords[trial_ws][trial_cell][trial_node][dim];
                    delta_norm_sqr += tmp*tmp;
                  }
                  if(delta_norm_sqr<=filter_radius_sqrd) {
                    int trial_node_gid = wsElNodeID[trial_ws][trial_cell][trial_node];
                    my_neighbors.insert(trial_node_gid);
                  }
                }
              }
            }
            neighbors.insert( std::pair<int,std::set<int> >(home_node_gid,my_neighbors) );
          }
        }
      }
    }
    delete [] home_coord;
  
    // now build filter operator
    int numnonzeros = 0;
    filterOperator = Teuchos::rcp(new Epetra_CrsMatrix(Copy,overlap_node_map,numnonzeros));
    for (std::map<int,std::set<int> >::iterator it=neighbors.begin(); it!=neighbors.end(); ++it) { 
      int home_node_gid = it->first;
      std::set<int> connected_nodes = it->second;
      for (std::set<int>::iterator set_it=connected_nodes.begin(); set_it!=connected_nodes.end(); ++set_it) {
         double value = 1.;
         int neighbor_node_gid = *set_it;
         filterOperator->InsertGlobalValues(home_node_gid,1,&value,&neighbor_node_gid);
      }
    }
  
    filterOperator->FillComplete();

// TEV: this is debugging code for examining filteroperator. Should be commented/ifdef'd out for
// ... final production code.
#ifdef ATO_FILTER_ON
    EpetraExt::RowMatrixToMatlabFile("ato_filter_operator.m",*filterOperator);
#endif //ATO_FILTER_ON
    
  /*
    int num_remote_lids = overlap_exporter.NumRemoteIDs();
    int *remote_lids = new int[num_remote_lids];
    remote_lids = overlap_exporter.RemoteLIDs();
    delete [] remote_lids;
  */
  } else {
    // Element centered filter
  }
  
  return;

}

#ifdef ATO_FILTER_ON
#undef ATO_FILTER_ON
#endif //ATO_FILTER_ON

