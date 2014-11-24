//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Solver.hpp"
#include "ATO_OptimizationProblem.hpp"
#include "ATO_TopoTools.hpp"

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
#include "Adapt_NodalDataBlock.hpp"
#include "Petra_Converters.hpp"

#define ATO_FILTER_ON
//#undef ATO_FILTER_ON

// TEV: Following is for debugging filter operator.
#ifdef ATO_FILTER_ON
#include "EpetraExt_RowMatrixOut.h"
#endif //ATO_FILTER_ON

MPI_Datatype MPI_GlobalPoint;

bool ATO::operator< (ATO::GlobalPoint const & a, ATO::GlobalPoint const & b){return a.gid < b.gid;}
ATO::GlobalPoint::GlobalPoint(){coords[0]=0.0; coords[1]=0.0; coords[2]=0.0;}


/******************************************************************************/
ATO::Solver::
Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
       const Teuchos::RCP<const Epetra_Comm>& comm,
       const Teuchos::RCP<const Epetra_Vector>& initial_guess)
: _solverComm(comm), _mainAppParams(appParams)
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
  ATO::TopologyFactory topoFactory;
  _topology = topoFactory.create(topoParams);

  // Parse filters
  if( problemParams.isType<Teuchos::ParameterList>("Spatial Filters")){
    Teuchos::ParameterList& filtersParams = problemParams.get<Teuchos::ParameterList>("Spatial Filters");
    int nFilters = filtersParams.get<int>("Number of Filters");
    for(int ifltr=0; ifltr<nFilters; ifltr++){
      std::stringstream filterStream;
      filterStream << "Filter " << ifltr;
      Teuchos::ParameterList& filterParams = filtersParams.get<Teuchos::ParameterList>(filterStream.str());
      Teuchos::RCP<ATO::SpatialFilter> newFilter = Teuchos::rcp( new ATO::SpatialFilter(filterParams) );
      filters.push_back(newFilter);
    }
  }
  
  int topologyFilterIndex = _topology->SpatialFilterIndex();
  if( topologyFilterIndex >= 0 ){
    TEUCHOS_TEST_FOR_EXCEPTION( topologyFilterIndex >= filters.size(),
      Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Spatial filter " << topologyFilterIndex << "requested but not defined." << std::endl);
    _topologyFilter = filters[topologyFilterIndex];
  }

  int topologyOutputFilter = _topology->TopologyOutputFilter();
  if( topologyOutputFilter >= 0 ){
    TEUCHOS_TEST_FOR_EXCEPTION( topologyOutputFilter >= filters.size(),
      Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Spatial filter " << topologyFilterIndex << "requested but not defined." << std::endl);
    _postTopologyFilter = filters[topologyOutputFilter];
  }

  int derivativeFilterIndex = aggregatorParams.get<int>("Spatial Filter", -1);
  if( derivativeFilterIndex >= 0 ){
    TEUCHOS_TEST_FOR_EXCEPTION( derivativeFilterIndex >= filters.size(),
      Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Spatial filter " << derivativeFilterIndex << "requested but not defined." << std::endl);
    _derivativeFilter = filters[derivativeFilterIndex];
  }

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
  _aggregator->SetCommunicator(comm);
  


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

  if( _topology->getCentering() == "Node" ){
    Teuchos::RCP<Albany::Application> app = _subProblems[0].app;
    Albany::StateManager& stateMgr = app->getStateMgr();

    // construct epetra maps for node ids. 
    Teuchos::RCP<const Epetra_Comm> comm = Albany::createEpetraCommFromTeuchosComm(_subProblems[0].app->getComm());
    Teuchos::RCP<const Epetra_BlockMap>
      local_node_blockmap   = stateMgr.getNodalDataBlock()->getLocalMapE();
    int num_global_elements = local_node_blockmap->NumGlobalElements();
    int num_my_elements     = local_node_blockmap->NumMyElements();
    int *global_node_ids    = new int[num_my_elements]; 
    local_node_blockmap->MyGlobalElements(global_node_ids);
    localNodeMap = Teuchos::rcp(new Epetra_Map(num_global_elements,num_my_elements,global_node_ids,0,*comm));
    delete [] global_node_ids;

    Teuchos::RCP<const Epetra_BlockMap>
      overlap_node_blockmap = stateMgr.getNodalDataBlock()->getOverlapMapE();
    num_global_elements = overlap_node_blockmap->NumGlobalElements();
    num_my_elements     = overlap_node_blockmap->NumMyElements();
    global_node_ids     = new int[num_my_elements]; 
    overlap_node_blockmap->MyGlobalElements(global_node_ids);
    overlapNodeMap = Teuchos::rcp(new Epetra_Map(num_global_elements,num_my_elements,global_node_ids,0,*comm));
    delete [] global_node_ids;

    if(_postTopologyFilter != Teuchos::null ){
      filteredOTopoVec = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
      filteredTopoVec  = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
    } else {
      filteredOTopoVec = Teuchos::null;
      filteredTopoVec = Teuchos::null;
    }

    // create overlap topo vector for output purposes
    overlapTopoVec = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
    overlapdfdpVec = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
    dfdpVec  = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
    topoVec  = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
                                              
                                              //* target *//   //* source *//
    importer = Teuchos::rcp(new Epetra_Import(*overlapNodeMap, *localNodeMap));


    // create exporter (for integration type operations):
                                              //* source *//   //* target *//
    exporter = Teuchos::rcp(new Epetra_Export(*overlapNodeMap, *localNodeMap));

#ifdef ATO_FILTER_ON
    // this should go somewhere else.  for now ...
    GlobalPoint gp;
    int blockcounts[3] = {1,3,1};
    MPI_Datatype oldtypes[3] = {MPI_INT, MPI_DOUBLE, MPI_UB};
    MPI_Aint offsets[3] = {(MPI_Aint)&(gp.gid)    - (MPI_Aint)&gp, 
                           (MPI_Aint)&(gp.coords) - (MPI_Aint)&gp, 
                           sizeof(GlobalPoint)};
    MPI_Type_struct(3,blockcounts,offsets,oldtypes,&MPI_GlobalPoint);
    MPI_Type_commit(&MPI_GlobalPoint);

    // initialize/build the filter operators. these are built once.
    int nFilters = filters.size();
    for(int ifltr=0; ifltr<nFilters; ifltr++){
      filters[ifltr]->buildOperator(
        _subProblems[0].app, _topology, 
        overlapNodeMap, localNodeMap,
        importer, exporter); 
    }
#endif //ATO_FILTER_ON

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

  _derivativeFilter   = Teuchos::null;
  _topologyFilter     = Teuchos::null;
  _postTopologyFilter = Teuchos::null;
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

  if(_solverComm->MyPID() == 0){
  }
  
}

/******************************************************************************/
void
ATO::Solver::copyTopologyIntoStateMgr( const double* p, Albany::StateManager& stateMgr )
/******************************************************************************/
{

  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  int numWorksets = dest.size();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const Teuchos::Array<std::string>& fixedBlocks = _topology->getFixedBlocks();

  if( _topology->getCentering() == "Element" ){
    int wsOffset = 0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topology->getName()];
      int wsSize = wsTopo.size();
      if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) == fixedBlocks.end() ){
        for(int i=0; i<wsSize; i++)
          wsTopo(i) = p[wsOffset+i];
      }
      wsOffset += wsSize;
    }
  } else 
  if( _topology->getCentering() == "Node" ){

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
      wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

    // communicate boundary info
    int numLocalNodes = topoVec->MyLength();
    double* ltopo; topoVec->ExtractView(&ltopo);
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topology->getName()];
      int numCells = wsTopo.dimension(0);
      int numNodes = wsTopo.dimension(1);
      if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) == fixedBlocks.end() ) {
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = localNodeMap->LID(gid);
            if(lid != -1) ltopo[lid] = p[lid];
          }
      } else {
        double matVal = _topology->getMaterialValue();
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = localNodeMap->LID(gid);
            if(lid != -1) ltopo[lid] = matVal;
          }
      }
    }

    // save topology to nodal data for output sake
    Teuchos::RCP<Albany::NodeFieldContainer> 
      nodeContainer = stateMgr.getNodalDataBlock()->getNodeContainer();

    const Teuchos::RCP<const Teuchos_Comm>
      commT = Albany::createTeuchosCommFromEpetraComm(overlapTopoVec->Comm());

    // apply filter if requested
    if(_topologyFilter != Teuchos::null){
      Epetra_Vector filtered_topoVec(*topoVec);
      _topologyFilter->FilterOperator()->Multiply(/*UseTranspose=*/false, *topoVec, filtered_topoVec);
      *topoVec = filtered_topoVec;
    } else
    if(_postTopologyFilter != Teuchos::null){
      _postTopologyFilter->FilterOperator()->Multiply(/*UseTranspose=*/false, *topoVec, *filteredTopoVec);
      filteredOTopoVec->Import(*filteredTopoVec, *importer, Insert);
      std::string nodal_topoName = _topology->getName()+"_node_filtered";
      const Teuchos::RCP<const Tpetra_Vector>
        filteredOTopoVecT = Petra::EpetraVector_To_TpetraVectorConst(
          *filteredOTopoVec, commT);      
      (*nodeContainer)[nodal_topoName]->saveFieldVector(filteredOTopoVecT,/*offset=*/0);
    }

    overlapTopoVec->Import(*topoVec, *importer, Insert);

    double* otopo; overlapTopoVec->ExtractView(&otopo);
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topology->getName()];
      int numCells = wsTopo.dimension(0);
      int numNodes = wsTopo.dimension(1);
      if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) != fixedBlocks.end() ) continue;
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMap->LID(gid);
          wsTopo(cell,node) = otopo[lid];
        }
    }

    std::string nodal_topoName = _topology->getName()+"_node";
    const Teuchos::RCP<const Tpetra_Vector>
      overlapTopoVecT = Petra::EpetraVector_To_TpetraVectorConst(
        *overlapTopoVec, commT);
    (*nodeContainer)[nodal_topoName]->saveFieldVector(overlapTopoVecT,/*offset=*/0);

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

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const Teuchos::Array<std::string>& fixedBlocks = _topology->getFixedBlocks();

  if( _topology->getCentering() == "Element" ){
    int wsOffset = 0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& dfdpSrc = src[ws][derName];
      int wsSize = dfdpSrc.size();
      if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) == fixedBlocks.end() ){
        for(int i=0; i<wsSize; i++)
          dfdp[wsOffset+i] = dfdpSrc(i);
      }
      wsOffset += wsSize;
    }
  } else
  if( _topology->getCentering() == "Node" ){

    Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
      wsElNodeID = disc->getWsElNodeID();

    dfdpVec->PutScalar(0.0);
    overlapdfdpVec->PutScalar(0.0);
    double* odfdp; overlapdfdpVec->ExtractView(&odfdp);

    for(int ws=0; ws<numWorksets; ws++){
      if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) != fixedBlocks.end() ) continue;
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
    dfdpVec->Export(*overlapdfdpVec, *exporter, Add);

    int numLocalNodes = dfdpVec->MyLength();
    double* lvec; dfdpVec->ExtractView(&lvec);



    // apply filter if requested
    if(_derivativeFilter != Teuchos::null){
      Epetra_Vector filtered_dfdpVec(*dfdpVec);
      _derivativeFilter->FilterOperator()->Multiply(/*UseTranspose=*/false, *dfdpVec, filtered_dfdpVec);
      filtered_dfdpVec.ExtractView(&lvec);
      std::memcpy((void*)dfdp, (void*)lvec, numLocalNodes*sizeof(double));
    } else {
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
  if( _topology->getCentering() == "Node" ){
    // communicate boundary topo data
    Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
    Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
    Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
    int numWorksets = dest.size();
  
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
      wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

    int numLocalNodes = topoVec->MyLength();
    double* ltopo; topoVec->ExtractView(&ltopo);
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topology->getName()];
      int numCells = wsTopo.dimension(0);
      int numNodes = wsTopo.dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = localNodeMap->LID(gid);
          if(lid != -1) ltopo[lid] = p[lid];
        }
      }
  
      overlapTopoVec->Import(*topoVec, *importer, Insert);
  
      double* otopo; overlapTopoVec->ExtractView(&otopo);
      for(int ws=0; ws<numWorksets; ws++){
        Albany::MDArray& wsTopo = dest[ws][_topology->getName()];
        int numCells = wsTopo.dimension(0);
        int numNodes = wsTopo.dimension(1);
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = overlapNodeMap->LID(gid);
            wsTopo(cell,node) = otopo[lid];
          }
      }
  
    return _atoProblem->ComputeVolume(otopo, v, dvdp);
  }

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
  if( _topology->getCentering() == "Element" ){
    Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
    Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
    Albany::StateArrayVec& dest = stateArrays.elemStateArrays;

    int numWorksets = dest.size();

    int numTotalElems = 0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topology->getName()];
      int wsSize = wsTopo.size();
      numTotalElems += wsSize;
    }
    return numTotalElems;
    
  } else
  if( _topology->getCentering() == "Node" ){
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

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "ATO Solver creating solver from " << appParams->name()
       << " parameter list" << std::endl;

  //! Create solver and application objects via solver factory
  {
    RCP<const Teuchos_Comm> commT = Albany::createTeuchosCommFromEpetraComm(comm);
    const RCP<const Epetra_Comm> appComm = Teuchos::rcpFromRef(comm);

    //! Create solver factory, which reads xml input filen
    Albany::SolverFactory slvrfctry(appParams, commT);

    RCP<const Tpetra_Vector> initial_guessT = Teuchos::null;
    if (!initial_guess.is_null())
      initial_guessT = Petra::EpetraVector_To_TpetraVectorConst(*initial_guess, commT);
    ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, appComm, appComm, initial_guessT);
  }


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
  physics_probParams.set<Teuchos::RCP<Topology> >("Topology",_topology);


  // Add aggregator information
  Teuchos::ParameterList& aggParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Objective Aggregator");
  physics_probParams.set<Teuchos::ParameterList>("Objective Aggregator",aggParams);

  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& physics_discList = physics_appParams->sublist("Discretization", false);
  physics_discList.setParameters(discList);
  // find the output file name and append "Physics_n_" to it. This only checks for exodus output.
  if( physics_discList.isType<std::string>("Exodus Output File Name") ){
    std::stringstream newname;
    newname << "physics_" << physIndex << "_" 
            << physics_discList.get<std::string>("Exodus Output File Name");
    physics_discList.set("Exodus Output File Name",newname.str());
  }

  if( _topology->getFixedBlocks().size() > 0 )
    physics_discList.set("Separate Evaluators by Element Block", true);

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

  // Specify topology
  validPL->sublist("Topology", false, "");

  // Specify responses
  validPL->sublist("Spatial Filters", false, "");

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
  //no index because num_g == 1 so j must be zero
  if      (j <  _num_responses) return _epetra_response_map; 
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
    //uses local length (need to modify to work with distributed params)
    if(solver_p != Teuchos::null) ret.pLength[i] = solver_p->MyLength();
    else ret.pLength[i] = 0;
  }

  ret.Ng = sub.responses_out->Ng();
  ret.gLength = std::vector<int>(ret.Ng);
  for(int i=0; i<ret.Ng; i++) {
    Teuchos::RCP<const Epetra_Vector> solver_g = sub.responses_out->get_g(i);
    //uses local length (need to modify to work with distributed responses)
    if(solver_g != Teuchos::null) ret.gLength[i] = solver_g->MyLength();
    else ret.gLength[i] = 0;
  }

  if(ret.Np > 0) {
    Teuchos::RCP<const Epetra_Vector> p_init =
      //only first p vector used - in the future could make ret.p_init an array of Np vectors
      sub.model->get_p_init(0);
    if(p_init != Teuchos::null) ret.p_init = Teuchos::rcp(new const Epetra_Vector(*p_init)); //copy
    else ret.p_init = Teuchos::null;
  }
  else ret.p_init = Teuchos::null;

  return ret;
}

/******************************************************************************/
void
ATO::SpatialFilter::buildOperator(
             Teuchos::RCP<Albany::Application> app,
             Teuchos::RCP<Topology>            topology,
             Teuchos::RCP<Epetra_Map>          overlapNodeMap,
             Teuchos::RCP<Epetra_Map>          localNodeMap,
             Teuchos::RCP<Epetra_Import>       importer,
             Teuchos::RCP<Epetra_Export>       exporter)
/******************************************************************************/
{

  if( topology->getCentering() == "Node" ){

    Teuchos::RCP<Adapt::NodalDataBlock> node_data = app->getStateMgr().getNodalDataBlock();

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
          wsElNodeID = app->getDiscretization()->getWsElNodeID();
  
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
      coords = app->getDiscretization()->getCoords();
  
    std::map< GlobalPoint, std::set<GlobalPoint> > neighbors;
  
    double filter_radius_sqrd = filterRadius*filterRadius;
    // awful n^2 search... all against all
    size_t dimension   = app->getDiscretization()->getNumDim();
    size_t num_worksets = coords.size();
    GlobalPoint homeNode;
    for (size_t home_ws=0; home_ws<num_worksets; home_ws++) {
      int home_num_cells = coords[home_ws].size();
      for (int home_cell=0; home_cell<home_num_cells; home_cell++) {
        size_t num_nodes = coords[home_ws][home_cell].size();
        for (int home_node=0; home_node<num_nodes; home_node++) {
          homeNode.gid = wsElNodeID[home_ws][home_cell][home_node];
          if(neighbors.find(homeNode)==neighbors.end()) {  // if this node was already accessed just skip
            for (int dim=0; dim<dimension; dim++)  {
              homeNode.coords[dim] = coords[home_ws][home_cell][home_node][dim];
            }
            std::set<GlobalPoint> my_neighbors;
            for (size_t trial_ws=0; trial_ws<num_worksets; trial_ws++) {
              int trial_num_cells = coords[trial_ws].size();
              for (int trial_cell=0; trial_cell<trial_num_cells; trial_cell++) {
                size_t trial_num_nodes = coords[trial_ws][trial_cell].size();
                for (int trial_node=0; trial_node<trial_num_nodes; trial_node++) {
                  double tmp;
                  double delta_norm_sqr = 0.;
                  for (int dim=0; dim<dimension; dim++)  { //individual coordinates
                    tmp = homeNode.coords[dim]-coords[trial_ws][trial_cell][trial_node][dim];
                    delta_norm_sqr += tmp*tmp;
                  }
                  if(delta_norm_sqr<=filter_radius_sqrd) {
                    GlobalPoint newIntx;
                    newIntx.gid = wsElNodeID[trial_ws][trial_cell][trial_node];
                    for (int dim=0; dim<dimension; dim++) 
                      newIntx.coords[dim] = coords[trial_ws][trial_cell][trial_node][dim];
                    my_neighbors.insert(newIntx);
                  }
                }
              }
            }
            neighbors.insert( std::pair<GlobalPoint,std::set<GlobalPoint> >(homeNode,my_neighbors) );
          }
        }
      }
    }

    // communicate neighbor data
    importNeighbors(neighbors,importer,exporter);

    
    // for each interior node, search boundary nodes for additional interactions off processor.
    
  
    // now build filter operator
    int numnonzeros = 0;
    filterOperator = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*localNodeMap,numnonzeros));
    for (std::map<GlobalPoint,std::set<GlobalPoint> >::iterator 
        it=neighbors.begin(); it!=neighbors.end(); ++it) { 
      GlobalPoint homeNode = it->first;
      int home_node_gid = homeNode.gid;
      std::set<GlobalPoint> connected_nodes = it->second;
      for (std::set<GlobalPoint>::iterator 
           set_it=connected_nodes.begin(); set_it!=connected_nodes.end(); ++set_it) {
         int neighbor_node_gid = set_it->gid;
         const double* coords = &(set_it->coords[0]);
         double distance = 0.0;
         for (int dim=0; dim<dimension; dim++) 
           distance += (coords[dim]-homeNode.coords[dim])*(coords[dim]-homeNode.coords[dim]);
         distance = (distance > 0.0) ? sqrt(distance) : 0.0;
         double weight = filterRadius - distance;
         filterOperator->InsertGlobalValues(home_node_gid,1,&weight,&neighbor_node_gid);
      }
    }
  
    filterOperator->FillComplete();

    // scale filter operator so rows sum to one.
    Epetra_Vector rowSums(*localNodeMap);
    filterOperator->InvRowSums(rowSums);
    filterOperator->RightScale(rowSums);

  } else {
    // Element centered filter
  }
  
  return;

}
/******************************************************************************/
ATO::SpatialFilter::SpatialFilter( Teuchos::ParameterList& params )
/******************************************************************************/
{
  filterRadius = params.get<double>("Filter Radius");
}

/******************************************************************************/
void 
ATO::SpatialFilter::importNeighbors( 
  std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >& neighbors,
  Teuchos::RCP<Epetra_Import> importer,
  Teuchos::RCP<Epetra_Export> exporter)
/******************************************************************************/
{
  // get from the exporter the node global ids and the associated processor ids
  std::map<int, std::set<int> > boundaryNodesByProc;

  const int* exportLIDs = exporter->ExportLIDs();
  const int* exportPIDs = exporter->ExportPIDs();
  int numExportIDs = exporter->NumExportIDs();

  const Epetra_BlockMap& expNodeMap = exporter->SourceMap();

  std::map<int, std::set<int> >::iterator procIter;
  for(int i=0; i<numExportIDs; i++){
    procIter = boundaryNodesByProc.find(exportPIDs[i]);
    int exportGID = expNodeMap.GID(exportLIDs[i]);
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(exportGID);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(exportPIDs[i],newSet) );
    } else {
      procIter->second.insert(exportGID);
    }
  }

  const Epetra_BlockMap& impNodeMap = importer->SourceMap();

  exportLIDs = importer->ExportLIDs();
  exportPIDs = importer->ExportPIDs();
  numExportIDs = importer->NumExportIDs();

  for(int i=0; i<numExportIDs; i++){
    procIter = boundaryNodesByProc.find(exportPIDs[i]);
    int exportGID = impNodeMap.GID(exportLIDs[i]);
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(exportGID);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(exportPIDs[i],newSet) );
    } else {
      procIter->second.insert(exportGID);
    }
  }

  int newPoints = 1;
  
  while(newPoints > 0){
    newPoints = 0;

    int numNeighborProcs = boundaryNodesByProc.size();
    std::vector<std::vector<int> > numNeighbors_send(numNeighborProcs);
    std::vector<std::vector<int> > numNeighbors_recv(numNeighborProcs);
    
 
    // determine number of neighborhood nodes to be communicated
    int index = 0;
    std::map<int, std::set<int> >::iterator boundaryNodesIter;
    for( boundaryNodesIter=boundaryNodesByProc.begin(); 
         boundaryNodesIter!=boundaryNodesByProc.end(); 
         boundaryNodesIter++){
   
      int send_to = boundaryNodesIter->first;
      int recv_from = send_to;
  
      std::set<int>& boundaryNodes = boundaryNodesIter->second; 
      int numNodes = boundaryNodes.size();
  
      numNeighbors_send[index].resize(numNodes);
      numNeighbors_recv[index].resize(numNodes);
  
      ATO::GlobalPoint sendPoint;
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator sendPointIter;
      int localIndex = 0;
      std::set<int>::iterator boundaryNodeGID;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        sendPoint.gid = *boundaryNodeGID;
        sendPointIter = neighbors.find(sendPoint);
        TEUCHOS_TEST_FOR_EXCEPT( sendPointIter == neighbors.end() );
        std::set<ATO::GlobalPoint>& sendPointSet = sendPointIter->second;
        numNeighbors_send[index][localIndex] = sendPointSet.size();
        localIndex++;
      }
  
      MPI_Status status;
      MPI_Sendrecv(&(numNeighbors_send[index][0]), numNodes, MPI_INT, send_to, 0,
                   &(numNeighbors_recv[index][0]), numNodes, MPI_INT, recv_from, 0,
                   MPI_COMM_WORLD, &status);
      index++;
    }
  
    // new neighbors can't be immediately added to the neighbor map or they'll be
    // found and added to the list that's communicated to other procs.  This causes
    // problems because the message length has already been communicated.  
    std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> > newNeighbors;
  
    // communicate neighborhood nodes
    index = 0;
    for( boundaryNodesIter=boundaryNodesByProc.begin(); 
         boundaryNodesIter!=boundaryNodesByProc.end(); 
         boundaryNodesIter++){
   
      // determine total message size
      int totalNumEntries_send = 0;
      int totalNumEntries_recv = 0;
      std::vector<int>& send = numNeighbors_send[index];
      std::vector<int>& recv = numNeighbors_recv[index];
      int totalNumNodes = send.size();
      for(int i=0; i<totalNumNodes; i++){
        totalNumEntries_send += send[i];
        totalNumEntries_recv += recv[i];
      }
  
      int send_to = boundaryNodesIter->first;
      int recv_from = send_to;
  
      ATO::GlobalPoint* GlobalPoints_send = new ATO::GlobalPoint[totalNumEntries_send];
      ATO::GlobalPoint* GlobalPoints_recv = new ATO::GlobalPoint[totalNumEntries_recv];
      
      // copy into contiguous memory
      std::set<int>& boundaryNodes = boundaryNodesIter->second;
      ATO::GlobalPoint sendPoint;
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator sendPointIter;
      std::set<int>::iterator boundaryNodeGID;
      int numNodes = boundaryNodes.size();
      int offset = 0;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        // get neighbors for boundary node i
        sendPoint.gid = *boundaryNodeGID;
        sendPointIter = neighbors.find(sendPoint);
        TEUCHOS_TEST_FOR_EXCEPT( sendPointIter == neighbors.end() );
        std::set<ATO::GlobalPoint>& sendPointSet = sendPointIter->second;
        // copy neighbors into contiguous memory
        for(std::set<ATO::GlobalPoint>::iterator igp=sendPointSet.begin(); 
            igp!=sendPointSet.end(); igp++){
          GlobalPoints_send[offset] = *igp;
          offset++;
        }
      }
  
      MPI_Status status;
      MPI_Sendrecv(GlobalPoints_send, totalNumEntries_send, MPI_GlobalPoint, send_to, 0,
                   GlobalPoints_recv, totalNumEntries_recv, MPI_GlobalPoint, recv_from, 0,
                   MPI_COMM_WORLD, &status);
  
      // copy out of contiguous memory
      ATO::GlobalPoint recvPoint;
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator recvPointIter;
      offset = 0;
      int localIndex=0;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        recvPoint.gid = *boundaryNodeGID;
        recvPointIter = newNeighbors.find(recvPoint);
        if( recvPointIter == newNeighbors.end() ){ // not found, add.
          std::set<ATO::GlobalPoint> newPointSet;
          int nrecv = recv[localIndex];
          for(int j=0; j<nrecv; j++){
            newPointSet.insert(GlobalPoints_recv[offset]);
            offset++;
          }
          newNeighbors.insert( std::pair<ATO::GlobalPoint,std::set<ATO::GlobalPoint> >(recvPoint,newPointSet) );
        } else {
          int nrecv = recv[localIndex];
          for(int j=0; j<nrecv; j++){
            recvPointIter->second.insert(GlobalPoints_recv[offset]);
            offset++;
          }
        }
        localIndex++;
      }
   
      delete [] GlobalPoints_send;
      delete [] GlobalPoints_recv;
      
      index++;
    }
  
    // add newNeighbors map to neighbors map
    std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator new_nbr;
    std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator nbr;
    std::set< ATO::GlobalPoint >::iterator newPoint;
    // loop on total neighbor list
    for(nbr=neighbors.begin(); nbr!=neighbors.end(); nbr++){
  
      std::set<ATO::GlobalPoint>& pointSet = nbr->second;
      int pointSetSize = pointSet.size();
  
      ATO::GlobalPoint home_point = nbr->first;
      double* home_coords = &(home_point.coords[0]);
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator nbrs;
      std::set< ATO::GlobalPoint >::iterator remote_point;
      for(nbrs=newNeighbors.begin(); nbrs!=newNeighbors.end(); nbrs++){
        std::set<ATO::GlobalPoint>& remote_points = nbrs->second;
        for(remote_point=remote_points.begin(); 
            remote_point!=remote_points.end();
            remote_point++){
          const double* remote_coords = &(remote_point->coords[0]);
          double distance = 0.0;
          for(int i=0; i<3; i++)
            distance += (remote_coords[i]-home_coords[i])*(remote_coords[i]-home_coords[i]);
          distance = (distance > 0.0) ? sqrt(distance) : 0.0;
          if( distance < filterRadius )
            pointSet.insert(*remote_point);
        }
      }
      // see if any new points where found off processor.  
      newPoints += (pointSet.size() - pointSetSize);
    }
    int globalNewPoints=0;
    impNodeMap.Comm().SumAll(&newPoints, &globalNewPoints, 1);
    newPoints = globalNewPoints;
  }
}
  
  
#ifdef ATO_FILTER_ON
#undef ATO_FILTER_ON
#endif //ATO_FILTER_ON
  
  
  
