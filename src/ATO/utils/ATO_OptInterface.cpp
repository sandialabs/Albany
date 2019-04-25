////*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_SpatialFilter.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Tpetra_RowMatrixTransposer.hpp"

#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Adapt_NodalDataVector.hpp"
#include "Petra_Converters.hpp"
#include "Epetra_LinearProblem.h"
#include "AztecOO.h"

#ifdef ATO_USES_ISOLIB
#include "Albany_STKDiscretization.hpp"
#include "STKExtract.hpp"
#endif


MPI_Datatype MPI_GlobalPoint;


/******************************************************************************/
void
ATO::OptInterface::ComputeMeasure(std::string measureType, const double* p, 
                                  double& measure)
/******************************************************************************/
{
  ComputeMeasure(measureType, p, measure, NULL, "Gauss Quadrature");
}

/******************************************************************************/
void
ATO::OptInterface::ComputeMeasure(std::string measureType, const double* p, 
                                  double& measure, double* dmdp)
/******************************************************************************/
{
  ComputeMeasure(measureType, p, measure, dmdp, "Gauss Quadrature");
}

/******************************************************************************/
void
ATO::OptInterface::ComputeMeasure(std::string measureType, const double* p, 
                                  double& measure, std::string integrationMethod)
/******************************************************************************/
{
  ComputeMeasure(measureType, p, measure, NULL, integrationMethod);
}

/******************************************************************************/
void
ATO::Solver::ComputeMeasure(std::string measureType, const double* p, 
                            double& measure, double* dmdp, 
                            std::string integrationMethod)
/******************************************************************************/
{
  // communicate boundary topo data
  Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
  
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

  int numWorksets = wsElNodeID.size();

  int ntopos = _topologyInfoStructsT.size();

  std::vector<Teuchos::RCP<TopologyStructT> > topologyStructsT(ntopos);

  for(int itopo=0; itopo<ntopos; itopo++){

    topologyStructsT[itopo] = Teuchos::rcp(new TopologyStructT);
  
    Teuchos::RCP<Tpetra_Vector> topoVecT = _topologyInfoStructsT[itopo]->localVectorT;
    int numLocalNodes = topoVecT->getLocalLength();
    int offset = itopo*numLocalNodes;
    Teuchos::ArrayRCP<double> ltopoT = topoVecT->get1dViewNonConst(); 
    for(int ws=0; ws<numWorksets; ws++){
      int numCells = wsElNodeID[ws].size();
      int numNodes = wsElNodeID[ws][0].size();
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = localNodeMapT->getLocalElement(gid);
          if(lid != -1) ltopoT[lid] = p[lid+offset];
        }
    }

    Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
    smoothTopologyT(topoStructT);

    Teuchos::RCP<Tpetra_Vector> overlapTopoVecT = _topologyInfoStructsT[itopo]->overlapVectorT;
    overlapTopoVecT->doImport(*topoVecT, *importerT, Tpetra::INSERT);

    topologyStructsT[itopo]->topologyT = _topologyInfoStructsT[itopo]->topologyT; 
    topologyStructsT[itopo]->dataVectorT = overlapTopoVecT;
  }

  return _atoProblem->ComputeMeasureT(measureType, topologyStructsT, 
                                     measure, dmdp, integrationMethod);
}


/******************************************************************************/
void
ATO::Solver::ComputeVolume(double* p, const double* dfdp, 
                           double& v, double threshhold, double minP)
/******************************************************************************/
{
  /*  Assumptions:
      -- dfdp is already consistent across proc boundaries.
      -- the volume computation that's done by the atoProblem updates the topology, p.
      -- Since dfdp is 'boundary consistent', the resulting topology, p, is also
         'boundary consistent', so no communication is necessary.
  */
  return _atoProblem->ComputeVolume(p, dfdp, v, threshhold, minP);
}

/******************************************************************************/
void
ATO::Solver::Compute(double* p, double& g, double* dgdp, double& c, double* dcdp)
/******************************************************************************/
{
  Compute((const double*)p, g, dgdp, c, dcdp);
}

/******************************************************************************/
void
ATO::Solver::Compute(const double* p, double& g, double* dgdp, double& c, double* dcdp)
/******************************************************************************/
{
  for(int i=0; i<_numPhysics; i++){

    // copy data from p into each stateManager
    if( entityType == "State Variable" ){
      Albany::StateManager& stateMgr = _subProblems[i].app->getStateMgr();
      copyTopologyIntoStateMgr( p, stateMgr );
    } else 
    if( entityType == "Distributed Parameter"){
      copyTopologyIntoParameter( p, _subProblems[i] );
    }

    // enforce PDE constraints
    _subProblems[i].model->evalModel((*_subProblems[i].params_in),
                                    (*_subProblems[i].responses_out));
  }

  if ( entityType == "Distributed Parameter" ) {
    updateTpetraResponseMaps(); 
    _objAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
  }
  _objAggregator->EvaluateT();
  copyObjectiveFromStateMgr( g, dgdp );
  
  if( !_conAggregator.is_null()){
    if ( entityType == "Distributed Parameter" ) {
      updateTpetraResponseMaps(); 
      _conAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
    }
    _conAggregator->EvaluateT();
    copyConstraintFromStateMgr( c, dcdp );
  } else c = 0.0;

  _iteration++;

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
//  return _subProblems[0].app->getDiscretization()->getNodeMap()->NumMyElements();
  auto nVecs = ObjectiveGradientVecT.size();
  return nVecs*ObjectiveGradientVecT[0]->getLocalLength();
}

/******************************************************************************/
///*********************** SETUP AND UTILITY FUNCTIONS **********************///
/******************************************************************************/


/******************************************************************************/
ATO::SolverSubSolver
ATO::Solver::CreateSubSolver( const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                              const Teuchos::RCP<const Teuchos_Comm>& commT,
                              const Teuchos::RCP<const Thyra_Vector>& initial_guess)
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

    //! Create solver factory, which reads xml input filen
    Albany::SolverFactory slvrfctry(appParams, commT);

    ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, commT, commT, initial_guess);
  }


  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");

  int numParameters = 0;
  if( problemParams.isType<Teuchos::ParameterList>("Parameters") )
    numParameters = problemParams.sublist("Parameters").get<int>("Number of Parameter Vectors");

  int numResponseSpecs = 0;
  if( problemParams.isType<Teuchos::ParameterList>("Response Functions") )
    numResponseSpecs = problemParams.sublist("Response Functions").get<int>("Number of Response Vectors");

  bool separateByBlock=false;
  Teuchos::ParameterList& discParams = appParams->sublist("Discretization");
  if(discParams.isType<bool>("Separate Evaluators by Element Block")){
    separateByBlock = discParams.get<bool>("Separate Evaluators by Element Block");
  }
  int numBlocks=1;
  if(separateByBlock){
    numBlocks = 
      problemParams.sublist("Configuration").sublist("Element Blocks").get<int>("Number of Element Blocks");
  }

  ret.params_in = rcp(new EpetraExt::ModelEvaluator::InArgs);
  ret.responses_out = rcp(new EpetraExt::ModelEvaluator::OutArgs);

  *(ret.params_in) = ret.model->createInArgs();
  *(ret.responses_out) = ret.model->createOutArgs();

  // the createOutArgs() function doesn't allocate storage
  RCP<Epetra_Vector> g1;
  int ss_num_g = ret.responses_out->Ng(); // Number of *vectors* of responses
  for(int ig=0; ig<ss_num_g; ig++){
    g1 = rcp(new Epetra_Vector(*(ret.model->get_g_map(ig))));
    ret.responses_out->set_g(ig,g1);
  }

  RCP<Epetra_Vector> p1;
  int ss_num_p = ret.params_in->Np();     // Number of *vectors* of parameters
  TEUCHOS_TEST_FOR_EXCEPTION (
    ss_num_p - numParameters > 1,
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error! Cannot have more than one distributed Parameter for topology optimization" << std::endl);
  for(int ip=0; ip<ss_num_p; ip++){
    p1 = rcp(new Epetra_Vector(*(ret.model->get_p_init(ip))));
    ret.params_in->set_p(ip,p1);
  }

  for(int iResponseSpec=0; iResponseSpec<numResponseSpecs; iResponseSpec++){
    if(ss_num_p > numParameters){
      int ip = ss_num_p-1;
      Teuchos::ParameterList& resParams = 
        problemParams.sublist("Response Functions").sublist(Albany::strint("Response Vector",iResponseSpec));
      std::string gName = resParams.get<std::string>("Response Name");
      std::string dgdpName = resParams.get<std::string>("Response Derivative Name");
 
      std::vector<RCP<const Epetra_Vector>> gVector(numBlocks);
      std::vector<RCP<Epetra_MultiVector>> dgdpVector(numBlocks);
      std::vector<RCP<const Tpetra_Vector>> gVectorT(numBlocks);
      std::vector<RCP<Tpetra_MultiVector>> dgdpVectorT(numBlocks);
      for(int iBlock=0; iBlock<numBlocks; iBlock++){
        int ig = iResponseSpec*numBlocks + iBlock;
        if(!ret.responses_out->supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, ig, ip).none()){
          RCP<const Epetra_Vector> p = ret.params_in->get_p(ip);
          gVector[iBlock]  = ret.responses_out->get_g(ig);
          gVectorT[iBlock] = Petra::EpetraVector_To_TpetraVectorConst(*gVector[iBlock], _solverComm); 

          dgdpVector[iBlock]  = rcp(new Epetra_MultiVector(p->Map(), gVector[iBlock]->GlobalLength() ));
          dgdpVectorT[iBlock] = Petra::EpetraMultiVector_To_TpetraMultiVector(*dgdpVector[iBlock], _solverComm); 

          if(ret.responses_out->supports(OUT_ARG_DgDp,ig,ip).supports(DERIV_TRANS_MV_BY_ROW)){
            Derivative dgdp_out(dgdpVector[iBlock], DERIV_TRANS_MV_BY_ROW);
            ret.responses_out->set_DgDp(ig,ip,dgdp_out);
          } else {
            ret.responses_out->set_DgDp(ig,ip,dgdpVector[iBlock]);
          }
        }
      }
      responseMap.insert(std::pair<std::string,std::vector<RCP<const Epetra_Vector>>>(gName,gVector));
      responseMapT.insert(std::pair<std::string,std::vector<RCP<const Tpetra_Vector>>>(gName, gVectorT));

      responseDerivMap.insert(std::pair<std::string,std::vector<RCP<Epetra_MultiVector>>>(dgdpName,dgdpVector));
      responseDerivMapT.insert(std::pair<std::string,std::vector<RCP<Tpetra_MultiVector>>>(dgdpName, dgdpVectorT));
    }
  }

  RCP<Epetra_Vector> xfinal =
    rcp(new Epetra_Vector(*(ret.model->get_g_map(ss_num_g-1)),true) );
  ret.responses_out->set_g(ss_num_g-1,xfinal);

  return ret;
}

/******************************************************************************/
void 
ATO::Solver::updateTpetraResponseMaps()
/******************************************************************************/
{
  std::map<std::string, std::vector<Teuchos::RCP<const Epetra_Vector>>>::const_iterator git;
  git = responseMap.cbegin();  
  for (int i = 0; i<responseMap.size(); i++) {
    std::string gName = git->first;
    std::vector<Teuchos::RCP<const Epetra_Vector>> gVector = git->second;
    int numVectors = gVector.size();
    for(int iVector=0; iVector<numVectors; iVector++){
      Teuchos::RCP<const Tpetra_Vector> 
        gT = Petra::EpetraVector_To_TpetraVectorConst(*gVector[iVector], _solverComm); 
      responseMapT[gName][iVector] = gT;
    }
    git++; 
  }
  std::map<std::string, std::vector<Teuchos::RCP<Epetra_MultiVector>>>::const_iterator git2;
  git2 = responseDerivMap.cbegin();  
  for (int i = 0; i<responseDerivMap.size(); i++) {
    std::string gName = git2->first; 
    std::vector<Teuchos::RCP<Epetra_MultiVector>> dgdpVector = git2->second;
    int numVectors = dgdpVector.size();
    for(int iVector=0; iVector<numVectors; iVector++){
      Teuchos::RCP<Tpetra_MultiVector> 
        dgdpT = Petra::EpetraMultiVector_To_TpetraMultiVector(*dgdpVector[iVector], _solverComm); 
      responseDerivMapT[gName][iVector] = dgdpT;
    }
    git2++; 
  }
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
  physics_probParams.set<Teuchos::RCP<TopologyArray> >("Topologies",_topologyArrayT);

  Teuchos::ParameterList& topoParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Topologies");
  physics_probParams.set<Teuchos::ParameterList>("Topologies Parameters",topoParams);

  // Check topology.  If the topology is a distributed parameter, then 1) check for existing 
  // "Distributed Parameter" list and error out if found, and 2) add a "Distributed Parameter" 
  // list to the input file, 
  if( entityType == "Distributed Parameter" ){
    TEUCHOS_TEST_FOR_EXCEPTION (
      physics_subList.isType<Teuchos::ParameterList>("Distributed Parameters"),
      Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error! Cannot have 'Distributed Parameters' in both Topology and subproblems" << std::endl);
    Teuchos::ParameterList distParams;
    int ntopos = _topologyInfoStructsT.size();
    distParams.set("Number of Parameter Vectors",ntopos);
    for(int itopo=0; itopo<ntopos; itopo++){
      distParams.set(Albany::strint("Parameter",itopo), _topologyInfoStructsT[itopo]->topologyT->getName());
    }
    physics_probParams.set<Teuchos::ParameterList>("Distributed Parameters", distParams);
  }

  // Add aggregator information
  Teuchos::ParameterList& aggParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Objective Aggregator");
  physics_probParams.set<Teuchos::ParameterList>("Objective Aggregator",aggParams);

  // Add configuration information
  Teuchos::ParameterList& conParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Configuration");
  physics_probParams.set<Teuchos::ParameterList>("Configuration",conParams);

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

  int ntopos = _topologyInfoStructsT.size();
  for(int itopo=0; itopo<ntopos; itopo++){
    if( _topologyInfoStructsT[itopo]->topologyT->getFixedBlocks().size() > 0 ){
      physics_discList.set("Separate Evaluators by Element Block", true);
      break;
    }
  }

  if( _writeDesignFrequency != 0 )
    physics_discList.set("Use Automatic Aura", true);

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
Teuchos::RCP<Teuchos::ParameterList> 
ATO::Solver::createHomogenizationInputFile( 
    const Teuchos::RCP<Teuchos::ParameterList>& appParams, 
    const Teuchos::ParameterList& homog_subList, 
    int homogProblemIndex, 
    int homogSubIndex, 
    int homogDim) const
/******************************************************************************/
{   

  const Teuchos::ParameterList& homog_problem_subList = 
    homog_subList.sublist("Problem");

  // Create input parameter list for app which mimics a separate input file
  std::stringstream appStream;
  appStream << "Parameters for Homogenization Subapplication " << homogSubIndex;
  Teuchos::RCP<Teuchos::ParameterList> homog_appParams = Teuchos::createParameterList(appStream.str());

  // get reference to Problem ParameterList in new input file and initialize it 
  // from Parameters in homogenization base problem
  Teuchos::ParameterList& homog_probParams = homog_appParams->sublist("Problem",false);
  homog_probParams.setParameters(homog_problem_subList);

  // set up BCs (this is a pretty bad klugde till periodic BC's are available)
  Teuchos::ParameterList& Params = homog_probParams.sublist("Dirichlet BCs",false);

  homog_probParams.set("Add Cell Problem Forcing",homogSubIndex);

  const Teuchos::ParameterList& bcIdParams = homog_subList.sublist("Cell BCs");
  Teuchos::Array<std::string> dofs = bcIdParams.get<Teuchos::Array<std::string> >("DOF Names");
  std::string dofsType = bcIdParams.get<std::string>("DOF Type");
  bool isVector;
  if( dofsType == "Scalar" ){
    isVector = false;
    TEUCHOS_TEST_FOR_EXCEPTION(dofs.size() != 1, Teuchos::Exceptions::InvalidParameter, 
                               std::endl << "Error: Expected DOF Names array to be length 1." << std::endl);
  } else
  if( dofsType == "Vector" ){
    isVector = true;
    TEUCHOS_TEST_FOR_EXCEPTION(dofs.size() != homogDim, Teuchos::Exceptions::InvalidParameter, 
                               std::endl << "Error: Expected DOF Names array to be length " << homogDim << "." << std::endl);
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                     std::endl << "Homogenization DOFs must be of type Scalar or Vector (not " << dofsType << ")." << std::endl);

  if(homogDim == 1){
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
    std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
  } else 
  if(homogDim == 2){
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    int negY = bcIdParams.get<int>("Negative Y Face");
    int posY = bcIdParams.get<int>("Positive Y Face");
    if( homogSubIndex < 2 ){
      std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
      std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      if( isVector ){
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[1]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[1]; Params.set(namePosY.str(),0.0);
      } else {
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
      }
    } else {
      std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
      std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
      if( isVector ){
        std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[1]; Params.set(nameNegX.str(),0.0);
        std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[1]; Params.set(namePosX.str(),0.0);
      } else {
        std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
        std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      }
    }
  } else 
  if(homogDim == 3){
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    int negY = bcIdParams.get<int>("Negative Y Face");
    int posY = bcIdParams.get<int>("Positive Y Face");
    int negZ = bcIdParams.get<int>("Negative Z Face");
    int posZ = bcIdParams.get<int>("Positive Z Face");
    if( homogSubIndex < 3 ){
      std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
      std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      if( isVector ){
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[1]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[1]; Params.set(namePosY.str(),0.0);
        std::stringstream nameNegZ; nameNegZ << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[2]; Params.set(nameNegZ.str(),0.0);
        std::stringstream namePosZ; namePosZ << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[2]; Params.set(namePosZ.str(),0.0);
      } else {
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
        std::stringstream nameNegZ; nameNegZ << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[0]; Params.set(nameNegZ.str(),0.0);
        std::stringstream namePosZ; namePosZ << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[0]; Params.set(namePosZ.str(),0.0);
      }
    } else {
      std::stringstream nameNegYX; nameNegYX << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegYX.str(),0.0);
      std::stringstream namePosYX; namePosYX << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosYX.str(),0.0);
      std::stringstream nameNegZX; nameNegZX << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[0]; Params.set(nameNegZX.str(),0.0);
      std::stringstream namePosZX; namePosZX << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[0]; Params.set(namePosZX.str(),0.0);
      if( isVector ){
        std::stringstream nameNegXY; nameNegXY << "DBC on NS nodelist_" << negX << " for DOF " << dofs[1]; Params.set(nameNegXY.str(),0.0);
        std::stringstream namePosXY; namePosXY << "DBC on NS nodelist_" << posX << " for DOF " << dofs[1]; Params.set(namePosXY.str(),0.0);
        std::stringstream nameNegZY; nameNegZY << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[1]; Params.set(nameNegZY.str(),0.0);
        std::stringstream namePosZY; namePosZY << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[1]; Params.set(namePosZY.str(),0.0);
        std::stringstream nameNegXZ; nameNegXZ << "DBC on NS nodelist_" << negX << " for DOF " << dofs[2]; Params.set(nameNegXZ.str(),0.0);
        std::stringstream namePosXZ; namePosXZ << "DBC on NS nodelist_" << posX << " for DOF " << dofs[2]; Params.set(namePosXZ.str(),0.0);
        std::stringstream nameNegYZ; nameNegYZ << "DBC on NS nodelist_" << negY << " for DOF " << dofs[2]; Params.set(nameNegYZ.str(),0.0);
        std::stringstream namePosYZ; namePosYZ << "DBC on NS nodelist_" << posY << " for DOF " << dofs[2]; Params.set(namePosYZ.str(),0.0);
      } else {
        std::stringstream nameNegXY; nameNegXY << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegXY.str(),0.0);
        std::stringstream namePosXY; namePosXY << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosXY.str(),0.0);
      }
    }
  }

  

  // Discretization sublist processing
  const Teuchos::ParameterList& discList = homog_subList.sublist("Discretization");
  Teuchos::ParameterList& homog_discList = homog_appParams->sublist("Discretization", false);
  homog_discList.setParameters(discList);
  // find the output file name and append "homog_n_" to it. This only checks for exodus output.
  if( homog_discList.isType<std::string>("Exodus Output File Name") ){
    std::stringstream newname;
    newname << "homog_" << homogProblemIndex << "_" << homogSubIndex << "_" 
            << homog_discList.get<std::string>("Exodus Output File Name");
    homog_discList.set("Exodus Output File Name",newname.str());
  }

  // Piro sublist processing
  homog_appParams->set("Piro",appParams->sublist("Piro"));

  
  
  return homog_appParams;
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
  validPL->set<int>("Number of Homogenization Problems", 0, "Number of homogenization problems");
  validPL->set<bool>("Verbose Output", false, "Enable detailed output mode");
  validPL->set<int>("Design Output Frequency", 0, "Write isosurface every N iterations");
  validPL->set<std::string>("Name", "", "String to designate Problem");

  // Specify physics problem(s)
  for(int i=0; i<_numPhysics; i++){
    std::stringstream physStream; physStream << "Physics Problem " << i;
    validPL->sublist(physStream.str(), false, "");
  }
  
  int numHomogProblems = _homogenizationSets.size();
  for(int i=0; i<numHomogProblems; i++){
    std::stringstream homogStream; homogStream << "Homogenization Problem " << i;
    validPL->sublist(homogStream.str(), false, "");
  }

  validPL->sublist("Objective Aggregator", false, "");

  validPL->sublist("Constraint Aggregator", false, "");

  validPL->sublist("Topological Optimization", false, "");

  validPL->sublist("Topologies", false, "");

  validPL->sublist("Configuration", false, "");

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
  inArgs.set_Np(c_num_parameters);
  return inArgs;
}

/******************************************************************************/
EpetraExt::ModelEvaluator::OutArgs 
ATO::Solver::createOutArgs() const
/******************************************************************************/
{
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("ATO Solver Multipurpose Model Evaluator");
  outArgs.set_Np_Ng(c_num_parameters, c_num_responses);
  return outArgs;
}

/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_g_map(int j) const
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(j != 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl << "Error in ATO::Solver::get_g_map():  " <<
                     "Invalid response index j = " << j << std::endl);

  return _epetra_x_map;
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
             Teuchos::RCP<const Tpetra_Map>    overlapNodeMapT,
             Teuchos::RCP<const Tpetra_Map>    localNodeMapT,
             Teuchos::RCP<Tpetra_Import>       importerT,
             Teuchos::RCP<Tpetra_Export>       exporterT)
/******************************************************************************/
{

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
          wsElNodeID = app->getDiscretization()->getWsElNodeID();
  
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
      coords = app->getDiscretization()->getCoords();

    const Albany::WorksetArray<std::string>::type& 
      wsEBNames = app->getDiscretization()->getWsEBNames();

    // if this filter operates on a subset of the blocks in the mesh, create a list
    // of nodes that are not smoothed:
    std::set<int> excludeNodes;
    if( blocks.size() > 0 ){
      size_t num_worksets = coords.size();
      // add to the excludeNodes set all nodes that are not to be smoothed
      for (size_t ws=0; ws<num_worksets; ws++) {
        if( find(blocks.begin(), blocks.end(), wsEBNames[ws]) != blocks.end() ) continue;
        int num_cells = coords[ws].size();
        for (int cell=0; cell<num_cells; cell++) {
          size_t num_nodes = coords[ws][cell].size();
          for (int node=0; node<num_nodes; node++) {
            int gid = wsElNodeID[ws][cell][node];
            excludeNodes.insert(gid);
          }
        }
      }
      // remove from the excludeNodes set all nodes that are on boundaries 
      // between smoothed and non-smoothed blocks
      std::set<int>::iterator it;
      for (size_t ws=0; ws<num_worksets; ws++) {
        if( find(blocks.begin(), blocks.end(), wsEBNames[ws]) == blocks.end() ) continue;
        int num_cells = coords[ws].size();
        for (int cell=0; cell<num_cells; cell++) {
          size_t num_nodes = coords[ws][cell].size();
          for (int node=0; node<num_nodes; node++) {
            int gid = wsElNodeID[ws][cell][node];
            it = excludeNodes.find(gid);
            excludeNodes.erase(it,excludeNodes.end());
          }
        }
      }
    }
  
    std::map< GlobalPoint, std::set<GlobalPoint> > neighbors;
  
    double filter_radius_sqrd = filterRadius*filterRadius;
    // awful n^2 search... all against all
    size_t dimension   = app->getDiscretization()->getNumDim();
    GlobalPoint homeNode;
    size_t num_worksets = coords.size();
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
            if( excludeNodes.find(homeNode.gid) == excludeNodes.end() ){
              for (size_t trial_ws=0; trial_ws<num_worksets; trial_ws++) {
                if( blocks.size() > 0 && 
                    find(blocks.begin(), blocks.end(), wsEBNames[trial_ws]) == blocks.end() ) continue;
                int trial_num_cells = coords[trial_ws].size();
                for (int trial_cell=0; trial_cell<trial_num_cells; trial_cell++) {
                  size_t trial_num_nodes = coords[trial_ws][trial_cell].size();
                  for (int trial_node=0; trial_node<trial_num_nodes; trial_node++) {
                    int gid = wsElNodeID[trial_ws][trial_cell][trial_node];
                    if( excludeNodes.find(gid) != excludeNodes.end() ) continue; // don't add excluded nodes
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
            }
            neighbors.insert( std::pair<GlobalPoint,std::set<GlobalPoint> >(homeNode,my_neighbors) );
          }
        }
      }
    }

    // communicate neighbor data
    importNeighbors(neighbors,importerT,*localNodeMapT,exporterT,*overlapNodeMapT);
    
    // for each interior node, search boundary nodes for additional interactions off processor.
    
    // now build filter operator
    int numnonzeros = 0;
    Teuchos::RCP<const Epetra_Comm> comm = 
      Albany::createEpetraCommFromTeuchosComm(localNodeMapT->getComm());
    Teuchos::RCP<const Epetra_Map> localNodeMap = Petra::TpetraMap_To_EpetraMap(localNodeMapT, comm);
    filterOperatorT = Teuchos::rcp(new Tpetra_CrsMatrix(localNodeMapT,numnonzeros));
    for (std::map<GlobalPoint,std::set<GlobalPoint> >::iterator 
        it=neighbors.begin(); it!=neighbors.end(); ++it) { 
      GlobalPoint homeNode = it->first;
      Tpetra_GO home_node_gid = homeNode.gid;
      std::set<GlobalPoint> connected_nodes = it->second;
      ST weight; 
      ST zero = 0.0;  
      Teuchos::Array<ST> weightT(1);
      if( connected_nodes.size() > 0 ){
        for (std::set<GlobalPoint>::iterator 
             set_it=connected_nodes.begin(); set_it!=connected_nodes.end(); ++set_it) {
           Tpetra_GO neighbor_node_gid = set_it->gid;
           const double* coords = &(set_it->coords[0]);
           double distance = 0.0;
           for (int dim=0; dim<dimension; dim++) 
             distance += (coords[dim]-homeNode.coords[dim])*(coords[dim]-homeNode.coords[dim]);
           distance = (distance > 0.0) ? sqrt(distance) : 0.0;
           weight = filterRadius - distance;
           filterOperatorT->insertGlobalValues(home_node_gid,1,&zero,&neighbor_node_gid); 
           filterOperatorT->replaceGlobalValues(home_node_gid,1,&weight,&neighbor_node_gid); 
        }
      } else {
         // if the list of connected nodes is empty, still add a one on the diagonal.
         weight = 1.0;
         filterOperatorT->insertGlobalValues(home_node_gid,1,&zero,&home_node_gid); 
         filterOperatorT->replaceGlobalValues(home_node_gid,1,&weight,&home_node_gid); 
      }
    }
  
    filterOperatorT->fillComplete();

    // scale filter operator so rows sum to one.
    Teuchos::RCP<Tpetra_Vector> rowSumsT = Teuchos::rcp(new Tpetra_Vector(filterOperatorT->getRowMap()));
    Albany::InvAbsRowSum(rowSumsT, filterOperatorT); 
    filterOperatorT->leftScale(*rowSumsT); 

    //IKT, FIXME: remove the following creation of filterOperatorTransposeT 
    //once Mark Hoemmen fixes apply method with TRANS mode in Tpetra::CrsMatrix.
    Tpetra::RowMatrixTransposer<ST,Tpetra_LO,Tpetra_GO,KokkosNode> transposer(filterOperatorT);
    filterOperatorTransposeT = transposer.createTranspose();

  return;

}


/******************************************************************************/
ATO::SpatialFilter::SpatialFilter( Teuchos::ParameterList& params )
/******************************************************************************/
{
  filterRadius = params.get<double>("Filter Radius");
  if( params.isType<Teuchos::Array<std::string> >("Blocks") ){
    blocks = params.get<Teuchos::Array<std::string> >("Blocks");
  }
  if( params.isType<int>("Iterations") ){
    iterations = params.get<int>("Iterations");
  } else
    iterations = 1;

}

/******************************************************************************/
void 
ATO::SpatialFilter::importNeighbors( 
  std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >& neighbors,
  Teuchos::RCP<Tpetra_Import> importerT, 
  const Tpetra_Map& impNodeMapT,
  Teuchos::RCP<Tpetra_Export> exporterT, 
  const Tpetra_Map& expNodeMapT)
/******************************************************************************/
{
  // get from the exporter the node global ids and the associated processor ids
  std::map<int, std::set<int> > boundaryNodesByProc;

  Teuchos::ArrayView<const LO> exportLIDsT = exporterT->getExportLIDs(); 
  Teuchos::ArrayView<const int> exportPIDsT = exporterT->getExportPIDs(); 
  int numExportIDsT = exporterT->getNumExportIDs();
  std::map<int, std::set<int> >::iterator procIter;
  for(int i=0; i<numExportIDsT; i++){
    procIter = boundaryNodesByProc.find(exportPIDsT[i]);
    int exportGIDT = expNodeMapT.getGlobalElement(exportLIDsT[i]);
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(exportGIDT);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(exportPIDsT[i],newSet) );
    } else {
      procIter->second.insert(exportGIDT);
    }
  }

  exportLIDsT = importerT->getExportLIDs();
  exportPIDsT = importerT->getExportPIDs();
  numExportIDsT = importerT->getNumExportIDs();

  for(int i=0; i<numExportIDsT; i++){
    procIter = boundaryNodesByProc.find(exportPIDsT[i]);
    int exportGIDT = impNodeMapT.getGlobalElement(exportLIDsT[i]);
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(exportGIDT);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(exportPIDsT[i],newSet) );
    } else {
      procIter->second.insert(exportGIDT);
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
    Teuchos::reduceAll(*(impNodeMapT.getComm()), Teuchos::REDUCE_SUM, 1, &newPoints, &globalNewPoints); 
    newPoints = globalNewPoints;
  }
}
  
