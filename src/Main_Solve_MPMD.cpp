//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include <matrix_container.hpp>
#include <communicator.hpp>

#include "Albany_Memory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"

#include "Piro_PerformSolve.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

#include "ATO_TopoTools.hpp"

// Uncomment for run time nan checking
// This is set in the toplevel CMakeLists.txt file
//#define ALBANY_CHECK_FPE

#if defined(ALBANY_CHECK_FPE)
#include <math.h>
//#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
#endif
//#define ALBANY_FLUSH_DENORMALS
#if defined(ALBANY_FLUSH_DENORMALS)
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif

#include "Albany_DataTypes.hpp"

#include "Phalanx_config.hpp"

#include "Kokkos_Core.hpp"

#if defined(ALBANY_APF)
#include "Albany_APFMeshStruct.hpp"
#endif

#include <Plato_Application.hpp>
#include <Plato_Exceptions.hpp>
#include <Plato_Interface.hpp>
#include <Plato_PenaltyModel.hpp>
#include <Plato_SharedData.hpp>
#include <Plato_SharedField.hpp>

/******************************************************************************/
class MPMD_App : public Plato::Application 
/******************************************************************************/
{
  public:
    MPMD_App(int argc, char **argv, MPI_Comm& localComm);
    virtual ~MPMD_App();
    void initialize();
    void compute(const std::string &);
    void finalize();
  
    void importData(const std::string & name, const Plato::SharedData& sf);
    void exportData(const std::string & name, Plato::SharedData& sf);
    void exportDataMap(const Plato::data::layout_t & aDataLayout, 
                       std::vector<int> & aMyOwnedGlobalIDs);
  
  private:

    // functions
    void throwParsingException(
      std::string spec, std::string name, 
      const std::map<std::string,std::vector<double>*>& valueMap);

    void throwParsingException(
      std::string spec, std::string name, 
      const std::map<std::string,DistributedVector*>& fieldMap);

    void copyFieldIntoState(const std::string& name, const Plato::SharedData& sf);
    void copyFieldFromState(const std::string& name, Plato::SharedData& sf);
    void copyFieldIntoDistParam(const std::string& name, const Plato::SharedData& sf);
    void copyFieldFromDistParam(const std::string& name, Plato::SharedData& sf);

    void copyValueFromState(const std::string& name, Plato::SharedData& sf);

    bool isElemNodeState(std::string localFieldName);
    bool isDistParam(std::string localFieldName);

    bool addField(pugi::xml_node&);
    bool addValue(pugi::xml_node&);

    // data
    Teuchos::RCP<Albany::Application> m_app;
    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>> m_solver;
    Teuchos::RCP<Albany::SolverFactory> m_solverFactory;
    Teuchos::RCP<const Teuchos_Comm> m_comm;
    Teuchos::Array<Teuchos::RCP<const Thyra_Vector>> m_responses;
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra_MultiVector>>> m_sensitivities;

    Teuchos::RCP<Thyra_Vector> m_localVector;
    Teuchos::RCP<Thyra_Vector> m_overlapVector;

    Teuchos::RCP<Albany::CombineAndScatterManager> m_cas_manager;

    pugi::xml_document m_inputTree;
  
    std::map<std::string,std::string> m_stateMap, m_distParamMap;


};
/******************************************************************************/



/******************************************************************************/
int main(int argc, char **argv)
/******************************************************************************/
{
    MPI_Init(&argc, &argv);

    Plato::Interface* platoInterface = new Plato::Interface();

    MPI_Comm localComm;
    platoInterface->getLocalComm(localComm);

    MPMD_App* myApp = new MPMD_App(argc, argv, localComm);
    platoInterface->registerPerformer(myApp);

    platoInterface->perform();

    delete myApp;

    MPI_Finalize();

}



/******************************************************************************/
MPMD_App::MPMD_App(int argc, char **argv, MPI_Comm& localComm)
/******************************************************************************/
{

  int status = 0;  // 0 = pass, failures are incremented
  bool success = true;

  Kokkos::initialize(argc, argv);

#ifdef ALBANY_FLUSH_DENORMALS
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#ifdef ALBANY_CHECK_FPE
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

#ifdef ALBANY_APF
  Albany::APFMeshStruct::initialize_libraries(&argc, &argv);
#endif

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Command-line argument for input file
  Albany::CmdLineArgs cmd;
  cmd.parse_cmdline(argc, argv, *out);

  try {
    auto totalTimer = Teuchos::rcp(new Teuchos::TimeMonitor(
        *Teuchos::TimeMonitor::getNewTimer("Albany: Total Time")));
    auto setupTimer = Teuchos::rcp(new Teuchos::TimeMonitor(
        *Teuchos::TimeMonitor::getNewTimer("Albany: Setup Time")));

    m_comm = Albany::getDefaultComm();

    // Connect vtune for performance profiling
    if (cmd.vtune) { Albany::connect_vtune(m_comm->getRank()); }

    // parse input into Teuchos::ParameterList
    Teuchos::RCP<Teuchos::ParameterList>
      appParams = Teuchos::createParameterList("Albany Parameters");
    Teuchos::updateParametersFromXmlFileAndBroadcast(cmd.yaml_filename, appParams.ptr(), *m_comm);

    const auto& bt = appParams->get("Build Type","Tpetra");
    if (bt=="Tpetra") {
      // Set the static variable that denotes this as a Tpetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));
    } else if (bt=="Epetra") {
      // Set the static variable that denotes this as a Epetra run
      static_cast<void>(Albany::build_type(Albany::BuildType::Epetra));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
                                 "Error! Invalid choice (" + bt + ") for 'BuildType'.\n"
                                 "       Valid choicses are 'Epetra', 'Tpetra'.\n");
    }

    Teuchos::ParameterList& probParams = appParams->sublist("Problem",false);

    Teuchos::ParameterList& topoParams = probParams.get<Teuchos::ParameterList>("Topologies");
    int ntopos = topoParams.get<int>("Number of Topologies");

    Teuchos::RCP<ATO::TopologyArray>
    topologyArray = Teuchos::rcp( new Teuchos::Array<Teuchos::RCP<ATO::Topology> >(ntopos) );
    for(int itopo=0; itopo<ntopos; itopo++){
      Teuchos::ParameterList& tParams = topoParams.sublist(Albany::strint("Topology",itopo));
      (*topologyArray)[itopo] = Teuchos::rcp(new ATO::Topology(tParams, itopo));
    }
    
    // add topology objects
    probParams.set<Teuchos::RCP<ATO::TopologyArray> >("Topologies",topologyArray);
    
    // send in ParameterList instead of yaml filename 
    m_solverFactory = rcp(new Albany::SolverFactory(appParams, m_comm));
    m_solver = m_solverFactory->createAndGetAlbanyApp(m_app, m_comm, m_comm);

    setupTimer = Teuchos::null;

    std::string solnMethod =
        m_solverFactory->getParameters().sublist("Problem").get<std::string>(
            "Solution Method");
    if (solnMethod == "Transient Tempus No Piro") {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Please run AlbanyTempus executable with Solution "
                 "Method = Transient Tempus No Piro.\n");
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  if (!success) status += 10000;

  
  const char* input_char = std::getenv("PLATO_APP_FILE");
  pugi::xml_parse_result result = m_inputTree.load_file(input_char);
  if(!result){
    //throw
  }
}



/******************************************************************************/
void MPMD_App::initialize()
/******************************************************************************/
{

  Albany::StateManager& stateMgr = m_app->getStateMgr();
  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  Teuchos::RCP<const Thyra_VectorSpace> localNodeVS   = disc->getNodeVectorSpace();
  Teuchos::RCP<const Thyra_VectorSpace> overlapNodeVS = disc->getOverlapNodeVectorSpace();

  m_localVector    = Thyra::createMember(localNodeVS);
  m_overlapVector  = Thyra::createMember(overlapNodeVS);
  m_cas_manager    = Albany::createCombineAndScatterManager(localNodeVS,overlapNodeVS);

  // parse Operation definition
  //
  pugi::xml_node node = m_inputTree.child("Operation");
  if( m_inputTree.next_sibling("Operation") ){ 
    std::stringstream message;
    message << "Only one 'Operation' definition allowed per Albany performer.  Multiple provided." << std::endl;
    Plato::ParsingException pe(message.str());
    throw pe;
  }

  for(pugi::xml_node inputNode : node.children("Input")){
    std::string strType = Plato::Parse::getString(inputNode,"Type");
    if(strType == "Field") addField(inputNode);
    else
    if(strType == "Value") addValue(inputNode);
  }
  for(pugi::xml_node inputNode : node.children("Output")){
    std::string strType = Plato::Parse::getString(inputNode,"Type");
    if(strType == "Field") addField(inputNode);
    else
    if(strType == "Value") addValue(inputNode);
  }
}

/******************************************************************************/
bool MPMD_App::addValue(pugi::xml_node& inputNode)
/******************************************************************************/
{
  string argumentName = Plato::Parse::getString(inputNode,"ArgumentName");

  // map from the argument name to the local data name
  string localName = Plato::Parse::getString(inputNode,"LocalName");
  if( isElemNodeState(localName) ){
    m_stateMap[argumentName] = localName;
  } else {
    std::stringstream message;
    message << "Cannot find specified LocalName: '" << localName << "'" << std::endl;
    Plato::ParsingException pe(message.str());
    throw pe;
  }
}

/******************************************************************************/
bool MPMD_App::addField(pugi::xml_node& inputNode)
/******************************************************************************/
{
  string argumentName = Plato::Parse::getString(inputNode,"ArgumentName");

  // map from the argument name to the local data name
  string localName = Plato::Parse::getString(inputNode,"LocalName");
  if( isElemNodeState(localName) ){
    m_stateMap[argumentName] = localName;
  } else
  if( isDistParam(localName) ){
    m_distParamMap[argumentName] = localName;
  } else {
    std::stringstream message;
    message << "Cannot find specified LocalName: '" << localName << "'" << std::endl;
    Plato::ParsingException pe(message.str());
    throw pe;
  }
}

/******************************************************************************/
bool MPMD_App::isElemNodeState(std::string localFieldName)
/******************************************************************************/
{
  Albany::StateManager&  stateMgr     = m_app->getStateMgr();
  Albany::StateArrays&   stateArrays  = stateMgr.getStateArrays();
  Albany::StateArrayVec& elemNodeFlds = stateArrays.elemStateArrays;
  int numWorksets = elemNodeFlds.size();
  if(numWorksets){
    auto it = elemNodeFlds[0].find(localFieldName);
    if(it != elemNodeFlds[0].end()){
      return true;
    }
  }
  return false;
}
/******************************************************************************/
bool MPMD_App::isDistParam(std::string localFieldName)
/******************************************************************************/
{
  return false;
}

/******************************************************************************/
void
MPMD_App::compute(const std::string & noOp)
/******************************************************************************/
{

  Teuchos::ParameterList &solveParams = m_solverFactory->getAnalysisParameters().sublist("Solve", false);

  Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<ST>>> thyraResponses;
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<ST>>>> thyraSensitivities;

  Piro::PerformSolve(*m_solver, solveParams, thyraResponses, thyraSensitivities);
}

/******************************************************************************/
void MPMD_App::finalize()
/******************************************************************************/
{
  Kokkos::finalize_all();
}

/******************************************************************************/
void MPMD_App::importData(const std::string& name, const Plato::SharedData& sf)
/******************************************************************************/
{
  if(sf.myLayout() == Plato::data::layout_t::SCALAR_FIELD){
    {
      auto it = m_stateMap.find(name);
      if(it != m_stateMap.end()){
        copyFieldIntoState(it->second, sf);
        return;
      }
    }
    {
      auto it = m_distParamMap.find(name);
      if(it != m_distParamMap.end()){
        copyFieldIntoDistParam(it->second, sf);
        return;
      }
    }
  } else
  if(sf.myLayout() == Plato::data::layout_t::SCALAR){
  }
}


/******************************************************************************/
void MPMD_App::copyFieldIntoState(const std::string& name, const Plato::SharedData& sf)
/******************************************************************************/
{
  Albany::StateManager& stateMgr = m_app->getStateMgr();

  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  int numWorksets = dest.size();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = disc->getWsElNodeID();

  m_localVector->assign(0.0);
  Teuchos::ArrayRCP<double> ltopo = Albany::getNonconstLocalData(m_localVector);
  int numLocalVals = ltopo.size();

  std::vector<double> outVector(numLocalVals);
  sf.getData(outVector);
  ltopo.assign(outVector.begin(),outVector.end());

  m_cas_manager->scatter(m_localVector,m_overlapVector,Albany::CombineMode::INSERT);
  Teuchos::RCP<Albany::NodeFieldContainer>
    nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();
  auto it = nodeContainer->find(name+"_node");
  if(it != nodeContainer->end())
    (*nodeContainer)[name+"_node"]->saveFieldVector(m_overlapVector,/*offset=*/0);

  // copy the field into the state manager
  Teuchos::RCP<const Thyra_VectorSpace> overlapNodeVS = m_overlapVector->space();
  Teuchos::ArrayRCP<double> otopo = Albany::getNonconstLocalData(m_overlapVector);
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& wsTopo = dest[ws][name];
    const int numCells = wsTopo.dimension(0);
    const int numNodes = wsTopo.dimension(1);
    for(int cell=0; cell<numCells; cell++)
      for(int node=0; node<numNodes; node++){
        int gid = wsElNodeID[ws][cell][node];
        int lid = Albany::getLocalElement(overlapNodeVS,gid);
        wsTopo(cell,node) = otopo[lid];
      }
  }
}

/******************************************************************************/
void MPMD_App::copyValueFromState(const std::string& name, Plato::SharedData& sf)
/******************************************************************************/
{
  Albany::StateManager& stateMgr = m_app->getStateMgr();

  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& src = stateArrays.elemStateArrays;

  // copy the field from the state manager
  Albany::MDArray& valSrc = src[/*worksetIndex=*/0][name];
  std::vector<double> globalVal(1);
  std::vector<double> val(1,valSrc(0));
  if( m_comm != Teuchos::null ){
    Teuchos::reduceAll(*m_comm, Teuchos::REDUCE_SUM, /*numvals=*/ 1, val.data(), globalVal.data());
  } else globalVal = val;
  sf.setData(globalVal);
}

/******************************************************************************/
void MPMD_App::copyFieldFromState(const std::string& name, Plato::SharedData& sf)
/******************************************************************************/
{
  Albany::StateManager& stateMgr = m_app->getStateMgr();

  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& src = stateArrays.elemStateArrays;
  int numWorksets = src.size();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = disc->getWsElNodeID();

  m_overlapVector->assign(0.0);

  // copy the field from the state manager
  auto data = Albany::getNonconstLocalData(m_overlapVector);
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& wsSrc = src[ws][name];
    const int numCells = wsSrc.dimension(0);
    const int numNodes = wsSrc.dimension(1);
    for(int cell=0; cell<numCells; cell++)
      for(int node=0; node<numNodes; node++) {
        int lid = Albany::getLocalElement(m_overlapVector->space(),wsElNodeID[ws][cell][node]);
        data[lid] += wsSrc(cell,node);
      }
  }

  m_localVector->assign(0.0);
  m_cas_manager->combine(m_overlapVector,m_localVector,Albany::CombineMode::ADD);
  Teuchos::ArrayRCP<double> ltopo = Albany::getNonconstLocalData(m_localVector);
  int numLocalVals = ltopo.size();
  std::vector<double> inVector(ltopo.begin(),ltopo.end());
  sf.setData(inVector);

  Teuchos::RCP<Albany::NodeFieldContainer>
    nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();
  auto it = nodeContainer->find(name+"_node");
  if(it != nodeContainer->end()){
    m_cas_manager->scatter(m_localVector,m_overlapVector,Albany::CombineMode::INSERT);
    (*nodeContainer)[name+"_node"]->saveFieldVector(m_overlapVector,/*offset=*/0);
  }
}

/******************************************************************************/
void MPMD_App::copyFieldIntoDistParam(const std::string& name, const Plato::SharedData& sf)
/******************************************************************************/
{
}
/******************************************************************************/
void MPMD_App::copyFieldFromDistParam(const std::string& name, Plato::SharedData& sf)
/******************************************************************************/
{
}

/******************************************************************************/
void MPMD_App::exportDataMap(const Plato::data::layout_t & aDataLayout, 
                             std::vector<int> & aMyOwnedGlobalIDs)
{
    aMyOwnedGlobalIDs.clear();

    if(aDataLayout == Plato::data::layout_t::SCALAR_FIELD)
    {
      auto spmd_vs = Albany::getSpmdVectorSpace(m_localVector->space());
      int numLocalVals = spmd_vs->localSubDim();
      aMyOwnedGlobalIDs.resize(numLocalVals);
      for(int lid=0; lid<numLocalVals; lid++){
        aMyOwnedGlobalIDs[lid] = Albany::getGlobalElement(spmd_vs,lid)+1; // Albany's gids start from 0
      }
    }
}
/******************************************************************************/


/******************************************************************************/
void MPMD_App::exportData(const std::string& name, Plato::SharedData& sf)
/******************************************************************************/
{
  if(sf.myLayout() == Plato::data::layout_t::SCALAR_FIELD){
    {
      auto it = m_stateMap.find(name);
      if(it != m_stateMap.end()){
        copyFieldFromState(it->second, sf);
        return;
      }
    }
    {
      auto it = m_distParamMap.find(name);
      if(it != m_distParamMap.end()){
        copyFieldFromDistParam(it->second, sf);
        return;
      }
    }
  } else
  if(sf.myLayout() == Plato::data::layout_t::SCALAR){
    {
      auto it = m_stateMap.find(name);
      if(it != m_stateMap.end()){
        copyValueFromState(it->second, sf);
        return;
      }
    }
  }
}


/******************************************************************************/
void
MPMD_App::throwParsingException(
  std::string spec, std::string name, 
  const std::map<std::string,std::vector<double>*>& valueMap)
/******************************************************************************/
{
    std::stringstream message;
    message << "Cannot find specified '" << spec << "': " << name << std::endl;
    message << "Values available: " << std::endl;
    for(auto f : valueMap){
      message << f.first << std::endl;
    }
    Plato::ParsingException pe(message.str());
    throw pe;
}

/******************************************************************************/
void
MPMD_App::throwParsingException(
  std::string spec, std::string name, 
  const std::map<std::string,DistributedVector*>& fieldMap)
/******************************************************************************/
{
    std::stringstream message;
    message << "Cannot find specified '" << spec << "': " << name << std::endl;
    message << "Fields available: " << std::endl;
    for(auto f : fieldMap){
      message << f.first << std::endl;
    }
    Plato::ParsingException pe(message.str());
    throw pe;
}



/******************************************************************************/
MPMD_App::~MPMD_App()
/******************************************************************************/
{
#ifdef ALBANY_APF
  Albany::APFMeshStruct::finalize_libraries();
#endif

}
