////*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Solver.hpp"
#include "ATO_OptimizationProblem.hpp"
#include "ATO_Optimizer.hpp"
#include "ATO_Aggregator.hpp"
#include "ATO_SpatialFilter.hpp"

#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#ifdef ATO_USES_ISOLIB
#include "Albany_STKDiscretization.hpp"
#include "STKExtract.hpp"
#endif

#include "Teuchos_XMLParameterListHelpers.hpp"

namespace ATO
{

//**********************************************************************
Solver::Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
               const Teuchos::RCP<const Teuchos_Comm>& comm,
               const Teuchos::RCP<const Thyra_Vector>& /* initial_guess */)
//**********************************************************************
 : m_iteration      (1)
 , m_num_parameters (0)
 , m_num_responses  (1)
 , m_is_verbose     (false)
 , m_is_restart     (false)
 , m_solverComm     (comm)
 , m_mainAppParams  (appParams)
{
  m_objectiveValue = Teuchos::rcp(new double[1]);
  *m_objectiveValue = 0.0;

  m_constraintValue = Teuchos::rcp(new double[1]);
  *m_constraintValue = 0.0;

  ///*** PROCESS TOP LEVEL PROBLEM ***///

  // Validate Problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  m_numPhysics = problemParams.get<int>("Number of Subproblems", 1);

  int numHomogProblems = problemParams.get<int>("Number of Homogenization Problems", 0);
  m_homogenizationSets.resize(numHomogProblems);

  problemParams.validateParameters(*getValidProblemParameters(),0);

  // Parse topologies
  Teuchos::ParameterList& topoParams = problemParams.get<Teuchos::ParameterList>("Topologies");
  int ntopos = topoParams.get<int>("Number of Topologies");

  if (topoParams.isType<bool>("Read From Restart")) {
     m_is_restart = topoParams.get<bool>("Read From Restart");
  }

  m_topologyInfoStructs.resize(ntopos);
  m_topologyArray = Teuchos::rcp( new Teuchos::Array<Teuchos::RCP<Topology> >(ntopos) );
  for (int itopo=0; itopo<ntopos; ++itopo) {
    m_topologyInfoStructs[itopo] = Teuchos::rcp( new TopologyInfoStruct() );
    Teuchos::ParameterList& tParams = topoParams.sublist(Albany::strint("Topology",itopo));
    m_topologyInfoStructs[itopo]->topology = Teuchos::rcp(new Topology(tParams, itopo));
    (*m_topologyArray)[itopo] = m_topologyInfoStructs[itopo]->topology;
  }

  // currently all topologies must have the same entity type
  m_entityType = m_topologyInfoStructs[0]->topology->getEntityType();
  for (int itopo=1; itopo<ntopos; ++itopo) {
    TEUCHOS_TEST_FOR_EXCEPTION(m_topologyInfoStructs[itopo]->topology->getEntityType() != m_entityType,
                               Teuchos::Exceptions::InvalidParameter,
                               "Error! Topologies must all have the same entity type.\n");
  }

  // Parse and create optimizer
  Teuchos::ParameterList& optimizerParams = problemParams.get<Teuchos::ParameterList>("Topological Optimization");
  m_optimizer = OptimizerFactory::create(optimizerParams);
  m_optimizer->SetInterface(this);
  m_optimizer->SetCommunicator(comm);

  m_writeDesignFrequency = problemParams.get<int>("Design Output Frequency", 0);

  // Parse and create objective aggregator
  Teuchos::ParameterList& objAggregatorParams = problemParams.get<Teuchos::ParameterList>("Objective Aggregator");
  m_objAggregator = AggregatorFactory::create(objAggregatorParams, m_entityType, ntopos);

  // Parse and create constraint aggregator
  if (problemParams.isType<Teuchos::ParameterList>("Constraint Aggregator")) {
    Teuchos::ParameterList& conAggregatorParams = problemParams.get<Teuchos::ParameterList>("Constraint Aggregator");
    m_conAggregator = AggregatorFactory::create(conAggregatorParams, m_entityType, ntopos);
  }

  // Parse filters
  if (problemParams.isType<Teuchos::ParameterList>("Spatial Filters")) {
    Teuchos::ParameterList& filtersParams = problemParams.get<Teuchos::ParameterList>("Spatial Filters");
    int nFilters = filtersParams.get<int>("Number of Filters");
    for (int ifltr=0; ifltr<nFilters; ++ifltr) {
      std::stringstream filterStream;
      filterStream << "Filter " << ifltr;
      Teuchos::ParameterList& filterParams = filtersParams.get<Teuchos::ParameterList>(filterStream.str());
      Teuchos::RCP<SpatialFilter> newFilter = Teuchos::rcp( new SpatialFilter(filterParams)) ;
      m_filters.push_back(newFilter);
    }
  }
  
  // Assign requested filters to topologies
  for (int itopo=0; itopo<ntopos; ++itopo) {
    Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
    Teuchos::RCP<Topology> topo = topoStruct->topology;

    topoStruct->filterIsRecursive = topoParams.get<bool>("Apply Filter Recursively", true);

    int topologyFilterIndex = topo->SpatialFilterIndex();
    if (topologyFilterIndex >= 0) {
      TEUCHOS_TEST_FOR_EXCEPTION (topologyFilterIndex >= static_cast<int>(m_filters.size()),
                                  Teuchos::Exceptions::InvalidParameter,
                                  "Error!  Spatial filter " << topologyFilterIndex << " requested but not defined.\n");
      topoStruct->filter = m_filters[topologyFilterIndex];
    }

    int topologyOutputFilter = topo->TopologyOutputFilter();
    if (topologyOutputFilter >= 0) {
      TEUCHOS_TEST_FOR_EXCEPTION (topologyOutputFilter >= static_cast<int>(m_filters.size()),
                                  Teuchos::Exceptions::InvalidParameter,
                                  "Error! Spatial filter " << topologyFilterIndex << " requested but not defined.\n");
      topoStruct->postFilter = m_filters[topologyOutputFilter];
    }
  }

  int derivativeFilterIndex = objAggregatorParams.get<int>("Spatial Filter", -1);
  if (derivativeFilterIndex >= 0) {
    TEUCHOS_TEST_FOR_EXCEPTION (derivativeFilterIndex >= static_cast<int>(m_filters.size()),
                                Teuchos::Exceptions::InvalidParameter,
                                "Error! Spatial filter " << derivativeFilterIndex << " requested but not defined.\n");
    m_derivativeFilter = m_filters[derivativeFilterIndex];
  }

  // Get and set the default Piro parameters from a file, if given
  std::string piroFilename  = problemParams.get<std::string>("Piro Defaults Filename", "");
  if (piroFilename.length() > 0) {
    Teuchos::RCP<Teuchos::ParameterList> defaultPiroParams = Teuchos::createParameterList("Default Piro Parameters");
    Teuchos::updateParametersFromXmlFileAndBroadcast(piroFilename, defaultPiroParams.ptr(), *comm);
    Teuchos::ParameterList& piroList = appParams->sublist("Piro", false);
    piroList.setParametersNotAlreadySet(*defaultPiroParams);
  }
  
  // set verbosity
  m_is_verbose = (comm->getRank() == 0) && problemParams.get<bool>("Verbose Output", false);

  ///*** PROCESS SUBPROBLEM(S) ***///
   
  m_subProblemAppParams.resize(m_numPhysics);
  m_subProblems.resize(m_numPhysics);
  for (int i=0; i<m_numPhysics; ++i) {
    m_subProblemAppParams[i] = createInputFile(appParams, i);
    m_subProblems[i] = CreateSubSolver( m_subProblemAppParams[i], m_solverComm);

    // ensure that all subproblems are topology based (i.e., optimizable)
    Teuchos::RCP<Albany::AbstractProblem> problem = m_subProblems[i].app->getProblem();
    OptimizationProblem* atoProblem = dynamic_cast<OptimizationProblem*>(problem.get());
    TEUCHOS_TEST_FOR_EXCEPTION (atoProblem == nullptr, Teuchos::Exceptions::InvalidParameter,
                                "Error! Requested subproblem does not support topologies.\n");
  }
  
  ///*** PROCESS HOMOGENIZATION SUBPROBLEM(S) ***///

  for (int iProb=0; iProb<numHomogProblems; ++iProb) {
    HomogenizationSet& hs = m_homogenizationSets[iProb];
    std::string homog_prob_name = Albany::strint("Homogenization Problem",iProb);
    Teuchos::ParameterList& homogParams = problemParams.get<Teuchos::ParameterList>(homog_prob_name);
    hs.homogDim = homogParams.get<int>("Number of Spatial Dimensions");
    
    // parse the name and type of the homogenized constants
    Teuchos::ParameterList& responsesList = homogParams.sublist("Problem").sublist("Response Functions");
    int nResponses = responsesList.get<int>("Number of Response Vectors");
    bool responseFound = false;
    for (int iResponse=0; iResponse<nResponses; ++iResponse) {
      Teuchos::ParameterList& responseList = responsesList.sublist(Albany::strint("Response Vector",iResponse));
      std::string rname = responseList.get<std::string>("Name");
      if (rname == "Homogenized Constants Response") {
        hs.name = responseList.get<std::string>("Homogenized Constants Name");
        hs.type = responseList.get<std::string>("Homogenized Constants Type");
        responseFound = true;
        break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION (!responseFound, Teuchos::Exceptions::InvalidParameter,
                                "Error! Could not find viable homogenization response.\n");

    int nHomogSubProblems = 0;
    if ( hs.type == "4th Rank Voigt") {
      for (int i=1; i<=hs.homogDim; ++i) {
        nHomogSubProblems += i;
      }
    } else if ( hs.type == "2nd Rank Tensor") {
      nHomogSubProblems = hs.homogDim;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                  "Error! Unknown type (" << hs.type << "). \n"
                                  "       Options are '4th Rank Voigt' or '2nd Rank Tensor'.\n");
    }

    hs.homogenizationAppParams.resize(nHomogSubProblems);
    hs.homogenizationProblems.resize(nHomogSubProblems);
 
    for (int iSub=0; iSub<nHomogSubProblems; ++iSub) {
      hs.homogenizationAppParams[iSub] = createHomogenizationInputFile(appParams, homogParams, iProb, iSub, hs.homogDim);
      hs.homogenizationProblems[iSub] = CreateSubSolver( hs.homogenizationAppParams[iSub], m_solverComm);
    }
  }

  // store a pointer to the first problem as an OptimizationProblem for callbacks
  Teuchos::RCP<Albany::AbstractProblem> problem = m_subProblems[0].app->getProblem();
  m_atoProblem = dynamic_cast<OptimizationProblem*>(problem.get());
  m_atoProblem->setDiscretization(m_subProblems[0].app->getDiscretization());
  m_atoProblem->setCommunicator(comm);
  m_atoProblem->InitTopOpt();

  // get solution map from first subproblem
  Teuchos::RCP<Albany::Application> app = m_subProblems[0].app;
  TEUCHOS_TEST_FOR_EXCEPT (app->getVectorSpace().is_null());
  m_x_vs = app->getVectorSpace();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = app->getDiscretization();

  m_localNodeVS   = Albany::getSpmdVectorSpace(disc->getNodeVectorSpace());
  m_overlapNodeVS = Albany::getSpmdVectorSpace(disc->getOverlapNodeVectorSpace());

  for (int itopo=0; itopo<ntopos; ++itopo) {
    Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
    if (topoStruct->postFilter != Teuchos::null) {
      topoStruct->filteredOverlapVector = Thyra::createMember(*m_overlapNodeVS);
      topoStruct->filteredVector        = Thyra::createMember(*m_localNodeVS);
    }

    // create overlap topo vector for output purposes
    topoStruct->overlapVector = Thyra::createMember(*m_overlapNodeVS);
    topoStruct->localVector   = Thyra::createMember(*m_localNodeVS);
  } 

  m_overlapObjectiveGradientVec.resize(ntopos);
  m_ObjectiveGradientVec.resize(ntopos);
  m_overlapConstraintGradientVec.resize(ntopos);
  m_ConstraintGradientVec.resize(ntopos);
  for (int itopo=0; itopo<ntopos; ++itopo) {
    m_overlapObjectiveGradientVec[itopo]  = Thyra::createMember(*m_overlapNodeVS);
    m_ObjectiveGradientVec[itopo]         = Thyra::createMember(*m_localNodeVS);
    m_overlapConstraintGradientVec[itopo] = Thyra::createMember(*m_overlapNodeVS);
    m_ConstraintGradientVec[itopo]        = Thyra::createMember(*m_localNodeVS);

    // Zero out the vectors
    m_overlapObjectiveGradientVec[itopo] ->assign(0.0);
    m_ObjectiveGradientVec[itopo]->assign(0.0);
    m_overlapConstraintGradientVec[itopo]->assign(0.0);
    m_ConstraintGradientVec[itopo]->assign(0.0);
  } 
  
  m_cas_manager = Albany::createCombineAndScatterManager(m_localNodeVS, m_overlapNodeVS);

  // initialize/build the filter operators. these are built once.
  int nFilters = m_filters.size();
  for (int ifltr=0; ifltr<nFilters; ++ifltr) {
    m_filters[ifltr]->buildOperator(*app,*m_cas_manager);
  }

  // pass subProblems to the objective aggregator
  if (m_entityType == "State Variable") {
    m_objAggregator->SetInputVariables(m_subProblems);
    m_objAggregator->SetOutputVariables(m_objectiveValue, m_overlapObjectiveGradientVec);
  } else if (m_entityType == "Distributed Parameter") {
    m_objAggregator->SetInputVariables(m_subProblems, m_responseMap, m_responseDerivMap);
    m_objAggregator->SetOutputVariables(m_objectiveValue, m_ObjectiveGradientVec);
  }
  m_objAggregator->SetCommunicator(comm);
  
  // pass subProblems to the constraint aggregator
  if (!m_conAggregator.is_null()) {
    if (m_entityType == "State Variable") {
      m_conAggregator->SetInputVariables(m_subProblems);
      m_conAggregator->SetOutputVariables(m_constraintValue, m_overlapConstraintGradientVec);
    } else if (m_entityType == "Distributed Parameter") {
      m_conAggregator->SetInputVariables(m_subProblems, m_responseMap, m_responseDerivMap);
      m_conAggregator->SetOutputVariables(m_constraintValue, m_ConstraintGradientVec);
    }
    m_conAggregator->SetCommunicator(comm);
  }
}

//**********************************************************************
void Solver::evalModelImpl (const Thyra_InArgs&  /* inArgs */,
                            const Thyra_OutArgs& /* outArgs */) const
//**********************************************************************
{
  const int numHomogenizationSets = m_homogenizationSets.size();
  for (int iHomog=0; iHomog<numHomogenizationSets; ++iHomog) {
    const HomogenizationSet& hs = m_homogenizationSets[iHomog];
    const int numColumns = hs.homogenizationProblems.size();
    for (int i=0; i<numColumns; ++i) {
      // enforce PDE constraints
      hs.homogenizationProblems[i].model->evalModel((*hs.homogenizationProblems[i].params_in),
                                                    (*hs.homogenizationProblems[i].responses_out));
    }

    if (numColumns > 0) {
      // collect homogenized values
      Kokkos::DynRankView<RealType, PHX::Device> Cvals("ZZZ", numColumns,numColumns);
      for (int i=0; i<numColumns; ++i) {
        Teuchos::RCP<const Thyra_Vector> g = hs.homogenizationProblems[i].responses_out->get_g(hs.responseIndex);
        auto g_data = Albany::getLocalData(g);
        for (int j=0; j<numColumns; ++j) {
          Cvals(i,j) = g_data[j];
        }
      }
      if (m_solverComm->getRank() == 0) {
        Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
        *out << "*****************************************" << std::endl;
        *out << " Homogenized parameters (" << hs.name << ") are: " << std::endl; 
        for (int i=0; i<numColumns; ++i) {
          for (int j=0; j<numColumns; ++j) {
            *out << std::setprecision(10) << 1.0/2.0*(Cvals(i,j)+Cvals(j,i)) << " ";
          }
          *out << std::endl;
        }
        *out << "*****************************************" << std::endl;
      }

      for (int iPhys=0; iPhys<m_numPhysics; ++iPhys) {
        Albany::StateManager& stateMgr = m_subProblems[iPhys].app->getStateMgr();
        Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
        Albany::StateArrayVec& src = stateArrays.elemStateArrays;
        const int numWorksets = src.size();

        for (int ws=0; ws<numWorksets; ++ws) {
          for (int i=0; i<numColumns; ++i) {
            for (int j=i; j<numColumns; ++j) {
              std::stringstream valname;
              valname << hs.name << " " << i+1 << j+1;
              Albany::MDArray& wsC = src[ws][valname.str()]; 
              if (wsC.size() != 0) {
                wsC(0) = (Cvals(j,i)+Cvals(i,j))/2.0;
              }
            }
          }
        }
      }
    }
  }

  auto indexer = Albany::createGlobalLocalIndexer(m_localNodeVS);
  for (int i=0; i<m_numPhysics; ++i) {
    Albany::StateManager& stateMgr = m_subProblems[i].app->getStateMgr();
    const auto& wsEBNames  = stateMgr.getDiscretization()->getWsEBNames();
    const auto& wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();
    Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
    Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
    const int numWorksets = dest.size();
  
    // initialize topology of fixed blocks
    const int ntopos = m_topologyInfoStructs.size();

    for (int ws=0; ws<numWorksets; ++ws) {

      for (int itopo=0; itopo<ntopos; ++itopo) {
        Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
        Teuchos::RCP<Topology> topology = topoStruct->topology;
        const Teuchos::Array<std::string>& fixedBlocks = topology->getFixedBlocks();

        const auto it = std::find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]);
        if (it != fixedBlocks.end()) {
          if (topology->getEntityType() == "State Variable") {
            const double matVal = topology->getMaterialValue();
            Albany::MDArray& wsTopo = dest[ws][topology->getName()];
            const int numCells = wsTopo.dimension(0);
            const int numNodes = wsTopo.dimension(1);
            for (int cell=0; cell<numCells; ++cell)
              for (int node=0; node<numNodes; ++node) {
                wsTopo(cell,node) = matVal;
              }
          } else if (topology->getEntityType() == "Distributed Parameter") {
            auto ltopo = Albany::getNonconstLocalData(topoStruct->localVector);
            const double matVal = topology->getMaterialValue();
            const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& elNodeID = wsElNodeID[ws];
            const int numCells = elNodeID.size();
            const int numNodes = elNodeID[0].size();
            for (int cell=0; cell<numCells; ++cell) {
              for (int node=0; node<numNodes; ++node) {
                const int gid = wsElNodeID[ws][cell][node];
                if (indexer->isLocallyOwnedElement(gid)) {
                  const int lid = indexer->getLocalElement(gid);
                  ltopo[lid] = matVal;
                }
              }
            }
          }
        }
      }
    }
  }

  if (m_is_verbose) {
    Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
    *out << "*** Performing Topology Optimization Loop ***" << std::endl;
  }

  m_optimizer->Initialize();
  m_optimizer->Optimize();
}

/******************************************************************************/
///*************** SOLVER - OPTIMIZER INTERFACE FUNCTIONS *******************///
/******************************************************************************/

//**********************************************************************
void Solver::getOptDofsUpperBound (Teuchos::Array<double>& b) const
//**********************************************************************
{
  const int nLocal = Albany::getSpmdVectorSpace(m_topologyInfoStructs[0]->localVector->range())->localSubDim();
  const int nTopos = m_topologyInfoStructs.size();
  const int nTerms = nTopos*nLocal;
 
  b.resize(nTerms);

  Teuchos::Array<double>::iterator from = b.begin();
  Teuchos::Array<double>::iterator to = from+nLocal;
  for (int itopo=0; itopo<nTopos; ++itopo) {
    TopologyInfoStruct& topoIS = *m_topologyInfoStructs[itopo];
    Teuchos::Array<double> bounds = topoIS.topology->getBounds();
    std::fill(from, to, bounds[1]);
    from += nLocal; to += nLocal;
  }
}

//**********************************************************************
void Solver::getOptDofsLowerBound (Teuchos::Array<double>& b) const
//**********************************************************************
{
  const int nLocal = Albany::getSpmdVectorSpace(m_topologyInfoStructs[0]->localVector->range())->localSubDim();
  const int nTopos = m_topologyInfoStructs.size();
  const int nTerms = nTopos*nLocal;
 
  b.resize(nTerms);

  Teuchos::Array<double>::iterator from = b.begin();
  Teuchos::Array<double>::iterator to = from+nLocal;
  for (int itopo=0; itopo<nTopos; ++itopo) {
    TopologyInfoStruct& topoIS = *m_topologyInfoStructs[itopo];
    Teuchos::Array<double> bounds = topoIS.topology->getBounds();
    std::fill(from, to, bounds[0]);
    from += nLocal; to += nLocal;
  }
}

//**********************************************************************
void Solver::InitializeOptDofs (double* p)
//**********************************************************************
{
  if (m_is_restart) {
// JR: this needs to be tested for multimaterial
    Albany::StateManager& stateMgr = m_subProblems[0].app->getStateMgr();
    copyTopologyFromStateMgr( p, stateMgr) ;
  } else {
    const int ntopos = m_topologyInfoStructs.size();
    for (int itopo=0; itopo<ntopos; ++itopo) {
      Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
      const int numLocalNodes = Albany::getSpmdVectorSpace(topoStruct->localVector->range())->localSubDim();
      const double initVal = topoStruct->topology->getInitialValue();
      const int fromIndex=itopo*numLocalNodes;
      const int toIndex=fromIndex+numLocalNodes;
      for (int lid=fromIndex; lid<toIndex; ++lid)
        p[lid] = initVal;
    }
  }
}

//**********************************************************************
void Solver::ComputeObjective(const double* p, double& g, double* dgdp)
//**********************************************************************
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "IKt, 12/22/16, WARNING: Tpetra-converted ComputeObjective has not been tested " 
       << "yet and may not work correctly! \n"; 
  for (int i=0; i<m_numPhysics; ++i) {
    // copy data from p into each stateManager
    if (m_entityType == "State Variable") {
      Albany::StateManager& stateMgr = m_subProblems[i].app->getStateMgr();
      copyTopologyIntoStateMgr( p, stateMgr) ;
    } else 
    if (m_entityType == "Distributed Parameter") {
      copyTopologyIntoParameter( p, m_subProblems[i]) ;
    }

    // enforce PDE constraints
    m_subProblems[i].model->evalModel((*m_subProblems[i].params_in),
                                    (*m_subProblems[i].responses_out));
  }

  if ( m_entityType == "Distributed Parameter") {
    m_objAggregator->SetInputVariables(m_subProblems, m_responseMap, m_responseDerivMap);
  }
  m_objAggregator->Evaluate();
  copyObjectiveFromStateMgr( g, dgdp) ;

  m_iteration++;
}

//**********************************************************************
void Solver::ComputeObjective(double* p, double& g, double* dgdp)
//**********************************************************************
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "IKt, 12/22/16, WARNING: Tpetra-converted ComputeObjective has not been tested " 
       << "yet and may not work correctly! \n"; 

  if (m_iteration!=1) {
    smoothTopology(p);
  }

  for (int i=0; i<m_numPhysics; ++i) {
    // copy data from p into each stateManager
    if (m_entityType == "State Variable") {
      Albany::StateManager& stateMgr = m_subProblems[i].app->getStateMgr();
      copyTopologyIntoStateMgr( p, stateMgr) ;
    } else 
    if (m_entityType == "Distributed Parameter") {
      copyTopologyIntoParameter( p, m_subProblems[i]) ;
    }

    // enforce PDE constraints
    m_subProblems[i].model->evalModel((*m_subProblems[i].params_in),
                                    (*m_subProblems[i].responses_out));
  }

  if ( m_entityType == "Distributed Parameter") {
    m_objAggregator->SetInputVariables(m_subProblems, m_responseMap, m_responseDerivMap);
  }
  m_objAggregator->Evaluate();
  copyObjectiveFromStateMgr( g, dgdp) ;

  // See if the user specified a new design frequency.
  GO new_frequency = -1;
  if (m_solverComm->getRank() == 0) {
    std::ifstream file("update_frequency.txt");
    if (file.is_open()) {
      file >> new_frequency;
    }
  }
  Teuchos::broadcast(*m_solverComm, 0, 1, &new_frequency);

  if (new_frequency != -1) {
    // the user has specified a new frequency to use
    m_writeDesignFrequency = new_frequency;
  }

  // Output a new result file if requested
  if (m_writeDesignFrequency && (m_iteration % m_writeDesignFrequency == 0)) {
     writeCurrentDesign();
  }
  m_iteration++;
}

//**********************************************************************
void Solver::writeCurrentDesign()
//**********************************************************************
{
#ifdef ATO_USES_ISOLIB
  auto disc = m_subProblems[0].app->getDiscretization();

  // The cast must succeed.
  auto stkDisc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc,true);

  MPI_Comm mpi_comm = Albany::getMpiCommFromTeuchosComm(*m_solverComm);
  iso::STKExtract ex;
  ex.create_mesh_apis_Albany(&mpi_comm,
             &(stkDisc->getSTKBulkData()),
             &(stkDisc->getSTKMetaData()),
             "", "iso.exo", "Rho_node", 1e-5, 0.5,
              0, 0, 1, 0);
  ex.run_Albany(m_iteration);
#else
  TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
         << "Error! Albany must be compiled with IsoLib support for runtime output." << std::endl);
#endif
}

//**********************************************************************
void Solver::copyTopologyIntoParameter (const double* p, SolverSubSolver& subSolver) 
//**********************************************************************
{
  Teuchos::RCP<Albany::Application> app = subSolver.app;
  Albany::StateManager& stateMgr = app->getStateMgr();

  Teuchos::RCP<Albany::DistributedParameterLibrary> distParams = app->getDistributedParameterLibrary();

  const Albany::WorksetArray<std::string>::type& wsEBNames = stateMgr.getDiscretization()->getWsEBNames();

  auto indexer = Albany::createGlobalLocalIndexer(m_localNodeVS);

  int ntopos = m_topologyInfoStructs.size();
  for (int itopo=0; itopo<ntopos; ++itopo) {
    Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
    Teuchos::RCP<Topology> topology = topoStruct->topology;
    const Teuchos::Array<std::string>& fixedBlocks = topology->getFixedBlocks();

    const auto& wsElDofs = distParams->get(topology->getName())->workset_elem_dofs();
    const auto& wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

    // enforce fixed blocks
    Teuchos::RCP<Thyra_Vector> topoVec = topoStruct->localVector;
    Teuchos::ArrayRCP<double> ltopo = Albany::getNonconstLocalData(topoVec);
    const int numMyNodes = ltopo.size();
    for (int i=0; i<numMyNodes; ++i) {
      ltopo[i] = p[i];
    }
  
    smoothTopology(topoStruct);
  
    const int numWorksets = wsElDofs.size();
    const double matVal = topology->getMaterialValue();
    for (int ws=0; ws<numWorksets; ++ws) {
      const auto it = std::find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]);
      if (it != fixedBlocks.end()) {
        const auto& elNodeID = wsElNodeID[ws];
        const int numCells = elNodeID.size();
        const int numNodes = elNodeID[0].size();
        for (int cell=0; cell<numCells; ++cell) {
          for (int node=0; node<numNodes; ++node) {
            const int gid = wsElNodeID[ws][cell][node];
            if (indexer->isLocallyOwnedElement(gid)) {
              const int lid = indexer->getLocalElement(gid);
              ltopo[lid] = matVal;
            }
          }
        }
      }
    }

    // save topology to nodal data for output sake
    auto nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();

    Teuchos::RCP<Thyra_Vector> overlapTopoVec = topoStruct->overlapVector;

    // JR: fix this.  you don't need to do this every time.  Just once at setup, after topoVec is built
    const int distParamIndex = subSolver.params_in->Np()-1;
    subSolver.params_in->set_p(distParamIndex,topoVec);
  
    m_cas_manager->scatter(topoVec,overlapTopoVec,Albany::CombineMode::INSERT);
    std::string nodal_topoName = topology->getName()+"_node";
    (*nodeContainer)[nodal_topoName]->saveFieldVector(overlapTopoVec,/*offset=*/0);
  }
}

//**********************************************************************
void Solver::copyTopologyFromStateMgr(double* p, Albany::StateManager& stateMgr) 
//**********************************************************************
{
  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& src = stateArrays.elemStateArrays;
  const int numWorksets = src.size();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const auto& wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

  // copy the topology from the state manager
  auto indexer = Albany::createGlobalLocalIndexer(m_localNodeVS);
  const int ntopos = m_topologyInfoStructs.size();
  for (int itopo=0; itopo<ntopos; ++itopo) {
    Teuchos::RCP<TopologyInfoStruct> topologyInfoStruct = m_topologyInfoStructs[itopo];
    const int numLocalNodes = Albany::getSpmdVectorSpace(topologyInfoStruct->localVector->range())->localSubDim();
    Teuchos::RCP<Topology> topology = topologyInfoStruct->topology;
    const int offset = itopo*numLocalNodes;
    for (int ws=0; ws<numWorksets; ++ws) {
      const Albany::MDArray& wsTopo = src[ws][topology->getName()+"_node"];
      const int numCells = wsTopo.dimension(0);
      const int numNodes = wsTopo.dimension(1);
      for (int cell=0; cell<numCells; ++cell) {
        for (int node=0; node<numNodes; ++node) {
          const int gid = wsElNodeID[ws][cell][node];
          if (indexer->isLocallyOwnedElement(gid)) {
            const int lid = indexer->getLocalElement(gid);
            p[lid+offset] = wsTopo(cell,node);
          }
        }
      }
    }
  }
}

//**********************************************************************
void Solver::smoothTopology(double* p)
//**********************************************************************
{
  // copy topology into Tpetra_Vectors to apply the filter and/or communicate boundary data
  const int ntopos = m_topologyInfoStructs.size();
  
  for (int itopo=0; itopo<ntopos; ++itopo) {
    Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
    Teuchos::RCP<Thyra_Vector> topoVec = topoStruct->localVector;
    Teuchos::ArrayRCP<double> ltopo = Albany::getNonconstLocalData(topoVec);
    const int numLocalNodes = ltopo.size();
    const int offset = itopo*numLocalNodes;
    p += offset;
    for (int lid=0; lid<numLocalNodes; ++lid) {
      ltopo[lid] = p[lid];
    }

    smoothTopology(topoStruct);

    // copy the topology back from the tpetra vectors
    for (int lid=0; lid<numLocalNodes; ++lid) {
      p[lid] = ltopo[lid];
    }
  }
}

//**********************************************************************
void Solver::smoothTopology(Teuchos::RCP<TopologyInfoStruct> topoStruct)
//**********************************************************************
{
  // apply filter if requested
  if (topoStruct->filter != Teuchos::null) {
    // The apply method does not allow the input to alias the output,
    // so in order to do x=A*x we need to do x=A*x_old;x_old=x
    Teuchos::RCP<Thyra_Vector> filtered_topoVec_old = Thyra::createMember(*m_localNodeVS);
    Teuchos::RCP<Thyra_Vector> filtered_topoVec = topoStruct->localVector;
    const int numIters = topoStruct->filter->getNumIterations();

    for (int i=0; i<numIters; ++i) {
      // Swap x and x_old
      std::swap(filtered_topoVec,filtered_topoVec_old);

      // Apply filter to x_old
      topoStruct->filter->getFilterOperator()->apply(Thyra::NOTRANS, *filtered_topoVec_old, filtered_topoVec.ptr(), 1.0, 0.0);
    }
    // Set topoVec to the filtered solution
    // Note: swapping the two rcps is probably easier, but there may be
    //       other places storing an RCP to the vector, which would no
    //       longer be pointing to the same object.
    topoStruct->localVector->assign(*filtered_topoVec);
  } else if (topoStruct->postFilter != Teuchos::null) {
    Teuchos::RCP<Thyra_Vector> topoVec = topoStruct->localVector;
    Teuchos::RCP<Thyra_Vector> filteredTopoVec = topoStruct->filteredVector;
    Teuchos::RCP<Thyra_Vector> filteredOTopoVec = topoStruct->filteredOverlapVector;
    topoStruct->postFilter->getFilterOperator()->apply(Thyra::NOTRANS,*topoVec,filteredTopoVec.ptr(), 1.0, 0.0);
    m_cas_manager->scatter(filteredTopoVec,filteredOTopoVec,Albany::CombineMode::INSERT);
  }
}

//**********************************************************************
void Solver::copyTopologyIntoStateMgr( const double* p, Albany::StateManager& stateMgr) 
//**********************************************************************
{
  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  const int numWorksets = dest.size();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const auto& wsEBNames  = disc->getWsEBNames();
  const auto& wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

  auto indexer = Albany::createGlobalLocalIndexer(m_localNodeVS);
  const int ntopos = m_topologyInfoStructs.size();
  for (int itopo=0; itopo<ntopos; ++itopo) {
    Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
    Teuchos::RCP<Topology> topology = topoStruct->topology;
    const Teuchos::Array<std::string>& fixedBlocks = topology->getFixedBlocks();

    // copy topology into Epetra_Vector to apply the filter and/or communicate boundary data
    Teuchos::RCP<Thyra_Vector> topoVec = topoStruct->localVector;
    Teuchos::ArrayRCP<double> ltopo = Albany::getNonconstLocalData(topoVec);
    const int numLocalNodes = ltopo.size();
    const int offset = itopo*numLocalNodes;
    for (int lid=0; lid<numLocalNodes; ++lid) {
      ltopo[lid] = p[lid+offset];
    }

    smoothTopology(topoStruct);

    Teuchos::RCP<Thyra_Vector> overlapTopoVec = topoStruct->overlapVector;
    m_cas_manager->scatter(topoVec,overlapTopoVec,Albany::CombineMode::INSERT);

    // copy the topology into the state manager
    Teuchos::ArrayRCP<double> otopo = Albany::getNonconstLocalData(overlapTopoVec);
    const double matVal = topology->getMaterialValue();
    for (int ws=0; ws<numWorksets; ++ws) {
      Albany::MDArray& wsTopo = dest[ws][topology->getName()];
      const int numCells = wsTopo.dimension(0);
      const int numNodes = wsTopo.dimension(1);
      for (int cell=0; cell<numCells; ++cell) {
        for (int node=0; node<numNodes; ++node) {
          int gid = wsElNodeID[ws][cell][node];
          int lid = indexer->getLocalElement(gid);
          wsTopo(cell,node) = otopo[lid];
        }
      }
    }

    auto overlapFreeNodeMask = Thyra::createMember(*m_overlapNodeVS);
    auto localFreeNodeMask   = Thyra::createMember(*m_localNodeVS);
    overlapFreeNodeMask->assign(0.0);
    Teuchos::ArrayRCP<double> fMask = Albany::getNonconstLocalData(overlapFreeNodeMask);
    for (int ws=0; ws<numWorksets; ++ws) {
      const Albany::MDArray& wsTopo = dest[ws][topology->getName()];
      const int numCells = wsTopo.dimension(0);
      const int numNodes = wsTopo.dimension(1);
      const auto it = std::find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]);
      if (it == fixedBlocks.end()) {
        for (int cell=0; cell<numCells; ++cell) {
          for (int node=0; node<numNodes; ++node) {
            int gid = wsElNodeID[ws][cell][node];
            int lid = indexer->getLocalElement(gid);
            fMask[lid] = 1.0;
          }
        }
      } else {
        for (int cell=0; cell<numCells; ++cell) {
          for (int node=0; node<numNodes; ++node) {
            wsTopo(cell,node) = matVal;
          }
        }
      }
    }
    localFreeNodeMask->assign(1.0);
    m_cas_manager->combine(overlapFreeNodeMask,localFreeNodeMask,Albany::CombineMode::ABSMAX);
    m_cas_manager->scatter(localFreeNodeMask,overlapFreeNodeMask,Albany::CombineMode::INSERT);
  
    // if it is a fixed block, set the topology variable to the material value
    for (int ws=0; ws<numWorksets; ++ws) {
      const Albany::MDArray& wsTopo = dest[ws][topology->getName()];
      const int numCells = wsTopo.dimension(0), numNodes = wsTopo.dimension(1);
      for (int cell=0; cell<numCells; ++cell) {
        for (int node=0; node<numNodes; ++node) {
          const int gid = wsElNodeID[ws][cell][node];
          const int lid = indexer->getLocalElement(gid);
          if (fMask[lid] != 1.0) {
            otopo[lid] = matVal;
          }
        }
      }
    }

    // save topology to nodal data for output sake
    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();

    std::string nodal_topoName = topology->getName()+"_node";
    (*nodeContainer)[nodal_topoName]->saveFieldVector(overlapTopoVec,/*offset=*/0);

    if (topoStruct->postFilter != Teuchos::null) {
      nodal_topoName = topology->getName()+"_node_filtered";
      Teuchos::RCP<Thyra_Vector> filteredOTopoVec = topoStruct->filteredOverlapVector;
      (*nodeContainer)[nodal_topoName]->saveFieldVector(filteredOTopoVec,/*offset=*/0);
    }
  }
}

//**********************************************************************
void Solver::copyConstraintFromStateMgr( double& c, double* dcdp) 
//**********************************************************************
{
  c = *m_constraintValue;
  const int nVecs = m_ConstraintGradientVec.size();
  for (int ivec=0; ivec<nVecs; ++ivec) {
    if (m_entityType == "State Variable") {
      m_ConstraintGradientVec[ivec]->assign(0.0);
      m_cas_manager->combine(m_overlapConstraintGradientVec[ivec],m_ConstraintGradientVec[ivec],Albany::CombineMode::ADD);
    }

    if (dcdp != nullptr) {
      const Teuchos::ArrayRCP<const double> lvec = Albany::getLocalData(m_ConstraintGradientVec[ivec].getConst());
      const int numLocalNodes = lvec.size();
      std::memcpy((void*)(dcdp+ivec*numLocalNodes), lvec.getRawPtr(), numLocalNodes*sizeof(double));
    }
  }
}

//**********************************************************************
void Solver::copyObjectiveFromStateMgr( double& g, double* dgdp) 
//**********************************************************************
{
  // aggregated objective derivative is stored in the first subproblem
  Albany::StateManager& stateMgr = m_subProblems[0].app->getStateMgr();

  g = *m_objectiveValue;

  const int nVecs = m_ObjectiveGradientVec.size();
  for (int ivec=0; ivec<nVecs; ++ivec) {

    if (m_entityType == "State Variable") {
      m_ObjectiveGradientVec[ivec]->assign(0.0);
      m_cas_manager->combine(m_overlapObjectiveGradientVec[ivec],m_ObjectiveGradientVec[ivec],Albany::CombineMode::ADD);
    }

    // apply filter if requested
    Teuchos::RCP<Thyra_Vector> filtered_ObjectiveGradientVec = m_ObjectiveGradientVec[ivec];
    if (m_derivativeFilter != Teuchos::null) {
      // The apply method does not allow the input to alias the output,
      // so in order to do x=A*x we need to do x=A*x_old;x_old=x
      Teuchos::RCP<Thyra_Vector> filtered_ObjectiveGradientVec_old = Thyra::createMember(*m_ObjectiveGradientVec[ivec]->space());
      const int numIters = m_derivativeFilter->getNumIterations();
      for (int i=0; i<numIters; ++i) {
        // Swap x and x_old
        std::swap(filtered_ObjectiveGradientVec,filtered_ObjectiveGradientVec_old);

        // Apply the filter to x_old
        m_derivativeFilter->getFilterOperator()->apply(Thyra::TRANS,
                                                       *filtered_ObjectiveGradientVec_old,
                                                       filtered_ObjectiveGradientVec.ptr(),
                                                       1.0, 0.0);
      }
      // Set the objective gradient to the filtered one.
      // Note: swapping the two rcps is probably easier, but are
      //       other places storing an RCP to the vector, which would no
      //       longer be pointing to the same object.
      m_ObjectiveGradientVec[ivec]->assign(*filtered_ObjectiveGradientVec);
      Teuchos::ArrayRCP<const double> lvec = Albany::getLocalData(m_ObjectiveGradientVec[ivec].getConst());
      const int numLocalNodes = lvec.size();
      std::memcpy((void*)(dgdp+ivec*numLocalNodes), lvec.getRawPtr(), numLocalNodes*sizeof(double));
    } else {
      const Teuchos::ArrayRCP<const double> lvec = Albany::getLocalData(m_ObjectiveGradientVec[ivec].getConst());
      const int numLocalNodes = lvec.size();
      std::memcpy((void*)(dgdp+ivec*numLocalNodes), lvec.getRawPtr(), numLocalNodes*sizeof(double));
    }

    // save dgdp to nodal data for output sake
    m_cas_manager->scatter(filtered_ObjectiveGradientVec,m_overlapObjectiveGradientVec[ivec],Albany::CombineMode::INSERT);
    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();
    std::string nodal_derName = Albany::strint(m_objAggregator->getOutputDerivativeName()+"_node", ivec);
    (*nodeContainer)[nodal_derName]->saveFieldVector(m_overlapObjectiveGradientVec[ivec],/*offset=*/0);
  }
}

//**********************************************************************
void Solver::ComputeMeasure(const std::string& measureType, double& measure)
//**********************************************************************
{
  m_atoProblem->ComputeMeasure(measureType, measure);
}

//**********************************************************************
void Solver::ComputeMeasure(const std::string& measureType, const double* p, 
                            double& measure, double* dmdp, 
                            const std::string& integrationMethod)
//**********************************************************************
{
  // communicate boundary topo data
  Albany::StateManager& stateMgr = m_subProblems[0].app->getStateMgr();
  
  const auto& wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

  const int numWorksets = wsElNodeID.size();
  const int ntopos = m_topologyInfoStructs.size();

  std::vector<Teuchos::RCP<TopologyStruct> > topologyStructs(ntopos);
  auto indexer = Albany::createGlobalLocalIndexer(m_localNodeVS);

  for (int itopo=0; itopo<ntopos; ++itopo) {

    topologyStructs[itopo] = Teuchos::rcp(new TopologyStruct());
  
    Teuchos::RCP<Thyra_Vector> topoVec = m_topologyInfoStructs[itopo]->localVector;
    Teuchos::ArrayRCP<double> ltopo = Albany::getNonconstLocalData(topoVec);
    const int numLocalNodes = ltopo.size();
    const int offset = itopo*numLocalNodes;
    for (int ws=0; ws<numWorksets; ++ws) {
      const int numCells = wsElNodeID[ws].size();
      const int numNodes = wsElNodeID[ws][0].size();
      for (int cell=0; cell<numCells; ++cell) {
        for (int node=0; node<numNodes; ++node) {
          const int gid = wsElNodeID[ws][cell][node];
          if (indexer->isLocallyOwnedElement(gid)) {
            const int lid = indexer->getLocalElement(gid);
            ltopo[lid] = p[lid+offset];
          }
        }
      }
    }
    Teuchos::RCP<TopologyInfoStruct> topoStruct = m_topologyInfoStructs[itopo];
    smoothTopology(topoStruct);

    Teuchos::RCP<Thyra_Vector> overlapTopoVec = m_topologyInfoStructs[itopo]->overlapVector;
    m_cas_manager->scatter(topoVec,overlapTopoVec,Albany::CombineMode::INSERT);

    topologyStructs[itopo]->topology = m_topologyInfoStructs[itopo]->topology; 
    topologyStructs[itopo]->dataVector = overlapTopoVec;
  }

  m_atoProblem->ComputeMeasure(measureType, topologyStructs, 
                                measure, dmdp, integrationMethod);
}

//**********************************************************************
void Solver::ComputeVolume(double* p, const double* dfdp, 
                           double& v, double threshhold, double minP)
//**********************************************************************
{
  /*  Assumptions:
      -- dfdp is already consistent across proc boundaries.
      -- the volume computation that's done by the atoProblem updates the topology, p.
      -- Since dfdp is 'boundary consistent', the resulting topology, p, is also
         'boundary consistent', so no communication is necessary.
  */
  m_atoProblem->ComputeVolume(p, dfdp, v, threshhold, minP);
}

//**********************************************************************
void Solver::Compute(double* p, double& g, double* dgdp, double& c, double* dcdp)
//**********************************************************************
{
  Compute((const double*)p, g, dgdp, c, dcdp);
}

//**********************************************************************
void Solver::Compute(const double* p, double& g, double* dgdp, double& c, double* dcdp)
//**********************************************************************
{
  for (int i=0; i<m_numPhysics; ++i) {
    // copy data from p into each stateManager
    if (m_entityType == "State Variable") {
      Albany::StateManager& stateMgr = m_subProblems[i].app->getStateMgr();
      copyTopologyIntoStateMgr( p, stateMgr) ;
    } else if (m_entityType == "Distributed Parameter") {
      copyTopologyIntoParameter( p, m_subProblems[i]) ;
    }

    // enforce PDE constraints
    m_subProblems[i].model->evalModel(*m_subProblems[i].params_in,
                                      *m_subProblems[i].responses_out);
  }

  if ( m_entityType == "Distributed Parameter") {
    m_objAggregator->SetInputVariables(m_subProblems, m_responseMap, m_responseDerivMap);
  }
  m_objAggregator->Evaluate();
  copyObjectiveFromStateMgr( g, dgdp) ;
  
  if (!m_conAggregator.is_null()) {
    if ( m_entityType == "Distributed Parameter") {
      m_conAggregator->SetInputVariables(m_subProblems, m_responseMap, m_responseDerivMap);
    }
    m_conAggregator->Evaluate();
    copyConstraintFromStateMgr( c, dcdp) ;
  } else {
    c = 0.0;
  }

  m_iteration++;
}

//**********************************************************************
int Solver::GetNumOptDofs() const
//**********************************************************************
{
  auto nVecs = m_ObjectiveGradientVec.size();
  return nVecs*Albany::getSpmdVectorSpace(m_ObjectiveGradientVec[0]->space())->localSubDim();
}

/******************************************************************************/
///*********************** SETUP AND UTILITY FUNCTIONS **********************///
/******************************************************************************/


//**********************************************************************
SolverSubSolver
Solver::CreateSubSolver (const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                         const Teuchos::RCP<const Teuchos_Comm>&    comm,
                         const Teuchos::RCP<const Thyra_Vector>&    initial_guess)
//**********************************************************************
{
  SolverSubSolver ret; //value to return

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "ATO Solver creating solver from " << appParams->name()
       << " parameter list" << std::endl;

  //! Create solver factory, which reads xml input filen
  Albany::SolverFactory slvrfctry(appParams, comm);
  ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, comm, comm, initial_guess);

  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");

  int numParameters = 0;
  if (problemParams.isType<Teuchos::ParameterList>("Parameters")) {
    numParameters = problemParams.sublist("Parameters").get<int>("Number of Parameter Vectors");
  }

  int numResponseSpecs = 0;
  if (problemParams.isType<Teuchos::ParameterList>("Response Functions")) {
    numResponseSpecs = problemParams.sublist("Response Functions").get<int>("Number of Response Vectors");
  }

  bool separateByBlock=false;
  Teuchos::ParameterList& discParams = appParams->sublist("Discretization");
  if (discParams.isType<bool>("Separate Evaluators by Element Block")) {
    separateByBlock = discParams.get<bool>("Separate Evaluators by Element Block");
  }

  int numBlocks=1;
  if (separateByBlock) {
    numBlocks = 
      problemParams.sublist("Configuration").sublist("Element Blocks").get<int>("Number of Element Blocks");
  }

  ret.params_in = Teuchos::rcp(new Thyra_InArgs(ret.model->createInArgs()));
  ret.responses_out = Teuchos::rcp(new Thyra_OutArgs(ret.model->createOutArgs()));

  // the createOutArgs() function doesn't allocate storage
  Teuchos::RCP<Thyra_Vector> g1;
  const int ss_num_g = ret.responses_out->Ng(); // Number of *vectors* of responses
  for (int ig=0; ig<ss_num_g; ++ig) {
    g1 = Thyra::createMember(*ret.model->get_g_space(ig));
    ret.responses_out->set_g(ig,g1);
  }

  const int ss_num_p = ret.params_in->Np();     // Number of *vectors* of parameters
  TEUCHOS_TEST_FOR_EXCEPTION (ss_num_p - numParameters > 1,  Teuchos::Exceptions::InvalidParameter,
                              "Error! Cannot have more than one distributed Parameter for topology optimization.\n");
  for (int ip=0; ip<ss_num_p; ++ip) {
    auto p1 = ret.model->getNominalValues().get_p(ip); 
    ret.params_in->set_p(ip,p1);
  }

  for (int iResponseSpec=0; iResponseSpec<numResponseSpecs; ++iResponseSpec) {
    if (ss_num_p > numParameters) {
      const int ip = ss_num_p-1;
      const Teuchos::ParameterList& resParams = problemParams.sublist("Response Functions").sublist(Albany::strint("Response Vector",iResponseSpec));
      const std::string gName = resParams.get<std::string>("Response Name");
      const std::string dgdpName = resParams.get<std::string>("Response Derivative Name");
 
      std::vector<Teuchos::RCP<const Thyra_Vector>> gVector(numBlocks);
      std::vector<Teuchos::RCP<Thyra_MultiVector>> dgdpVector(numBlocks);
      for (int iBlock=0; iBlock<numBlocks; ++iBlock) {
        const int ig = iResponseSpec*numBlocks + iBlock;
        if (!ret.responses_out->supports(Thyra_ModelEvaluator::OUT_ARG_DgDp, ig, ip).none()) {
          const auto p_space = ret.model->get_p_space(ip);
          gVector[iBlock]  = ret.responses_out->get_g(ig);

          dgdpVector[iBlock] = Thyra::createMembers(*p_space,gVector[iBlock]->space()->dim());

          if (ret.responses_out->supports(OUT_ARG_DgDp,ig,ip).supports(DERIV_TRANS_MV_BY_ROW)) {
            Thyra_Derivative dgdp_out(dgdpVector[iBlock], DERIV_TRANS_MV_BY_ROW);
            ret.responses_out->set_DgDp(ig,ip,dgdp_out);
          } else {
            ret.responses_out->set_DgDp(ig,ip,dgdpVector[iBlock]);
          }
        }
      }
      m_responseMap.emplace(gName,gVector);
      m_responseDerivMap.emplace(dgdpName,dgdpVector);
    }
  }

  auto x_final = Thyra::createMember(*ret.model->get_g_space(ss_num_g-1));
  x_final->assign(0.0);
  ret.responses_out->set_g(ss_num_g-1,x_final);

  return ret;
}

//**********************************************************************
Teuchos::RCP<Teuchos::ParameterList> 
Solver::createInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams, int physIndex) const
//**********************************************************************
{   
  ///*** CREATE INPUt FILE FOR SUBPROBLEM: ***///

  // Get physics (pde) problem sublist, i.e., Physics Problem N, where N = physIndex.
  std::string phys_pl_name = Albany::strint("Physics Problem",physIndex);
  Teuchos::ParameterList& physics_subList = appParams->sublist("Problem").sublist(phys_pl_name,false);

  // Create input parameter list for physics app which mimics a separate input file
  std::string app_pl_name = Albany::strint("Parameters for Subapplication",physIndex);
  Teuchos::RCP<Teuchos::ParameterList> physics_appParams = Teuchos::createParameterList(app_pl_name);

  // get reference to Problem ParameterList in new input file and initialize it 
  // from Parameters in Physics Problem N.
  Teuchos::ParameterList& physics_probParams = physics_appParams->sublist("Problem",false);
  physics_probParams.setParameters(physics_subList);

  // Add topology information
  physics_probParams.set<Teuchos::RCP<TopologyArray> >("Topologies",m_topologyArray);

  Teuchos::ParameterList& topoParams = appParams->sublist("Problem").get<Teuchos::ParameterList>("Topologies");
  physics_probParams.set<Teuchos::ParameterList>("Topologies Parameters",topoParams);

  // Check topology.  If the topology is a distributed parameter, then 1) check for existing 
  // "Distributed Parameter" list and error out if found, and 2) add a "Distributed Parameter" 
  // list to the input file, 
  if (m_entityType == "Distributed Parameter") {
    TEUCHOS_TEST_FOR_EXCEPTION (physics_subList.isType<Teuchos::ParameterList>("Distributed Parameters"),
                                Teuchos::Exceptions::InvalidParameter,
                                "Error! Cannot have 'Distributed Parameters' in both Topology and subproblems.\n");
    Teuchos::ParameterList distParams;
    const int ntopos = m_topologyInfoStructs.size();
    distParams.set("Number of Parameter Vectors",ntopos);
    for (int itopo=0; itopo<ntopos; ++itopo) {
      distParams.set(Albany::strint("Parameter",itopo), m_topologyInfoStructs[itopo]->topology->getName());
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

  physics_probParams.set<bool>("Overwrite Nominal Values With Final Point", true);

  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& physics_discList = physics_appParams->sublist("Discretization", false);
  physics_discList.setParameters(discList);
  // find the output file name and append "Physics_n_" to it. This only checks for exodus output.
  if (physics_discList.isType<std::string>("Exodus Output File Name")) {
    std::stringstream newname;
    newname << "physics_" << physIndex << "_" 
            << physics_discList.get<std::string>("Exodus Output File Name");
    physics_discList.set("Exodus Output File Name",newname.str());
  }

  int ntopos = m_topologyInfoStructs.size();
  for (int itopo=0; itopo<ntopos; ++itopo) {
    if (m_topologyInfoStructs[itopo]->topology->getFixedBlocks().size() > 0) {
      physics_discList.set("Separate Evaluators by Element Block", true);
      break;
    }
  }

  if (m_writeDesignFrequency != 0) 
    physics_discList.set("Use Automatic Aura", true);

  // Piro sublist processing
  physics_appParams->set("Piro",appParams->sublist("Piro"));

  ///*** VERIFY SUBPROBLEM: ***///

  // extract physics and dimension of the subproblem
  Teuchos::ParameterList& subProblemParams = appParams->sublist("Problem").sublist(phys_pl_name);
  std::string problemName = subProblemParams.get<std::string>("Name");
  // "xD" where x = 1, 2, or 3
  std::string problemDimStr = problemName.substr( problemName.length()-2) ;
  //remove " xD" where x = 1, 2, or 3
  std::string problemNameBase = problemName.substr( 0, problemName.length()-3) ;
  
  //// check dimensions
  int numDimensions = 0;
  if (problemDimStr == "1D") {
    numDimensions = 1;
  } else if (problemDimStr == "2D") {
    numDimensions = 2;
  } else if (problemDimStr == "3D") {
    numDimensions = 3;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                "Error! Cannot extract dimension from problem name: " + problemName + ".\n");
  }
  TEUCHOS_TEST_FOR_EXCEPTION (numDimensions == 1, Teuchos::Exceptions::InvalidParameter,
                              "Error!  Topology optimization is not avaliable in 1D.\n");

  return physics_appParams;
}

//**********************************************************************
Teuchos::RCP<Teuchos::ParameterList> 
Solver::createHomogenizationInputFile( 
    const Teuchos::RCP<Teuchos::ParameterList>& appParams, 
    const Teuchos::ParameterList& homog_subList, 
    int homogProblemIndex, 
    int homogSubIndex, 
    int homogDim) const
//**********************************************************************
{   
  const Teuchos::ParameterList& homog_problem_subList = homog_subList.sublist("Problem");

  // Create input parameter list for app which mimics a separate input file
  std::string app_pl_name = Albany::strint("Parameters for Homogenization Subapplication",homogSubIndex);
  Teuchos::RCP<Teuchos::ParameterList> homog_appParams = Teuchos::createParameterList(app_pl_name);

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
  if (dofsType == "Scalar") {
    isVector = false;
    TEUCHOS_TEST_FOR_EXCEPTION(dofs.size() != 1, Teuchos::Exceptions::InvalidParameter, 
                               std::endl << "Error: Expected DOF Names array to be length 1." << std::endl);
  } else if (dofsType == "Vector") {
    isVector = true;
    TEUCHOS_TEST_FOR_EXCEPTION(dofs.size() != homogDim, Teuchos::Exceptions::InvalidParameter, 
                               std::endl << "Error: Expected DOF Names array to be length " << homogDim << ".\n");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               "Homogenization DOFs must be of type Scalar or Vector (not " << dofsType << ").\n");
  }

  if (homogDim == 1) {
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
    std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
  } else if (homogDim == 2) {
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    int negY = bcIdParams.get<int>("Negative Y Face");
    int posY = bcIdParams.get<int>("Positive Y Face");
    if (homogSubIndex < 2) {
      std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
      std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      if (isVector) {
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[1]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[1]; Params.set(namePosY.str(),0.0);
      } else {
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
      }
    } else {
      std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
      std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
      if (isVector) {
        std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[1]; Params.set(nameNegX.str(),0.0);
        std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[1]; Params.set(namePosX.str(),0.0);
      } else {
        std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
        std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      }
    }
  } else if (homogDim == 3) {
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    int negY = bcIdParams.get<int>("Negative Y Face");
    int posY = bcIdParams.get<int>("Positive Y Face");
    int negZ = bcIdParams.get<int>("Negative Z Face");
    int posZ = bcIdParams.get<int>("Positive Z Face");
    if (homogSubIndex < 3) {
      std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
      std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      if (isVector) {
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
      if (isVector) {
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
  if (homog_discList.isType<std::string>("Exodus Output File Name")) {
    std::stringstream newname;
    newname << "homog_" << homogProblemIndex << "_" << homogSubIndex << "_" 
            << homog_discList.get<std::string>("Exodus Output File Name");
    homog_discList.set("Exodus Output File Name",newname.str());
  }

  // Piro sublist processing
  homog_appParams->set("Piro",appParams->sublist("Piro"));
  
  return homog_appParams;
}

//**********************************************************************
Teuchos::RCP<const Teuchos::ParameterList>
Solver::getValidProblemParameters() const
//**********************************************************************
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList("ValidTopologicalOptimizationProblemParams");

  // Basic set-up
  validPL->set<int>("Number of Subproblems", 1, "Number of PDE constraint problems");
  validPL->set<int>("Number of Homogenization Problems", 0, "Number of homogenization problems");
  validPL->set<bool>("Verbose Output", false, "Enable detailed output mode");
  validPL->set<int>("Design Output Frequency", 0, "Write isosurface every N iterations");
  validPL->set<std::string>("Name", "", "String to designate Problem");

  // Specify physics problem(s)
  for (int i=0; i<m_numPhysics; ++i) {
    std::stringstream physStream; physStream << "Physics Problem " << i;
    validPL->sublist(physStream.str(), false, "");
  }
  
  int numHomogProblems = m_homogenizationSets.size();
  for (int i=0; i<numHomogProblems; ++i) {
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

//**********************************************************************
Thyra_InArgs Solver::createInArgs() const
//**********************************************************************
{
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.set_Np(m_num_parameters);
  return static_cast<Thyra_InArgs>(inArgs);
}

//**********************************************************************
Thyra_OutArgs Solver::createOutArgsImpl() const
//**********************************************************************
{
  Thyra::ModelEvaluatorBase::OutArgsSetup<ST> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.set_Np_Ng(m_num_parameters, m_num_responses);
  return static_cast<Thyra_OutArgs>(outArgs);
}

//**********************************************************************
Teuchos::RCP<const Thyra_VectorSpace> Solver::get_g_space(int j) const
//**********************************************************************
{
  TEUCHOS_TEST_FOR_EXCEPTION (j<0 || j>m_num_responses, Teuchos::Exceptions::InvalidParameter,
                              "Error in Solver::get_g_space(): Invalid response index j = " << j << std::endl);

  return m_x_vs;
}

//**********************************************************************
SolverSubSolverData
Solver::CreateSubSolverData(const SolverSubSolver& sub) const
//**********************************************************************
{
  SolverSubSolverData ret;
  if (sub.params_in->Np() > 0 && sub.responses_out->Ng() > 0) {
    ret.deriv_support = sub.model->createOutArgs().supports(OUT_ARG_DgDp, 0, 0);
  } else {
    ret.deriv_support = Thyra_DerivativeSupport();
  }

  ret.Np = sub.params_in->Np();
  ret.pLength = std::vector<int>(ret.Np);
  for (int i=0; i<ret.Np; ++i) {
    const auto p_space = sub.params_in->get_p(i)->space();
    //uses local length (need to modify to work with distributed params)
    if (p_space != Teuchos::null) {
      ret.pLength[i] = Albany::getSpmdVectorSpace(p_space)->localSubDim();
    } else {
      ret.pLength[i] = 0;
    }
  }

  ret.Ng = sub.responses_out->Ng();
  ret.gLength = std::vector<int>(ret.Ng);
  for (int i=0; i<ret.Ng; ++i) {
    const auto g_space = sub.responses_out->get_g(i)->space();
    //uses local length (need to modify to work with distributed responses)
    if (g_space != Teuchos::null) {
      ret.gLength[i] = Albany::getSpmdVectorSpace(g_space)->localSubDim();
    } else {
      ret.gLength[i] = 0;
    }
  }

  if (ret.Np > 0) {
    //only first p vector used - in the future could make ret.p_init an array of Np vectors
    ret.p_init = sub.model->getNominalValues().get_p(0);
  }

  return ret;
}

} // namespace ATO
