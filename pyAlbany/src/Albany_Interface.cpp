//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Interface.hpp"

#include <iostream>
#include <string>

#include "Albany_Memory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_RegressionTests.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Albany_FactoriesHelpers.hpp"

#include "Piro_PerformSolve.hpp"
#include "Piro_PerformAnalysis.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_MultiVectorStdOps.hpp"

#include "Albany_TpetraThyraUtils.hpp"

#include "Albany_ObserverImpl.hpp"

#include "ROL_Types.hpp"

#if defined(ALBANY_CHECK_FPE) || defined(ALBANY_STRONG_FPE_CHECK) || defined(ALBANY_FLUSH_DENORMALS)
#include <xmmintrin.h>
#endif

#if defined(ALBANY_CHECK_FPE) || defined(ALBANY_STRONG_FPE_CHECK)
#include <cmath>
#endif

#if defined(ALBANY_FLUSH_DENORMALS)
#include <pmmintrin.h>
#endif

#include "Albany_DataTypes.hpp"

#include "Phalanx_config.hpp"

using namespace PyAlbany;

using Teuchos::RCP;
using Teuchos::rcp;

PyParallelEnv::PyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads, int _num_numa, int _device_id) : comm(_comm), num_threads(_num_threads), num_numa(_num_numa), device_id(_device_id)
{
    Kokkos::InitArguments args;
    args.num_threads = this->num_threads;
    args.num_numa = this->num_numa;
    args.device_id = this->device_id;

    if(!Kokkos::is_initialized())
        Kokkos::initialize(args);

    rank = comm->getRank();
}

PyProblem::PyProblem(std::string filename, Teuchos::RCP<PyParallelEnv> _pyParallelEnv) : pyParallelEnv(_pyParallelEnv)
{

    RCP<Teuchos::FancyOStream> out(
        Teuchos::VerboseObjectBase::getDefaultOStream());

    PrintPyHeader(*out);

    stackedTimer = Teuchos::rcp(
        new Teuchos::StackedTimer("PyAlbany Total Time"));
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);

    stackedTimer->start("PyAlbany: Setup Time");

    comm = this->pyParallelEnv->getComm();

    slvrfctry = rcp(new Albany::SolverFactory(filename, comm));

    auto const &bt = slvrfctry->getParameters()->get<std::string>("Build Type", "NONE");

    if (bt == "Tpetra")
    {
        // Set the static variable that denotes this as a Tpetra run
        static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));
    }
    else
    {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
                                   "Error! Invalid choice (" + bt + ") for 'BuildType'.\n"
                                                                    "       The only valid choice for PyAlbany is 'Tpetra'.\n");
    }

    // Make sure all the pb factories are registered *before* the Application
    // is created (since in the App ctor the pb factories are queried)
    Albany::register_pb_factories();

    // Create app (null initial guess)
    albanyApp = slvrfctry->createApplication(comm);
    albanyModel = slvrfctry->createModel(albanyApp);
    solver = slvrfctry->createSolver(comm, albanyModel, Teuchos::null);

    thyraDirections.resize(solver->Np());
    thyraParameter.resize(solver->Np());

    forwardHasBeenSolved = false;
    inverseHasBeenSolved = false;

    stackedTimer->stop("PyAlbany: Setup Time");
    stackedTimer->stopBaseTimer();
}

PyProblem::PyProblem(Teuchos::RCP<Teuchos::ParameterList> params, Teuchos::RCP<PyParallelEnv> _pyParallelEnv) : pyParallelEnv(_pyParallelEnv)
{

    RCP<Teuchos::FancyOStream> out(
        Teuchos::VerboseObjectBase::getDefaultOStream());

    PrintPyHeader(*out);

    stackedTimer = Teuchos::rcp(
        new Teuchos::StackedTimer("PyAlbany Total Time"));
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);

    stackedTimer->start("PyAlbany: Setup Time");

    comm = this->pyParallelEnv->getComm();

    slvrfctry = rcp(new Albany::SolverFactory(params, comm));

    auto const &bt = slvrfctry->getParameters()->get<std::string>("Build Type", "NONE");

    if (bt == "Tpetra")
    {
        // Set the static variable that denotes this as a Tpetra run
        static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));
    }
    else
    {
        TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
                                   "Error! Invalid choice (" + bt + ") for 'BuildType'.\n"
                                                                    "       The only valid choice for PyAlbany is 'Tpetra'.\n");
    }

    // Make sure all the pb factories are registered *before* the Application
    // is created (since in the App ctor the pb factories are queried)
    Albany::register_pb_factories();

    // Create app (null initial guess)
    albanyApp = slvrfctry->createApplication(comm);
    albanyModel = slvrfctry->createModel(albanyApp);
    solver = slvrfctry->createSolver(comm, albanyModel, Teuchos::null);

    thyraDirections.resize(solver->Np());
    thyraParameter.resize(solver->Np());

    forwardHasBeenSolved = false;
    inverseHasBeenSolved = false;

    stackedTimer->stop("PyAlbany: Setup Time");
    stackedTimer->stopBaseTimer();
}

Teuchos::RCP<const Tpetra_Map> PyProblem::getResponseMap(const int g_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getResponseMap");
    if (forwardHasBeenSolved == false)
    {
        std::cout << "Warning: getResponseMap() must be called after performSolve()" << std::endl;
        stackedTimer->stop("PyAlbany: getResponseMap");
        stackedTimer->stopBaseTimer();
        return Teuchos::null;
    }
    Teuchos::RCP<const Thyra_Vector> g = thyraResponses[g_index];
    if (Teuchos::nonnull(g))
    {
        auto g_space = g->space();
        stackedTimer->stop("PyAlbany: getResponseMap");
        stackedTimer->stopBaseTimer();
        return Albany::getTpetraMap(g_space);
    }
    stackedTimer->stop("PyAlbany: getResponseMap");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<const Tpetra_Map> PyProblem::getStateMap()
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getStateMap");
    if (forwardHasBeenSolved == false)
    {
        std::cout << "Warning: getStateMap() must be called after performSolve()" << std::endl;
        stackedTimer->stop("PyAlbany: getStateMap");
        stackedTimer->stopBaseTimer();
        return Teuchos::null;
    }
    Teuchos::RCP<const Thyra_Vector> s = thyraResponses.back();
    if (Teuchos::nonnull(s))
    {
        auto s_space = s->space();
        stackedTimer->stop("PyAlbany: getStateMap");
        stackedTimer->stopBaseTimer();
        return Albany::getTpetraMap(s_space);
    }
    stackedTimer->stop("PyAlbany: getStateMap");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<const Tpetra_Map> PyProblem::getParameterMap(const int p_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getParameterMap");
    auto p_space = solver->get_p_space(p_index);
    auto outputMap = Albany::getTpetraMap(p_space);
    stackedTimer->stop("PyAlbany: getParameterMap");
    stackedTimer->stopBaseTimer();
    return outputMap;
}

void PyProblem::setDirections(const int p_index, Teuchos::RCP<Tpetra_MultiVector> direction)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: setDirections");
    forwardHasBeenSolved = false;

    const unsigned int n_directions = direction->getNumVectors();

    for (int l = 0; l < solver->Np(); l++)
    {
        if (p_index == l)
        {
            thyraDirections[l] = Albany::createThyraMultiVector(direction);
            continue;
        }

        bool is_null = Teuchos::is_null(thyraDirections[l]);
        if (is_null)
        {
            auto p_space = solver->getNominalValues().get_p(l)->space();
            thyraDirections[l] = Thyra::createMembers(p_space, n_directions);
            for (size_t i_direction = 0; i_direction < n_directions; i_direction++)
                thyraDirections[l]->col(i_direction)->assign(0.0);
        }
    }

    stackedTimer->stop("PyAlbany: setDirections");
    stackedTimer->stopBaseTimer();
}

void PyProblem::setParameter(const int p_index, Teuchos::RCP<Tpetra_Vector> p)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: setParameter");
    RCP<Teuchos::ParameterList> appParams = slvrfctry->getParameters();
    if (appParams->isSublist("Piro"))
    {
        RCP<Teuchos::ParameterList> piroParams = Teuchos::sublist(appParams, "Piro");
        if (piroParams->isSublist("Optimization Status"))
        {
            RCP<Teuchos::ParameterList> optimizationParams =
                Teuchos::sublist(piroParams, "Optimization Status");
            optimizationParams->set<bool>("Compute State", true);
        }
    }

    forwardHasBeenSolved = false;
    inverseHasBeenSolved = false;

    for (int l = 0; l < solver->Np(); l++)
    {
        if (p_index == l)
        {
            thyraParameter[l] = Albany::createThyraVector(p);
            albanyModel->setNominalValue(l, thyraParameter[l]);
            continue;
        }

        bool is_null = Teuchos::is_null(thyraParameter[l]);
        if (is_null)
        {
            auto p_space = solver->getNominalValues().get_p(l)->space();
            thyraParameter[l] = Thyra::createMember(p_space);
            thyraParameter[l]->assign(0.0);
        }
    }

    stackedTimer->stop("PyAlbany: setParameter");
    stackedTimer->stopBaseTimer();
}

Teuchos::RCP<Tpetra_Vector> PyProblem::getParameter(const int p_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getParameter");
    Teuchos::RCP<Thyra_Vector> p = thyraParameter[p_index];
    Teuchos::RCP<Tpetra_Vector> p_out = Albany::getTpetraVector(p);
    stackedTimer->stop("PyAlbany: getParameter");
    stackedTimer->stopBaseTimer();
    return p_out;
}

Teuchos::RCP<Tpetra_Vector> PyProblem::getResponse(const int g_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getResponse");
    if (forwardHasBeenSolved == false)
    {
        std::cout << "Warning: getResponse() must be called after performSolve()" << std::endl;
    }
    else
    {
        Teuchos::RCP<Thyra_Vector> g = thyraResponses[g_index];
        Teuchos::RCP<Tpetra_Vector> g_out = Albany::getTpetraVector(g);
        stackedTimer->stop("PyAlbany: getResponse");
        stackedTimer->stopBaseTimer();
        return g_out;
    }
    stackedTimer->stop("PyAlbany: getResponse");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<Tpetra_Vector> PyProblem::getState()
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getState");
    if (forwardHasBeenSolved == false)
    {
        std::cout << "Warning: getState() must be called after performSolve()" << std::endl;
    }
    else
    {
        Teuchos::RCP<Thyra_Vector> s = thyraResponses.back();
        Teuchos::RCP<Tpetra_Vector> s_out = Albany::getTpetraVector(s);
        stackedTimer->stop("PyAlbany: getState");
        stackedTimer->stopBaseTimer();
        return s_out;
    }
    stackedTimer->stop("PyAlbany: getState");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<Tpetra_MultiVector> PyProblem::getSensitivity(const int g_index, const int p_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getSensitivity");
    if (forwardHasBeenSolved == false)
    {
        std::cout << "Warning: getSensitivity() must be called after performSolve()" << std::endl;
    }
    else
    {
        Teuchos::RCP<Tpetra_MultiVector> dg_out = Albany::getTpetraMultiVector(thyraSensitivities[g_index][p_index]);
        stackedTimer->stop("PyAlbany: getSensitivity");
        stackedTimer->stopBaseTimer();
        return dg_out;
    }
    stackedTimer->stop("PyAlbany: getSensitivity");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<Tpetra_MultiVector> PyProblem::getReducedHessian(const int g_index, const int p_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getReducedHessian");
    if (forwardHasBeenSolved == false)
    {
        std::cout << "Warning: getReducedHessian() must be called after performSolve()" << std::endl;
    }
    else
    {
        Teuchos::RCP<Tpetra_MultiVector> hv_out = Albany::getTpetraMultiVector(thyraReducedHessian[g_index][p_index]);
        stackedTimer->stop("PyAlbany: getReducedHessian");
        stackedTimer->stopBaseTimer();
        return hv_out;
    }
    stackedTimer->stop("PyAlbany: getReducedHessian");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

bool PyProblem::performSolve()
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: performSolve");

    Teuchos::ParameterList &solveParams =
        slvrfctry->getAnalysisParameters().sublist(
            "Solve", /*mustAlreadyExist =*/false);

    Piro::PerformSolve(
        *solver, solveParams, thyraResponses, thyraSensitivities, thyraDirections, thyraReducedHessian);

    forwardHasBeenSolved = true;

    stackedTimer->stop("PyAlbany: performSolve");
    stackedTimer->stopBaseTimer();
    bool error = (albanyApp->getSolutionStatus() != Albany::Application::SolutionStatus::Converged);
    return error;
}

bool PyProblem::performAnalysis()
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: performAnalysis");

    Teuchos::RCP<Albany::ObserverImpl> observer = Teuchos::rcp(new Albany::ObserverImpl(albanyApp));

    Teuchos::RCP<Thyra::VectorBase<double>> p;

    Teuchos::ParameterList &piroParams =
        slvrfctry->getParameters()->sublist("Piro");

    int status = Piro::PerformAnalysis(*solver, piroParams, p, observer);

    auto p_dpv = Teuchos::rcp_dynamic_cast<Thyra::DefaultProductVector<double>>(p);

    size_t n_params = solver->Np() > p_dpv->productSpace()->numBlocks() ? p_dpv->productSpace()->numBlocks() : solver->Np();
    for (size_t l = 0; l < n_params; l++)
    {
        thyraParameter[l] = p_dpv->getNonconstVectorBlock(l);
        albanyModel->setNominalValue(l, thyraParameter[l]);
    }

    inverseHasBeenSolved = true;

    stackedTimer->stop("PyAlbany: performAnalysis");
    stackedTimer->stopBaseTimer();
    bool error = (status != ROL::EXITSTATUS_CONVERGED && status != ROL::EXITSTATUS_STEPTOL);
    return error;
}

void PyProblem::reportTimers()
{
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = true;
    options.output_minmax = true;
    stackedTimer->report(std::cout, Teuchos::DefaultComm<int>::getComm(), options);
}

Teuchos::RCP<Tpetra_MultiVector> PyAlbany::scatterMVector(Teuchos::RCP<Tpetra_MultiVector> inVector, Teuchos::RCP<const Tpetra_Map> distributedMap)
{
    auto rankZeroMap = getRankZeroMap(distributedMap);
    Teuchos::RCP<Tpetra_Export> exportZero = rcp(new Tpetra_Export(distributedMap, rankZeroMap));
    Teuchos::RCP<Tpetra_MultiVector> outVector = rcp(new Tpetra_MultiVector(distributedMap, inVector->getNumVectors()));
    outVector->doImport(*inVector, *exportZero, Tpetra::INSERT);
    return outVector;
}

Teuchos::RCP<Tpetra_MultiVector> PyAlbany::gatherMVector(Teuchos::RCP<Tpetra_MultiVector> inVector, Teuchos::RCP<const Tpetra_Map> distributedMap)
{
    auto rankZeroMap = getRankZeroMap(distributedMap);
    Teuchos::RCP<Tpetra_Export> exportZero = rcp(new Tpetra_Export(distributedMap, rankZeroMap));
    Teuchos::RCP<Tpetra_MultiVector> outVector = rcp(new Tpetra_MultiVector(rankZeroMap, inVector->getNumVectors()));
    outVector->doExport(*inVector, *exportZero, Tpetra::ADD);
    return outVector;
}

Teuchos::RCP<Teuchos::ParameterList> PyAlbany::getParameterList(std::string inputFile, Teuchos::RCP<PyParallelEnv> pyParallelEnv)
{
    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::createParameterList("Albany Parameters");

    std::string const input_extension = Albany::getFileExtension(inputFile);

    if (input_extension == "yaml" || input_extension == "yml")
    {
        Teuchos::updateParametersFromYamlFileAndBroadcast(
            inputFile, params.ptr(), *(pyParallelEnv->getComm()));
    }
    else
    {
        Teuchos::updateParametersFromXmlFileAndBroadcast(
            inputFile, params.ptr(), *(pyParallelEnv->getComm()));
    }

    return params;
}

void PyAlbany::writeParameterList(std::string outputFile, Teuchos::RCP<Teuchos::ParameterList> parameterList)
{
    std::string const output_extension = Albany::getFileExtension(outputFile);

    if (output_extension == "yaml" || output_extension == "yml")
    {
        Teuchos::writeParameterListToYamlFile(*parameterList, outputFile);
    }
    else
    {
        Teuchos::writeParameterListToXmlFile(*parameterList, outputFile);
    }    
}

void PyAlbany::orthogTpMVecs(Teuchos::RCP<Tpetra_MultiVector> inputVecs, int blkSize)
{
  typedef double                            ScalarType;
  typedef int                               OT;
  typedef typename Teuchos::SerialDenseMatrix<OT,ScalarType> MAT;
  typedef Tpetra::MultiVector<ScalarType>   MV;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  typedef Tpetra::Operator<ScalarType>             OP;
  typedef Belos::OperatorTraits<ScalarType,MV,OP> OPT;
  int numVecs = inputVecs->getNumVectors();
  std::string orthogType("ICGS");

  Teuchos::RCP<MAT> B = Teuchos::rcp(new MAT(blkSize, blkSize)); //Matrix for coeffs of X
  Teuchos::Array<Teuchos::RCP<MAT>> C; 

  Belos::OrthoManagerFactory<ScalarType, MV, OP> factory;
  Teuchos::RCP<Teuchos::ParameterList> paramsOrtho;   // can be null

  //Default OutputManager is std::cout.
  Teuchos::RCP<Belos::OutputManager<ScalarType> > myOutputMgr = Teuchos::rcp( new Belos::OutputManager<ScalarType>() );
  const Teuchos::RCP<Belos::OrthoManager<ScalarType,MV>> orthoMgr = factory.makeOrthoManager (orthogType, Teuchos::null, myOutputMgr, "Tpetra OrthoMgr", paramsOrtho); 
  
  int numLoops = numVecs/blkSize;
  int remainder = numVecs % blkSize;

  Teuchos::RCP<MV> vecBlock = inputVecs->subViewNonConst(Teuchos::Range1D(0,blkSize-1));
  orthoMgr->normalize(*vecBlock, B);
  std::vector<Teuchos::RCP<const MV>> pastVecArray;
  pastVecArray.push_back(vecBlock);
  Teuchos::ArrayView<Teuchos::RCP<const MV>> pastVecArrayView; 

  for(int k=1; k<numLoops; k++){
    pastVecArrayView = arrayViewFromVector(pastVecArray);
    vecBlock = inputVecs->subViewNonConst(Teuchos::Range1D(k*blkSize,k*blkSize + blkSize - 1));
    C.append(rcp(new MAT(blkSize, blkSize)));
    orthoMgr->projectAndNormalize(*vecBlock, C, B, pastVecArrayView);
    pastVecArray.push_back(vecBlock);
  }
  if( remainder > 0){
    pastVecArrayView = arrayViewFromVector(pastVecArray);
    vecBlock = inputVecs->subViewNonConst(Teuchos::Range1D(numVecs-remainder, numVecs-1));
    B = Teuchos::rcp(new MAT(remainder, remainder));
    C.append(Teuchos::rcp(new MAT(remainder, remainder)));
    orthoMgr->projectAndNormalize(*vecBlock, C, B, pastVecArrayView);
  }
}

void PyAlbany::finalizeKokkos()
{
    if(Kokkos::is_initialized())
        Kokkos::finalize_all();
}

Teuchos::RCP<const Tpetra_Map> PyAlbany::getRankZeroMap(Teuchos::RCP<const Tpetra_Map> distributedMap)
{
    int numGlobalElements = distributedMap->getGlobalNumElements();
    int numLocalElements = distributedMap->getLocalNumElements();

    // GO and Tpetra_Map::global_ordinal_type can be different
    auto nodes_gids_view = distributedMap->getMyGlobalIndices();
    Teuchos::Array<GO> nodes_gids(numLocalElements);
    for (int i=0; i<numLocalElements; ++i)
        nodes_gids[i] = nodes_gids_view(i);
    Teuchos::Array<GO> all_nodes_gids;
    Albany::gatherV(distributedMap->getComm(),nodes_gids(),all_nodes_gids,0);
    std::sort(all_nodes_gids.begin(),all_nodes_gids.end());
    auto it = std::unique(all_nodes_gids.begin(),all_nodes_gids.end());
    all_nodes_gids.erase(it,all_nodes_gids.end());

    Teuchos::Array<Tpetra_Map::global_ordinal_type> all_nodes_py_gids(all_nodes_gids.size());
    for (int i=0; i<all_nodes_gids.size(); ++i)
        all_nodes_py_gids[i] = all_nodes_gids[i];

    return rcp(new Tpetra_Map(numGlobalElements, all_nodes_py_gids, distributedMap->getIndexBase(), distributedMap->getComm()));
}