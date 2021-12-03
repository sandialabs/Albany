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

PyParallelEnv::PyParallelEnv(Teuchos::RCP<Teuchos::Comm<int>> _comm, int _num_threads, int _num_numa, int _device_id) : comm(_comm), num_threads(_num_threads), num_numa(_num_numa), device_id(_device_id)
{
    Kokkos::InitArguments args;
    args.num_threads = this->num_threads;
    args.num_numa = this->num_numa;
    args.device_id = this->device_id;

    Kokkos::initialize(args);
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

    comm = this->pyParallelEnv->comm;

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

    comm = this->pyParallelEnv->comm;

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

Teuchos::RCP<const PyTrilinosMap> PyProblem::getResponseMap(const int g_index)
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
        return getPyTrilinosMap(Albany::getTpetraMap(g_space), false);
    }
    stackedTimer->stop("PyAlbany: getResponseMap");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<const PyTrilinosMap> PyProblem::getStateMap()
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
        return getPyTrilinosMap(Albany::getTpetraMap(s_space), false);
    }
    stackedTimer->stop("PyAlbany: getStateMap");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<const PyTrilinosMap> PyProblem::getParameterMap(const int p_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getParameterMap");
    auto p_space = solver->get_p_space(p_index);
    auto outputMap = getPyTrilinosMap(Albany::getTpetraMap(p_space), true);
    stackedTimer->stop("PyAlbany: getParameterMap");
    stackedTimer->stopBaseTimer();
    return outputMap;
}

void PyProblem::setDirections(const int p_index, Teuchos::RCP<PyTrilinosMultiVector> direction)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: setDirections");
    forwardHasBeenSolved = false;

    const unsigned int n_directions = direction->getNumVectors();

    for (size_t l = 0; l < solver->Np(); l++)
    {
        if (p_index == l)
        {
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
            thyraDirections[l] = Albany::createThyraMultiVector(direction);
            continue;
#endif
        }

        bool is_null = Teuchos::is_null(thyraDirections[l]);
        if (is_null)
        {
            auto p_space = solver->getNominalValues().get_p(l)->space();
            thyraDirections[l] = Thyra::createMembers(p_space, n_directions);
        }
        if (p_index == l)
        {
            Teuchos::RCP<Tpetra_MultiVector> thyraDirectionsT = Albany::getTpetraMultiVector(thyraDirections[l]);

            auto directions_in_view = direction->getLocalView<PyTrilinosMultiVector::node_type::device_type>();
            auto directions_out_view = thyraDirectionsT->getLocalView<Tpetra_MultiVector::node_type::device_type>();
            Kokkos::deep_copy(directions_out_view, directions_in_view);
        }
        else if (is_null)
            for (size_t i_direction = 0; i_direction < n_directions; i_direction++)
                thyraDirections[l]->col(i_direction)->assign(0.0);
    }

    stackedTimer->stop("PyAlbany: setDirections");
    stackedTimer->stopBaseTimer();
}

void PyProblem::setParameter(const int p_index, Teuchos::RCP<PyTrilinosVector> p)
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

    for (size_t l = 0; l < solver->Np(); l++)
    {
        if (p_index == l)
        {
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
            thyraParameter[l] = Albany::createThyraVector(p);
            albanyModel->setNominalValue(l, thyraParameter[l]);
            continue;
#endif
        }

        bool is_null = Teuchos::is_null(thyraParameter[l]);
        if (is_null)
        {
            auto p_space = solver->getNominalValues().get_p(l)->space();
            thyraParameter[l] = Thyra::createMember(p_space);
        }
        if (p_index == l)
        {
            Teuchos::RCP<Tpetra_MultiVector> thyraParameterT = Albany::getTpetraVector(thyraParameter[l]);

            auto p_in_view = p->getLocalView<PyTrilinosVector::node_type::device_type>();
            auto p_out_view = thyraParameterT->getLocalView<Tpetra_Vector::node_type::device_type>();
            Kokkos::deep_copy(p_out_view, p_in_view);
        }
        else if (is_null)
            thyraParameter[l]->assign(0.0);

        if (p_index == l)
            albanyModel->setNominalValue(l, thyraParameter[l]);
    }

    stackedTimer->stop("PyAlbany: setParameter");
    stackedTimer->stopBaseTimer();
}

Teuchos::RCP<PyTrilinosVector> PyProblem::getParameter(const int p_index)
{
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
    stackedTimer->startBaseTimer();
    stackedTimer->start("PyAlbany: getParameter");
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
    Teuchos::RCP<Thyra_Vector> p = thyraParameter[p_index];
    Teuchos::RCP<PyTrilinosVector> p_out = Albany::getTpetraVector(p);
    stackedTimer->stop("PyAlbany: getParameter");
    stackedTimer->stopBaseTimer();
    return p_out;
#else
    Teuchos::RCP<const Thyra_Vector> p = thyraParameter[p_index];
    Teuchos::RCP<PyTrilinosVector> p_out = rcp(new PyTrilinosVector(this->getParameterMap(p_index)));
    if (Teuchos::nonnull(p))
    {
        Teuchos::RCP<const Tpetra_Vector> pT = Albany::getConstTpetraVector(p);

        auto p_out_view = p_out->getLocalView<PyTrilinosMultiVector::node_type::device_type>();
        auto p_in_view = pT->getLocalView<Tpetra_MultiVector::node_type::device_type>();
        Kokkos::deep_copy(p_out_view, p_in_view);
        stackedTimer->stop("PyAlbany: getParameter");
        stackedTimer->stopBaseTimer();
        return p_out;
    }
    stackedTimer->stop("PyAlbany: getParameter");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
#endif
}

Teuchos::RCP<PyTrilinosVector> PyProblem::getResponse(const int g_index)
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
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
        Teuchos::RCP<Thyra_Vector> g = thyraResponses[g_index];
        Teuchos::RCP<PyTrilinosVector> g_out = Albany::getTpetraVector(g);
        stackedTimer->stop("PyAlbany: getResponse");
        stackedTimer->stopBaseTimer();
        return g_out;
#else
        Teuchos::RCP<const Thyra_Vector> g = thyraResponses[g_index];
        Teuchos::RCP<PyTrilinosVector> g_out = rcp(new PyTrilinosVector(this->getResponseMap(g_index)));
        if (Teuchos::nonnull(g))
        {
            Teuchos::RCP<const Tpetra_Vector> gT = Albany::getConstTpetraVector(g);

            auto g_out_view = g_out->getLocalView<PyTrilinosMultiVector::node_type::device_type>();
            auto g_in_view = gT->getLocalView<Tpetra_MultiVector::node_type::device_type>();
            Kokkos::deep_copy(g_out_view, g_in_view);
            stackedTimer->stop("PyAlbany: getResponse");
            stackedTimer->stopBaseTimer();
            return g_out;
        }
#endif
    }
    stackedTimer->stop("PyAlbany: getResponse");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<PyTrilinosVector> PyProblem::getState()
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
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
        Teuchos::RCP<Thyra_Vector> s = thyraResponses.back();
        Teuchos::RCP<PyTrilinosVector> s_out = Albany::getTpetraVector(s);
        stackedTimer->stop("PyAlbany: getState");
        stackedTimer->stopBaseTimer();
        return s_out;
#else
        Teuchos::RCP<const Thyra_Vector> s = thyraResponses.back();
        Teuchos::RCP<PyTrilinosVector> s_out = rcp(new PyTrilinosVector(this->getStateMap()));
        if (Teuchos::nonnull(s))
        {
            Teuchos::RCP<const Tpetra_Vector> sT = Albany::getConstTpetraVector(s);
            auto s_out_view = s_out->getLocalView<PyTrilinosMultiVector::node_type::device_type>();
            auto s_in_view = sT->getLocalView<Tpetra_MultiVector::node_type::device_type>();
            Kokkos::deep_copy(s_out_view, s_in_view);
            stackedTimer->stop("PyAlbany: getState");
            stackedTimer->stopBaseTimer();
            return s_out;
        }
#endif
    }
    stackedTimer->stop("PyAlbany: getState");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<PyTrilinosMultiVector> PyProblem::getSensitivity(const int g_index, const int p_index)
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
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
        Teuchos::RCP<PyTrilinosMultiVector> dg_out = Albany::getTpetraMultiVector(thyraSensitivities[g_index][p_index]);
        stackedTimer->stop("PyAlbany: getSensitivity");
        stackedTimer->stopBaseTimer();
        return dg_out;
#else
        Teuchos::RCP<const Thyra_MultiVector> dg = thyraSensitivities[g_index][p_index];
        if (Teuchos::nonnull(dg))
        {
            Teuchos::RCP<const Tpetra_MultiVector> dgT = Albany::getConstTpetraMultiVector(dg);
            Teuchos::RCP<PyTrilinosMultiVector> dg_out = rcp(new PyTrilinosMultiVector(this->getParameterMap(p_index),
                                                                                       dgT->getNumVectors()));
            auto dg_out_view = dg_out->getLocalView<PyTrilinosMultiVector::node_type::device_type>();
            auto dg_in_view = dgT->getLocalView<Tpetra_MultiVector::node_type::device_type>();
            Kokkos::deep_copy(dg_out_view, dg_in_view);
            stackedTimer->stop("PyAlbany: getSensitivity");
            stackedTimer->stopBaseTimer();
            return dg_out;
        }
#endif
    }
    stackedTimer->stop("PyAlbany: getSensitivity");
    stackedTimer->stopBaseTimer();
    return Teuchos::null;
}

Teuchos::RCP<PyTrilinosMultiVector> PyProblem::getReducedHessian(const int g_index, const int p_index)
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
#ifdef PYALBANY_DOES_NOT_USE_DEEP_COPY
        Teuchos::RCP<PyTrilinosMultiVector> hv_out = Albany::getTpetraMultiVector(thyraReducedHessian[g_index][p_index]);
        stackedTimer->stop("PyAlbany: getReducedHessian");
        stackedTimer->stopBaseTimer();
        return hv_out;
#else
        Teuchos::RCP<const Thyra_MultiVector> hv = thyraReducedHessian[g_index][p_index];
        if (Teuchos::nonnull(hv))
        {
            Teuchos::RCP<const Tpetra_MultiVector> hvT = Albany::getConstTpetraMultiVector(hv);
            Teuchos::RCP<PyTrilinosMultiVector> hv_out = rcp(new PyTrilinosMultiVector(this->getParameterMap(p_index),
                                                                                       hvT->getNumVectors()));
            auto hv_out_view = hv_out->getLocalView<PyTrilinosMultiVector::node_type::device_type>();
            auto hv_in_view = hvT->getLocalView<Tpetra_MultiVector::node_type::device_type>();
            Kokkos::deep_copy(hv_out_view, hv_in_view);
            stackedTimer->stop("PyAlbany: getReducedHessian");
            stackedTimer->stopBaseTimer();
            return hv_out;
        }
#endif
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

    for (size_t l = 0; l < solver->Np(); l++)
    {
        thyraParameter[l] = p_dpv->getNonconstVectorBlock(l);
        albanyModel->setNominalValue(l, thyraParameter[l]);
    }

    inverseHasBeenSolved = true;

    stackedTimer->stop("PyAlbany: performAnalysis");
    stackedTimer->stopBaseTimer();
    bool error = (status != 0);
    return error;
}

void PyProblem::reportTimers()
{
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = true;
    options.output_minmax = true;
    stackedTimer->report(std::cout, Teuchos::DefaultComm<int>::getComm(), options);
}
