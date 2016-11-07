// CTM
#include "CTM_Application.hpp"
#include "CTM_SolutionInfo.hpp"
// Albany
#include "AAdapt_InitialCondition.cpp"
#if (defined(ALBANY_SCOREC) || defined(ALBANY_AMP))
#include "Albany_APFDiscretization.hpp"
#include "Albany_SimDiscretization.hpp"
#endif

namespace CTM {

    int countJac; //counter which counts instances of Jacobian (for debug output)
    int countRes; //counter which counts instances of residual (for debug output)
    int countScale;

    Application::Application(Teuchos::RCP<Teuchos::ParameterList> p,
            Teuchos::RCP<SolutionInfo> sinfo,
            Teuchos::RCP<Albany::AbstractProblem> prob,
            Teuchos::RCP<Albany::AbstractDiscretization> d,
            const Albany::StateManager& state_mgr,
            bool isThermal) :
    params(p),
    solution_info(sinfo),
    problem(prob),
    disc(d),
    stateMgr(state_mgr),
    phxGraphVisDetail(0),
    stateGraphVisDetail(0),
    out(Teuchos::VerboseObjectBase::getDefaultOStream()) {

        // Set up memory for workset
        fm = problem->getFieldManager();
        TEUCHOS_TEST_FOR_EXCEPTION(fm == Teuchos::null, std::logic_error,
                "getFieldManager not implemented!!!");
        dfm = problem->getDirichletFieldManager();

        nfm = problem->getNeumannFieldManager();

        meshSpecs = disc->getMeshStruct()->getMeshSpecs();

        // get initial conditions
        Teuchos::ArrayRCP<
                Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > > wsElNodeEqID =
                disc->getWsElNodeEqID();
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords =
                disc->getCoords();
        Teuchos::ArrayRCP<std::string> wsEBNames = disc->getWsEBNames();
        numDim = disc->getNumDim();
        neq = disc->getNumEq();

        // Create problem PL
        Teuchos::RCP<Teuchos::ParameterList> problemParams;
        if (isThermal) {
            problemParams = Teuchos::sublist(params, "Temperature Problem", true);
        } else {
            // Create problem PL
            problemParams =
                    Teuchos::sublist(params, "Mechanics Problem", true);
        }
        // get solution ghost vector
        Teuchos::RCP<Tpetra_MultiVector> ghost_soln = solution_info->getGhostMV();
        AAdapt::InitialConditionsT(
                ghost_soln->getVectorNonConst(0), wsElNodeEqID, wsEBNames, coords, neq, numDim,
                problemParams->sublist("Initial Condition"),
                disc->hasRestartSolution());
        //
        Teuchos::RCP<Tpetra_MultiVector> owned_soln = solution_info->getOwnedMV();
        // get exporter
        Teuchos::RCP<Tpetra_Export> exporter = solution_info->getExporter();

        owned_soln->getVectorNonConst(0)->doExport(*ghost_soln->getVector(0), *exporter, Tpetra::INSERT);

        
#if (defined(ALBANY_SCOREC) || defined(ALBANY_AMP))
        {
            const Teuchos::RCP< Albany::APFDiscretization > apf_disc =
                    Teuchos::rcp_dynamic_cast< Albany::APFDiscretization >(disc);
            if (!apf_disc.is_null()) {
                apf_disc->writeSolutionMVToMeshDatabase(*ghost_soln, 0, true);
            }
//            apf_disc->initTemperatureHack();
        }
#endif

        // Create debug output object
        RCP<Teuchos::ParameterList> debugParams =
                Teuchos::sublist(params, "Debug Output", false);
        writeToMatrixMarketJac = debugParams->get("Write Jacobian to MatrixMarket", 0);
        writeToMatrixMarketRes = debugParams->get("Write Residual to MatrixMarket", 0);
        writeToCoutJac = debugParams->get("Write Jacobian to Standard Output", 0);
        writeToCoutRes = debugParams->get("Write Residual to Standard Output", 0);
        //the above 4 parameters cannot have values < -1
        if (writeToMatrixMarketJac < -1) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "Error in Albany::Application constructor:  " <<
                    "Invalid Parameter Write Jacobian to MatrixMarket.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);
        }
        if (writeToMatrixMarketRes < -1) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "Error in Albany::Application constructor:  " <<
                    "Invalid Parameter Write Residual to MatrixMarket.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);
        }
        if (writeToCoutJac < -1) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "Error in Albany::Application constructor:  " <<
                    "Invalid Parameter Write Jacobian to Standard Output.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);
        }
        if (writeToCoutRes < -1) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "Error in Albany::Application constructor:  " <<
                    "Invalid Parameter Write Residual to Standard Output.  Acceptable values are -1, 0, 1, 2, ... " << std::endl);
        }
        if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0)
            countJac = 0; //initiate counter that counts instances of Jacobian matrix to 0
        if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0)
            countRes = 0; //initiate counter that counts instances of Jacobian matrix to 0

    }

    Application::
    ~Application() {
        //
    }

    Teuchos::RCP<Albany::AbstractDiscretization>
    Application::getDiscretization() const {
        return disc;
    }

    Teuchos::RCP<Albany::AbstractProblem>
    Application::getProblem() const {
        return problem;
    }

    Teuchos::RCP<SolutionInfo> Application::getSolutionInfo() const {
        return solution_info;
    }

    namespace {
        //amb-nfm I think right now there is some confusion about nfm. Long ago, nfm was
        // like dfm, just a single field manager. Then it became an array like fm. At
        // that time, it may have been true that nfm was indexed just like fm, using
        // wsPhysIndex. However, it is clear at present (7 Nov 2014) that nfm is
        // definitely not indexed like fm. As an example, compare nfm in
        // Albany::MechanicsProblem::constructNeumannEvaluators and fm in
        // Albany::MechanicsProblem::buildProblem. For now, I'm going to keep nfm as an
        // array, but this this new function is a wrapper around the unclear intended
        // behavior.

        inline Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >&
        deref_nfm(
                Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > >& nfm,
                const Albany::WorksetArray<int>::type& wsPhysIndex, int ws) {
            return
            nfm.size() == 1 ? // Currently, all problems seem to have one nfm ...
                    nfm[0] : // ... hence this is the intended behavior ...
                    nfm[wsPhysIndex[ws]]; // ... and this is not, but may one day be again.
        }

        // Convenience routine for setting dfm workset data. Cut down on redundant code.

        void dfm_set(
                PHAL::Workset& workset,
                const Teuchos::RCP<const Tpetra_Vector>& x,
                const Teuchos::RCP<const Tpetra_Vector>& xd,
                const Teuchos::RCP<const Tpetra_Vector>& xdd) {
            workset.xT = x;
            workset.transientTerms = Teuchos::nonnull(xd);
            workset.accelerationTerms = Teuchos::nonnull(xdd);
        }

    }

    void Application::computeGlobalResidualT(const double current_time,
            const Tpetra_Vector* xdotT,
            const Tpetra_Vector* xdotdotT,
            const Tpetra_Vector& xT,
            Tpetra_Vector& fT) {
        // Create non-owning RCPs to Tpetra objects
        // to be passed to the implementation
        this->computeGlobalResidualImplT(
                current_time,
                Teuchos::rcp(xdotT, false),
                Teuchos::rcp(xdotdotT, false),
                Teuchos::rcpFromRef(xT),
                Teuchos::rcpFromRef(fT));

        //Debut output
        if (writeToMatrixMarketRes != 0) { //If requesting writing to MatrixMarket of residual...
            char name[100]; //create string for file name
            if (writeToMatrixMarketRes == -1) { //write residual to MatrixMarket every time it arises
                sprintf(name, "rhs%i.mm", countRes);
                Tpetra_MatrixMarket_Writer::writeDenseFile(name, Teuchos::rcpFromRef(fT));
            } else {
                if (countRes == writeToMatrixMarketRes) { //write residual only at requested count#
                    sprintf(name, "rhs%i.mm", countRes);
                    Tpetra_MatrixMarket_Writer::writeDenseFile(
                            name,
                            Teuchos::rcpFromRef(fT));
                }
            }
        }
        if (writeToCoutRes != 0) { //If requesting writing of residual to cout...
            if (writeToCoutRes == -1) { //cout residual time it arises
                std::cout << "Global Residual #" << countRes << ": " << std::endl;
                fT.describe(*out, Teuchos::VERB_EXTREME);
            } else {
                if (countRes == writeToCoutRes) { //cout residual only at requested count#
                    std::cout << "Global Residual #" << countRes << ": " << std::endl;
                    fT.describe(*out, Teuchos::VERB_EXTREME);
                }
            }
        }
        if (writeToMatrixMarketRes != 0 || writeToCoutRes != 0) {
            countRes++; //increment residual counter
        }
    }

    void
    Application::computeGlobalResidualImplT(
            const double current_time,
            const Teuchos::RCP<const Tpetra_Vector>& xdotT,
            const Teuchos::RCP<const Tpetra_Vector>& xdotdotT,
            const Teuchos::RCP<const Tpetra_Vector>& xT,
            const Teuchos::RCP<Tpetra_Vector>& fT) {

        TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Residual");
        postRegSetup("Residual");

        // Load connectivity map and coordinates
        const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
                wsElNodeEqID = disc->getWsElNodeEqID();
        const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
                coords = disc->getCoords();
        const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
        const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

        int numWorksets = wsElNodeEqID.size();

        const Teuchos::RCP<Tpetra_Vector> overlapped_fT = solution_info->getGhostResidual();
        const Teuchos::RCP<Tpetra_Export> exporterT = solution_info->getExporter();
        const Teuchos::RCP<Tpetra_Import> importerT = solution_info->getImporter();

        // Scatter x and xdot to the overlapped distribution
        solution_info->scatter_x(*xT, xdotT.get(), xdotdotT.get());

        // Zero out overlapped residual - Tpetra
        overlapped_fT->putScalar(0.0);
        fT->putScalar(0.0);

        // Set data in Workset struct, and perform fill via field manager
        {
            PHAL::Workset workset;

            loadBasicWorksetInfoT(workset, current_time);
            workset.fT = overlapped_fT;

            for (int ws = 0; ws < numWorksets; ws++) {
                loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(workset, ws);

                // FillType template argument used to specialize Sacado
                fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
                if (nfm != Teuchos::null)
                    deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
            }
        }

        // Assemble the residual into a non-overlapping vector
        fT->doExport(*overlapped_fT, *exporterT, Tpetra::ADD);

#ifdef WRITE_TO_MATRIX_MARKET
        char nameResUnscaled[100]; //create string for file name
        sprintf(nameResUnscaled, "resUnscaled%i_residual.mm", countScale);
        Tpetra_MatrixMarket_Writer::writeDenseFile(nameResUnscaled, fT);
#endif

#ifdef ALBANY_LCM
        // Write the residual to the discretization, which will later (optionally) be written to the output file
        disc->setResidualFieldT(*overlapped_fT);
#endif

        // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)

        if (dfm != Teuchos::null) {
            PHAL::Workset workset;

            workset.fT = fT;
            loadWorksetNodesetInfo(workset);
            dfm_set(workset, xT, xdotT, xdotdotT);
            workset.current_time = current_time;
            workset.distParamLib = Teuchos::null;
            workset.disc = disc;

            // FillType template argument used to specialize Sacado
            dfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
        }
    }

    void
    Application::computeGlobalJacobianT(
            const double alpha,
            const double beta,
            const double omega,
            const double current_time,
            const Tpetra_Vector* xdotT,
            const Tpetra_Vector* xdotdotT,
            const Tpetra_Vector& xT,
            Tpetra_Vector* fT,
            Tpetra_CrsMatrix& jacT) {
        // Create non-owning RCPs to Tpetra objects
        // to be passed to the implementation
        this->computeGlobalJacobianImplT(
                alpha,
                beta,
                omega,
                current_time,
                Teuchos::rcp(xdotT, false),
                Teuchos::rcp(xdotdotT, false),
                Teuchos::rcpFromRef(xT),
                Teuchos::rcp(fT, false),
                Teuchos::rcpFromRef(jacT));
        //Debut output
        if (writeToMatrixMarketJac != 0) { //If requesting writing to MatrixMarket of Jacobian...
            char name[100]; //create string for file name
            if (writeToMatrixMarketJac == -1) { //write jacobian to MatrixMarket every time it arises
                sprintf(name, "jac%i.mm", countJac);
                Tpetra_MatrixMarket_Writer::writeSparseFile(
                        name,
                        Teuchos::rcpFromRef(jacT));
            } else {
                if (countJac == writeToMatrixMarketJac) { //write jacobian only at requested count#
                    sprintf(name, "jac%i.mm", countJac);
                    Tpetra_MatrixMarket_Writer::writeSparseFile(
                            name,
                            Teuchos::rcpFromRef(jacT));
                }
            }
        }
        if (writeToCoutJac != 0) { //If requesting writing Jacobian to standard output (cout)...
            if (writeToCoutJac == -1) { //cout jacobian every time it arises
                *out << "Global Jacobian #" << countJac << ": " << std::endl;
                jacT.describe(*out, Teuchos::VERB_EXTREME);
            } else {
                if (countJac == writeToCoutJac) { //cout jacobian only at requested count#
                    *out << "Global Jacobian #" << countJac << ": " << std::endl;
                    jacT.describe(*out, Teuchos::VERB_EXTREME);
                }
            }
        }
        if (writeToMatrixMarketJac != 0 || writeToCoutJac != 0) {
            countJac++; //increment Jacobian counter
        }
    }

    void
    Application::computeGlobalJacobianImplT(
            const double alpha,
            const double beta,
            const double omega,
            const double current_time,
            const Teuchos::RCP<const Tpetra_Vector>& xdotT,
            const Teuchos::RCP<const Tpetra_Vector>& xdotdotT,
            const Teuchos::RCP<const Tpetra_Vector>& xT,
            const Teuchos::RCP<Tpetra_Vector>& fT,
            const Teuchos::RCP<Tpetra_CrsMatrix>& jacT) {
        TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian");

        postRegSetup("Jacobian");

        // Load connectivity map and coordinates
        const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
                wsElNodeEqID = disc->getWsElNodeEqID();
        const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
                coords = disc->getCoords();
        const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
        const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

        int numWorksets = wsElNodeEqID.size();

        //Teuchos::RCP<Tpetra_Vector> overlapped_fT = solMgrT->get_overlapped_fT();
        Teuchos::RCP<Tpetra_Vector> overlapped_fT;
        if (Teuchos::nonnull(fT)) {
            overlapped_fT = solution_info->getGhostResidual();
        } else {
            overlapped_fT = Teuchos::null;
        }
        Teuchos::RCP<Tpetra_CrsMatrix> overlapped_jacT = solution_info->getGhostJacobian();
        Teuchos::RCP<Tpetra_Export> exporterT = solution_info->getExporter();

        // Scatter x and xdot to the overlapped distribution
        solution_info->scatter_x(*xT, xdotT.get(), xdotdotT.get());

        // Zero out overlapped residual
        if (Teuchos::nonnull(fT)) {
            overlapped_fT->putScalar(0.0);
            fT->putScalar(0.0);
        }

        // Zero out Jacobian
        jacT->resumeFill();
        jacT->setAllToScalar(0.0);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
        if (!overlapped_jacT->isFillActive())
            overlapped_jacT->resumeFill();
#endif
        overlapped_jacT->setAllToScalar(0.0);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
        if (overlapped_jacT->isFillActive()) {
            // Makes getLocalMatrix() valid.
            overlapped_jacT->fillComplete();
        }
        if (!overlapped_jacT->isFillActive())
            overlapped_jacT->resumeFill();

#endif

        // Set data in Workset struct, and perform fill via field manager
        {
            PHAL::Workset workset;
            loadBasicWorksetInfoT(workset, current_time);

            workset.fT = overlapped_fT;
            workset.JacT = overlapped_jacT;
            loadWorksetJacobianInfo(workset, alpha, beta, omega);

            //fill Jacobian derivative dimensions:
            for (int ps = 0; ps < fm.size(); ps++) {
                // get derivative dimension
                int node_count = this->getEnrichedMeshSpecs()[ps].get()->ctd.node_count;
                int derivDimenensions = neq * node_count;
                (workset.Jacobian_deriv_dims).push_back(derivDimenensions);
            }


            for (int ws = 0; ws < numWorksets; ws++) {
                loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(workset, ws);
                // FillType template argument used to specialize Sacado
                fm[wsPhysIndex[ws]]->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
                if (Teuchos::nonnull(nfm))
                    deref_nfm(nfm, wsPhysIndex, ws)->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
            }
        }

        {
            TEUCHOS_FUNC_TIME_MONITOR("> Albany Fill: Jacobian Export");

            // Assemble global residual
            if (Teuchos::nonnull(fT))
                fT->doExport(*overlapped_fT, *exporterT, Tpetra::ADD);

            // Assemble global Jacobian
            jacT->doExport(*overlapped_jacT, *exporterT, Tpetra::ADD);

        } // End timer
        // Apply Dirichlet conditions using dfm (Dirchelt Field Manager)
        if (Teuchos::nonnull(dfm)) {
            PHAL::Workset workset;

            workset.fT = fT;
            workset.JacT = jacT;
            workset.m_coeff = alpha;
            workset.n_coeff = omega;
            workset.j_coeff = beta;
            workset.current_time = current_time;

            dfm_set(workset, xT, xdotT, xdotdotT);

            loadWorksetNodesetInfo(workset);
            workset.distParamLib = Teuchos::null;
            workset.disc = disc;

            // FillType template argument used to specialize Sacado
            dfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
        }
        jacT->fillComplete();


#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
        if (overlapped_jacT->isFillActive()) {
            // Makes getLocalMatrix() valid.
            overlapped_jacT->fillComplete();
        }
#endif

    }

    void Application::loadWorksetSidesetInfo(PHAL::Workset& workset, const int ws) {

        workset.sideSets = Teuchos::rcpFromRef(disc->getSideSets(ws));

    }

    void Application::loadBasicWorksetInfoT(
            PHAL::Workset& workset,
            double current_time) {

        workset.numEqs = neq;

        Teuchos::RCP<Tpetra_MultiVector> overlapped_MV = solution_info->getGhostMV();
        workset.xT = overlapped_MV->getVectorNonConst(0);
        workset.xdotT =
                (overlapped_MV->getNumVectors() > 1) ? overlapped_MV->getVectorNonConst(1) : Teuchos::null;
        workset.xdotdotT =
                (overlapped_MV->getNumVectors() > 2) ? overlapped_MV->getVectorNonConst(2) : Teuchos::null;
        workset.current_time = current_time;
        workset.distParamLib = Teuchos::null;
        workset.disc = disc;
        workset.transientTerms = Teuchos::nonnull(workset.xdotT);
        workset.accelerationTerms = Teuchos::nonnull(workset.xdotdotT);
    }

    void Application::loadWorksetJacobianInfo(PHAL::Workset& workset,
            const double& alpha, const double& beta, const double& omega) {
        workset.m_coeff = alpha;
        workset.n_coeff = omega;
        workset.j_coeff = beta;
        workset.ignore_residual = false;
        workset.is_adjoint = false;
    }

    void Application::loadWorksetNodesetInfo(PHAL::Workset& workset) {
        workset.nodeSets = Teuchos::rcpFromRef(disc->getNodeSets());
        workset.nodeSetCoords = Teuchos::rcpFromRef(disc->getNodeSetCoords());
    }

    void Application::postRegSetup(std::string eval) {
        if (setupSet.find(eval) != setupSet.end()) return;

        setupSet.insert(eval);

        if (eval == "Residual") {
            for (int ps = 0; ps < fm.size(); ps++)
                fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
            if (dfm != Teuchos::null)
                dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
            if (nfm != Teuchos::null)
                for (int ps = 0; ps < nfm.size(); ps++)
                    nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Residual>(eval);
        } else if (eval == "Jacobian") {
            for (int ps = 0; ps < fm.size(); ps++) {
                std::vector<PHX::index_size_type> derivative_dimensions;
                // get derivative dimension
                int node_count = this->getEnrichedMeshSpecs()[ps].get()->ctd.node_count;
                int derivDimenensions = neq * node_count;
                derivative_dimensions.push_back(derivDimenensions);
                fm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
                fm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
                if (nfm != Teuchos::null && ps < nfm.size()) {
                    nfm[ps]->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
                    nfm[ps]->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
                }
            }
            if (dfm != Teuchos::null) {
                //amb Need to look into this. What happens with DBCs in meshes having
                // different element types?
                std::vector<PHX::index_size_type> derivative_dimensions;
                // get derivative dimension
                int node_count = this->getEnrichedMeshSpecs()[0].get()->ctd.node_count;
                int derivDimenensions = neq * node_count;
                derivative_dimensions.push_back(derivDimenensions);
                dfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(derivative_dimensions);
                dfm->postRegistrationSetupForType<PHAL::AlbanyTraits::Jacobian>(eval);
            }
        } else
            TEUCHOS_TEST_FOR_EXCEPTION(eval != "Known Evaluation Name", std::logic_error,
                "Error in setup call \n" << " Unrecognized name: " << eval << std::endl);


        // Write out Phalanx Graph if requested, on Proc 0, for Resid and Jacobian
        bool alreadyWroteResidPhxGraph = false;
        bool alreadyWroteJacPhxGraph = false;

        if (phxGraphVisDetail > 0) {
            bool detail = false;
            if (phxGraphVisDetail > 1) detail = true;

            if ((eval == "Residual") && (alreadyWroteResidPhxGraph == false)) {
                *out << "Phalanx writing graphviz file for graph of Residual fill (detail ="
                        << phxGraphVisDetail << ")" << std::endl;
                *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << std::endl;
                for (int ps = 0; ps < fm.size(); ps++) {
                    std::stringstream pg;
                    pg << "phalanx_graph_" << ps;
                    fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Residual>(pg.str(), detail, detail);
                }
                alreadyWroteResidPhxGraph = true;
                //      phxGraphVisDetail = -1;
            } else if ((eval == "Jacobian") && (alreadyWroteJacPhxGraph == false)) {
                *out << "Phalanx writing graphviz file for graph of Jacobian fill (detail ="
                        << phxGraphVisDetail << ")" << std::endl;
                *out << "Process using 'dot -Tpng -O phalanx_graph' \n" << std::endl;
                for (int ps = 0; ps < fm.size(); ps++) {
                    std::stringstream pg;
                    pg << "phalanx_graph_jac_" << ps;
                    fm[ps]->writeGraphvizFile<PHAL::AlbanyTraits::Jacobian>(pg.str(), detail, detail);
                }
                alreadyWroteJacPhxGraph = true;
            }
            //Stop writing out phalanx graphs only when a Jacobian and a Residual graph has been written out
            if ((alreadyWroteResidPhxGraph == true) && (alreadyWroteJacPhxGraph == true))
                phxGraphVisDetail = -2;
        }
    }

}
