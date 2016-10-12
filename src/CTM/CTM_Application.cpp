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

    Application::Application(Teuchos::RCP<Teuchos::ParameterList> p,
            Teuchos::RCP<SolutionInfo> sinfo,
            Teuchos::RCP<Albany::AbstractProblem> prob,
            Teuchos::RCP<Albany::AbstractDiscretization> d) :
    params(p),
    solution_info(sinfo),
    problem(prob),
    disc(d),
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
        Teuchos::RCP<Teuchos::ParameterList> problemParams =
                Teuchos::sublist(params, "Temperature Problem", true);

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
        }
#endif

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
    
    void Application::loadWorksetSidesetInfo(PHAL::Workset& workset, const int ws) {

        workset.sideSets = Teuchos::rcpFromRef(disc->getSideSets(ws));

    }

    void Application::loadBasicWorksetInfoT(
            PHAL::Workset& workset,
            double current_time) {
        
        workset.numEqs = neq;
        
        Teuchos::RCP<Tpetra_MultiVector> overlapped_MV = solution_info->getOwnedMV();
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
