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

        // get initial conditions
        Teuchos::ArrayRCP<
                Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > > wsElNodeEqID =
                disc->getWsElNodeEqID();
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords =
                disc->getCoords();
        Teuchos::ArrayRCP<std::string> wsEBNames = disc->getWsEBNames();
        const int numDim = disc->getNumDim();
        const int neq = disc->getNumEq();

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
        // get mesh
	Teuchos::RCP<Albany::SimDiscretization> sim_disc =
	  Teuchos::rcp_dynamic_cast<Albany::SimDiscretization>(disc);
        Teuchos::RCP<Albany::APFMeshStruct> apf_ms = sim_disc->getAPFMeshStruct();
        
        apf::Mesh* apf_m = apf_ms->getMesh();
        
        apf::writeVtkFiles("out",apf_m);

    }

    Application::
    ~Application() {
        //
    }
}
