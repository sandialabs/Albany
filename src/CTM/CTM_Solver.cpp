#include "CTM_Solver.hpp"
#include "CTM_Application.hpp"
#include "CTM_ThermalProblem.hpp"
#include "CTM_MechanicsProblem.hpp"
#include <Albany_DiscretizationFactory.hpp>
#include "Albany_APFDiscretization.hpp"
#include <Albany_AbstractDiscretization.hpp>
#include "CTM_SolutionInfo.hpp"
#include "linear_solver.hpp"

#ifdef ALBANY_AMP
#include "AMP/problems/PhaseProblem.hpp"
#endif

namespace CTM {

    static RCP<ParameterList> get_valid_params() {
        auto p = rcp(new ParameterList);
        p->sublist("Temperature Problem");
        p->sublist("Temperature Problem").sublist("Thermal Discretization");
        p->sublist("Mechanics Problem");
        p->sublist("Mechanics Problem").sublist("Mechanics Discretization");
        p->sublist("Linear Algebra");
        p->sublist("Time");
    }

    static void validate_params(RCP<const ParameterList> p) {
        assert(p->isSublist("Temperature Problem"));
        assert(p->isSublist("Mechanics Problem"));
        assert(p->isSublist("Thermal Discretization"));
        assert(p->isSublist("Mechanics Discretization"));
        assert(p->isSublist("Linear Algebra"));
        assert(p->isSublist("Time"));
        //
        const Teuchos::ParameterList &time_list = p->sublist("Time");
        assert(time_list.isType<double>("Initial Time"));
        assert(time_list.isType<double>("Step Size"));
        assert(time_list.isType<int>("Number of Steps"));
        //
        const Teuchos::ParameterList &la_list = p->sublist("Linear Algebra");
        // check if solver was specified
        assert((la_list.isSublist("GMRES Solver")) ||
                (la_list.isType<std::string>("Solver")));
        // Get GMRES solver
        if (la_list.isSublist("GMRES Solver")) {
            const Teuchos::ParameterList &params = la_list.sublist("GMRES Solver");
            assert(params.isType<double>("Linear Tolerance"));
            assert(params.isType<int>("Linear Max. Iterations"));
            assert(params.isType<int>("Linear Krylov Size"));
        } else {
            assert(la_list.get<std::string>("Solver") == "SuperLU_DIST");
        }
        assert(la_list.isType<double>("Nonlinear Tolerance"));
        assert(la_list.isType<int>("Nonlinear Max. Iterations"));
    }

    Solver::Solver(
            RCP<const Teuchos_Comm> c,
            RCP<ParameterList> p) :
    comm(c),
    out(Teuchos::VerboseObjectBase::getDefaultOStream()),
    params(p) {

        validate_params(params);
        temp_params = rcpFromRef(params->sublist("Temperature Problem", true));
        //
        mech_params = rcpFromRef(params->sublist("Mechanics Problem", true));
        //
        Teuchos::ParameterList &time_list = params->sublist("Time");
        t_old = time_list.get<double>("Initial Time");
        dt = time_list.get<double>("Step Size");
        num_steps = time_list.get<int>("Number of Steps");
        //
        t_current = t_old + dt;

        initial_setup();
    }

    void Solver::initial_setup() {

        // create parameter libraries
        // note: we never intend to use these objects, we create them because they
        // are inputs to constructors for various other objects.
        param_lib = rcp(new ParamLib);
        dist_param_lib = rcp(new DistParamLib);

        // create the mesh specs struct
        // We use the thermal discretization parameter list to load the model
        // and mesh. Later we will build a mechanics discretization using same
        // discretization factory. The purpose is to reuse mesh structure.
        Teuchos::RCP<Teuchos::ParameterList> disc_t_params = Teuchos::rcp(new Teuchos::ParameterList);
        disc_t_params = rcpFromRef(params->sublist("Thermal Discretization"));
        disc_factory = rcp(new Albany::DiscretizationFactory(disc_t_params, comm, false));
        mesh_specs = disc_factory->createMeshSpecs();

        // create the problem objects
        auto dim = mesh_specs[0]->numDim;

        std::string& method = temp_params->get("Name", "Thermal");
        if (method == "Thermal") {
            t_problem = rcp(new ThermalProblem(temp_params, param_lib, dim, comm));
        }
#ifdef ALBANY_AMP
        else if (method == "Phase3D") {
            t_problem = rcp(new Albany::PhaseProblem(temp_params, param_lib, 3, comm));
        }
#endif
        else {
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                    std::endl <<
                    "Error!  Unknown problem " << method <<
                    "!" << std::endl << "Supplied parameter list is " <<
                    std::endl << *temp_params);
        }
        
        temp_params->validateParameters(*(t_problem->getValidProblemParameters()), 0);
        t_problem->buildProblem(mesh_specs, t_state_mgr);


        // Mechanics problem
        Teuchos::RCP<Teuchos::ParameterList> disc_m_params = Teuchos::rcp(new Teuchos::ParameterList);
        disc_m_params = rcpFromRef(params->sublist("Mechanics Discretization").sublist("Discretization"));
        //
        m_problem = rcp(new CTM::MechanicsProblem(mech_params, param_lib, dim, comm));
        mech_params->validateParameters(*(m_problem->getValidProblemParameters()), 0);
        m_problem->buildProblem(mesh_specs, m_state_mgr);

        // create the initial discretization object
        auto neq = t_problem->numEquations();
        t_disc = disc_factory->createDiscretization(
                neq,
                t_problem->getSideSetEquations(),
                t_state_mgr.getStateInfoStruct(),
                t_state_mgr.getSideSetStateInfoStruct(),
                t_problem->getFieldRequirements(),
                t_problem->getSideSetFieldRequirements(),
                t_problem->getNullSpace());

        t_sol_info = Teuchos::rcp(new SolutionInfo());
        //
        t_sol_info->resize(t_disc, true);

        // create the initial discretization object
        neq = m_problem->numEquations();
        // set new discretization parameter list
        disc_factory->setDiscretizationParameters(disc_m_params);
        m_disc = disc_factory->createDiscretization(
                neq,
                m_problem->getSideSetEquations(),
                m_state_mgr.getStateInfoStruct(),
                m_state_mgr.getSideSetStateInfoStruct(),
                m_problem->getFieldRequirements(),
                m_problem->getSideSetFieldRequirements(),
                m_problem->getNullSpace());

        m_sol_info = Teuchos::rcp(new SolutionInfo());
        //
        m_sol_info->resize(m_disc, false);


    }

    void Solver::solve() {
        // get some information
        Teuchos::RCP<const Teuchos::ParameterList> p = Teuchos::rcpFromRef(params->sublist("Linear Algebra"));
        int max_iter = p->get<int>("Nonlinear Max. Iterations");
        double tolerance = p->get<double>("Nonlinear Tolerance");


        ///////////////////
        // get the solution information for thermal problem
        ///////////////////
        Teuchos::RCP<Tpetra_Vector> u_t = t_sol_info->getOwnedMV()->getVectorNonConst(0);
        //
        Teuchos::RCP<Tpetra_Vector> v_t = (t_sol_info->getOwnedMV()->getNumVectors() > 1) ? t_sol_info->getOwnedMV()->getVectorNonConst(1) : Teuchos::null;
        //
        Teuchos::RCP<Tpetra_Vector> xdotdot_t = (t_sol_info->getOwnedMV()->getNumVectors() > 2) ? t_sol_info->getOwnedMV()->getVectorNonConst(2) : Teuchos::null;
        // get residual
        Teuchos::RCP<Tpetra_Vector> r_t = t_sol_info->getOwnedResidual();
        // get Jacobian
        Teuchos::RCP<Tpetra_CrsMatrix> J_t = t_sol_info->getOwnedJacobian();

        // create new vectors
        Teuchos::RCP<const Tpetra_Map> t_map_owned = t_disc->getMapT();
        Teuchos::RCP<Tpetra_Vector> u_v_t = Teuchos::rcp(new Tpetra_Vector(t_map_owned));
        // incremental solution
        Teuchos::RCP<Tpetra_Vector> du_t = Teuchos::rcp(new Tpetra_Vector(t_map_owned));
        ///////////////////


        ///////////////////
        // get the solution information for mechanics problem
        ///////////////////
        Teuchos::RCP<Tpetra_Vector> u_m = m_sol_info->getOwnedMV()->getVectorNonConst(0);
        //
        Teuchos::RCP<Tpetra_Vector> v_m = (m_sol_info->getOwnedMV()->getNumVectors() > 1) ? m_sol_info->getOwnedMV()->getVectorNonConst(1) : Teuchos::null;
        //
        Teuchos::RCP<Tpetra_Vector> xdotdot_m = (m_sol_info->getOwnedMV()->getNumVectors() > 2) ? m_sol_info->getOwnedMV()->getVectorNonConst(2) : Teuchos::null;
        // get residual
        Teuchos::RCP<Tpetra_Vector> r_m = m_sol_info->getOwnedResidual();
        // get Jacobian
        Teuchos::RCP<Tpetra_CrsMatrix> J_m = m_sol_info->getOwnedJacobian();

        // create new vectors
        Teuchos::RCP<const Tpetra_Map> m_map_owned = m_disc->getMapT();
        Teuchos::RCP<Tpetra_Vector> u_v_m = Teuchos::rcp(new Tpetra_Vector(m_map_owned));
        // incremental solution
        Teuchos::RCP<Tpetra_Vector> du_m = Teuchos::rcp(new Tpetra_Vector(m_map_owned));
        ///////////////////

        // Set thermal application
        Teuchos::RCP<CTM::Application> t_application =
                Teuchos::rcp(new CTM::Application(params, comm, t_sol_info, t_problem, t_disc, t_state_mgr, m_state_mgr, true));


        // Set mechanics application
        Teuchos::RCP<CTM::Application> m_application =
                Teuchos::rcp(new CTM::Application(params, comm, m_sol_info, m_problem, m_disc, t_state_mgr, m_state_mgr, false));

        // Get thermal discretization
        Teuchos::RCP<Albany::APFDiscretization> apf_t_disc =
                Teuchos::rcp_dynamic_cast<Albany::APFDiscretization>(t_disc);

        t_state_mgr.setStateArrays(t_disc);

        // Get mechanics discretization
        Teuchos::RCP<Albany::APFDiscretization> apf_m_disc =
                Teuchos::rcp_dynamic_cast<Albany::APFDiscretization>(m_disc);

        m_state_mgr.setStateArrays(m_disc);

        apf_t_disc->initTemperatureHack();

        // time loop
        double norm;
        *out << std::endl;
        for (int step = 1; step <= num_steps; ++step) {
            *out << "*** Time Step: " << step << std::endl;
            *out << "*** from time: " << t_old << std::endl;
            *out << "*** to time: " << t_current << std::endl;

            // compute fad coefficients
            double omega = 0.0;
            double beta = 1.0; // (j_coeff in workset)
            double alpha = 1.0 / dt; // (m_coeff in workset)

            // predictor phase
            u_v_t->assign(*u_t);
            //
            int iter = 1;
            bool converged = false;
            // start newton loop
            v_t->update(alpha, *u_t, -alpha, *u_v_t, 0.0);
            *out << "Solving thermal problem" << std::endl;
            while ((iter <= max_iter) && (!converged)) {
                *out << "  " << iter << " newton iteration" << std::endl;
                // compute residual
                t_application->computeGlobalResidualT(t_current, t_old, v_t.get(),
                        xdotdot_t.get(), *u_t, *r_t);
                // compute Jacobian
                t_application->computeGlobalJacobianT(alpha, beta, omega,
                        t_current, t_old, v_t.get(), xdotdot_t.get(), *u_t, r_t.get(), *J_t);
                // scale residual
                r_t->scale(-1.0);
                //
                du_t->putScalar(0.0);
                // solve the linear system of equations
                solve_linear_system(p, J_t, du_t, r_t);
                // update solution
                u_t->update(1.0, *du_t, 1.0);
                v_t->update(alpha, *u_t, -alpha, *u_v_t, 0.0);
                // compute residual
                t_application->computeGlobalResidualT(t_current, t_old, v_t.get(),
                        xdotdot_t.get(), *u_t, *r_t);
                // compute norm
                norm = r_t->norm2();
                *out << "  ||r|| = " << norm << std::endl;
                if (norm < tolerance) converged = true;
                iter++;
                // 
            } // end newton loop
            TEUCHOS_TEST_FOR_EXCEPTION((iter > max_iter) && (!converged), std::out_of_range,
                    "\nnewton's method failed in " << max_iter << " iterations" << std::endl);
            // predictor
            u_v_m->assign(*u_m);
            // update thermal states
            t_application->evaluateStateFieldManagerT(t_current, t_old, *(t_sol_info->getOwnedMV()));
            //
            t_state_mgr.updateStates();
            apf_t_disc->writeSolutionToMeshDatabaseT(*(t_sol_info->getGhostMV()->getVector(0)), t_current, true);
            iter = 1;
            converged = false;
            *out << "Solving mechanics problem" << std::endl;
            //            apf_t_disc->initTemperatureHack();
            while ((iter <= max_iter) && (!converged)) {
                *out << "  " << iter << " newton iteration" << std::endl;
                // compute residual
                m_application->computeGlobalResidualT(t_current, t_old, v_m.get(),
                        xdotdot_m.get(), *u_m, *r_m);
                // compute Jacobian
                m_application->computeGlobalJacobianT(alpha, beta, omega,
                        t_current, t_old, v_m.get(), xdotdot_m.get(), *u_m, r_m.get(), *J_m);
                // scale residual
                r_m->scale(-1.0);
                //
                du_m->putScalar(0.0);
                // solve the linear system of equations
                solve_linear_system(p, J_m, du_m, r_m);
                // update solution
                u_m->update(1.0, *du_m, 1.0);
                // compute residual
                m_application->computeGlobalResidualT(t_current, t_old, v_m.get(),
                        xdotdot_m.get(), *u_m, *r_m);
                // compute norm
                norm = r_m->norm2();
                *out << "  ||r|| = " << norm << std::endl;
                if (norm < tolerance) converged = true;
                iter++;
                // 
            } // end newton loop
            m_application->evaluateStateFieldManagerT(t_current, t_old, *(m_sol_info->getOwnedMV()));
            //
            m_state_mgr.updateStates();
            apf_m_disc->writeSolutionT(*(m_sol_info->getGhostMV()->getVector(0)), t_current, true);
            TEUCHOS_TEST_FOR_EXCEPTION((iter > max_iter) && (!converged), std::out_of_range,
                    "\nnewton's method failed in " << max_iter << " iterations" << std::endl);

            // updates
            t_old = t_current;
            t_current = t_current + dt;
        }
    }
}
