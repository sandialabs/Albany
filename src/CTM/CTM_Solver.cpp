#include "CTM_Solver.hpp"
#include "CTM_ThermalProblem.hpp"
#include <Albany_DiscretizationFactory.hpp>
#include <Albany_AbstractDiscretization.hpp>
#include "CTM_SolutionInfo.hpp"

namespace CTM {

    static RCP<ParameterList> get_valid_params() {
        auto p = rcp(new ParameterList);
        p->sublist("Discretization");
        p->sublist("Temperature Problem");
        //  p->sublist("Mechanics");
        p->sublist("Linear Algebra");
        p->sublist("Time");
    }

    static void validate_params(RCP<const ParameterList> p) {
        assert(p->isSublist("Discretization"));
        assert(p->isSublist("Temperature Problem"));
        //  assert(p->isSublist("Mechanics"));
        assert(p->isSublist("Linear Algebra"));
        assert(p->isSublist("Time"));
        //
        const Teuchos::ParameterList &time_list = p->sublist("Time");
        assert(time_list.isType<double>("Initial Time"));
        assert(time_list.isType<double>("Step Size"));
        assert(time_list.isType<int>("Number of Steps"));
        //
        const Teuchos::ParameterList &la_list = p->sublist("Linear Algebra");
        assert(la_list.isType<double>("Linear Tolerance"));
        assert(la_list.isType<int>("Linear Max. Iterations"));
        assert(la_list.isType<int>("Linear Krylov Size"));
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
        bool explicit_scheme = false;
        disc_factory = rcp(new Albany::DiscretizationFactory(params, comm, false));
        mesh_specs = disc_factory->createMeshSpecs();

        // create the problem objects
        auto dim = mesh_specs[0]->numDim;
        t_problem = rcp(new ThermalProblem(temp_params, param_lib, dim, comm));

        temp_params->validateParameters(*(t_problem->getValidProblemParameters()), 0);
        t_problem->buildProblem(mesh_specs, state_mgr);

        // create the initial discretization object
        auto neq = t_problem->numEquations();
        disc = disc_factory->createDiscretization(
                neq,
                t_problem->getSideSetEquations(),
                state_mgr.getStateInfoStruct(),
                state_mgr.getSideSetStateInfoStruct(),
                t_problem->getFieldRequirements(),
                t_problem->getSideSetFieldRequirements(),
                t_problem->getNullSpace());

        sol_info = Teuchos::rcp(new SolutionInfo());
        //
        sol_info->resize(disc, true);

    }

    void Solver::solve() {
        // get some information
        Teuchos::RCP<const Teuchos::ParameterList> p = Teuchos::rcpFromRef(params->sublist("Linear Algebra"));
        int max_iter = p->get<int>("Nonlinear Max. Iterations");
        double tolerance = p->get<double>("Nonlinear Tolerance");

        // get the solution information
        Teuchos::RCP<Tpetra_Vector> u = sol_info->owned_x->getVectorNonConst(0);
        // IMPORTANT: For now I am assuming that we have x_dot
        Teuchos::RCP<Tpetra_Vector> v = sol_info->owned_x->getVectorNonConst(1);
        // get residual
        Teuchos::RCP<Tpetra_Vector> r = sol_info->owned_f;
        // get Jacobian
        Teuchos::RCP<Tpetra_CrsMatrix> J = sol_info->owned_J;

        // create new vectors
        Teuchos::RCP<const Tpetra_Map> map_owned = disc->getMapT();
        Teuchos::RCP<Tpetra_Vector> u_v = Teuchos::rcp(new Tpetra_Vector(map_owned));
        // incremental solution
        Teuchos::RCP<Tpetra_Vector> du = Teuchos::rcp(new Tpetra_Vector(map_owned));

        // time loop
        *out << std::endl;
        for (int step = 1; step <= num_steps; ++step) {
            *out << "*** Time Step: " << step << std::endl;
            *out << "*** from time: " << t_old << std::endl;
            *out << "*** to time: " << t_current << std::endl;

            // compute fad coefficients
            double beta = 1.0 / dt; // (m_coeff in workset)
            double alpha = 1.0; // (j_coeff in workset))

            // predictor phase
            u_v->assign(*u);
            u_v->update(dt, *v, 1.0);
            //
            int iter = 1;
            bool converged = false;
            // start newton loop
            while ((iter <= max_iter) && (!converged)) {
                *out << "  " << iter << " newton iteration" << std::endl;
                v->update(beta, *u, -beta, *u_v, 0.0);
                // solve the linear system of equations
                // residual
                r->scale(-1.0);
                //
                du->putScalar(0.0);
                // update solution
                u->update(1.0, *du, 1.0);
                // compute residual
                // compute norm
                double norm = r->norm2();
                *out << "  ||r|| = " << norm << std::endl;
                if (norm < tolerance) converged = true;
                iter++;
                // 
            } // end newton loop 
            TEUCHOS_TEST_FOR_EXCEPTION((iter > max_iter) && (!converged), std::out_of_range,
                    "\nnewton's method failed in " << max_iter << " iterations" << std::endl);
            // updates
            t_old = t_current;
            t_current = t_new + dt;
        }

    }

}
