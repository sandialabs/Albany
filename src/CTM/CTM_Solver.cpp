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
        sol_info->resize(disc,true);

    }

    void Solver::solve() {
        
    }

}
