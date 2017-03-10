#ifndef CTM_SOLVER_HPP
#define CTM_SOLVER_HPP

#include <Albany_DataTypes.hpp>
#include <PHAL_Workset.hpp>

namespace Albany {
class AbstractProblem;
class AbstractDiscretization;
class DiscretizationFactory;
class StateManager;
class MeshSpecsStruct;
} // namespace Albany

namespace CTM {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::ParameterList;

class SolutionInfo;

class Solver {

  public:

    Solver(RCP<const Teuchos_Comm> comm, RCP<ParameterList> p);

    void solve();

  private:

    RCP<const Teuchos_Comm> comm;

    RCP<ParameterList> params;
    RCP<ParameterList> temp_params;
    RCP<ParameterList> mech_params;

    RCP<ParamLib> param_lib;
    RCP<DistParamLib> dist_param_lib;

    RCP<Albany::DiscretizationFactory> disc_factory;
    RCP<Albany::AbstractDiscretization> temp_disc;
    RCP<Albany::AbstractDiscretization> mech_disc;

    ArrayRCP<RCP<Albany::MeshSpecsStruct> > mesh_specs;
    RCP<Albany::StateManager> temp_state_mgr;
    RCP<Albany::StateManager> mech_state_mgr;

    RCP<Albany::AbstractProblem> temp_problem;
    RCP<Albany::AbstractProblem> mech_problem;

    int num_steps;
    double dt;
    double t_old;
    double t_current;

    RCP<Teuchos::FancyOStream> out;

    void set_params(RCP<ParameterList> p);
    void initial_setup();
    void solve_temp();
    void solve_mech();
    void adapt_mesh();

};

} // namespace CTM


#endif
