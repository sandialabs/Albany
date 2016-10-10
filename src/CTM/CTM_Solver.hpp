#ifndef CTM_SOLVER_HPP
#define CTM_SOLVER_HPP

/// \file CTM_Solver.hpp

#include "CTM_Teuchos.hpp"

#include <Albany_DataTypes.hpp>
#include "Albany_StateManager.hpp"
#include <PHAL_Workset.hpp>

namespace Albany {
    class AbstractProblem;
    class AbstractDiscretization;
    class DiscretizationFactory;
    class StateManager;
    struct MeshSpecsStruct;
}

namespace CTM {
    class SolutionInfo;
}

/// \brief All CTM symbols are contained in this namespace.
namespace CTM {

    /// \brief The coupled thermomechanics solver.
    /// \details The intent of this solver is to provide an interface to solve
    /// adaptive one-way coupled thermomechanics problems by performing
    /// successive iterations of the following steps:
    ///
    /// - Build all data structures for the temperature problem.
    /// - Solve the temperature problem at time t_n.
    /// - Attach the temperature field T to the discretization.
    /// - Destroy all data structures for the temperature problem.
    /// - Build all data structures for the mechanics problem.
    /// - Solve the mechnaics problem at time t_n using the temperature field T.
    /// - Attach the mechanics fields u to the discretization.
    /// - Destroy all data structures for the mechanics problem.
    /// - Adapt the mesh if desired, performing solution transfer for T and u.

    class Solver {
    public:

        /// \brief Constructor.
        /// \param comm The global Teuchos communicator object.
        /// \param params The global CTM input file parameters:
        ///
        /// Parameter Name    | Parameter Type
        /// --------------    | --------------
        /// "Discretization"  | sublist
        /// "Temperature"     | sublist
        /// "Mechanics"       | sublist
        /// "Linear Algebra"  | sublist
        /// "Time"            | sublist
        Solver(RCP<const Teuchos_Comm> comm, RCP<ParameterList> params);

        /// \brief Solve the problem for the entire specified time domain.
        void solve();

    private:

        RCP<const Teuchos_Comm> comm;
        RCP<ParameterList> params;
        RCP<ParameterList> temp_params;
        RCP<ParameterList> mech_params;

        RCP<ParamLib> param_lib;
        RCP<DistParamLib> dist_param_lib;

        RCP<Albany::DiscretizationFactory> disc_factory;
        RCP<Albany::AbstractDiscretization> disc;
        ArrayRCP<RCP<Albany::MeshSpecsStruct> > mesh_specs;
        Albany::StateManager state_mgr;
        Teuchos::RCP<SolutionInfo> sol_info;

        RCP<Albany::AbstractProblem> t_problem;
        RCP<Albany::AbstractProblem> m_problem;

        double t_old;
        double t_current;
        double dt;
        int num_steps;
        
        // Output stream, defaults to printing just Proc 0
       Teuchos::RCP<Teuchos::FancyOStream> out;

        void initial_setup();

    };

}

#endif
