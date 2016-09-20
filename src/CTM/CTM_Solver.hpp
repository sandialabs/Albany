#ifndef CTM_SOLVER_HPP
#define CTM_SOLVER_HPP

/// \file CTM_Solver.hpp

#include "CTM_Teuchos.hpp"
#include <Albany_DataTypes.hpp>

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

    double t_current;
    double t_previous;

};

}

#endif
