#ifndef CTM_LINEAR_SOLVER_HPP
#define CTM_LINEAR_SOLVER_HPP

#include "Albany_DataTypes.hpp"

namespace CTM {

using Teuchos::RCP;
using Teuchos::ParameterList;

void solve_linear_system(
    RCP<const ParameterList> p,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b);

} // namespace CTM

#endif
