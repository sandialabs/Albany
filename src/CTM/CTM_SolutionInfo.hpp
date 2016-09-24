#ifndef CTM_SOLUTION_INFO_HPP
#define CTM_SOLUTION_INFO_HPP

#include "CTM_Teuchos.hpp"
#include <Albany_DataTypes.hpp>

namespace Albany {
class AbstractDiscretization;
}

namespace CTM {

class SolutionInfo {

  public:

    SolutionInfo();
    SolutionInfo(const SolutionInfo&) = delete;
    SolutionInfo& operator=(const SolutionInfo&) = delete;

    RCP<Tpetra_MultiVector> owned_x;
    RCP<Tpetra_MultiVector> ghost_x;

    RCP<Tpetra_Vector> owned_f;
    RCP<Tpetra_Vector> ghost_f;

    RCP<Tpetra_CrsMatrix> owned_J;
    RCP<Tpetra_CrsMatrix> ghost_J;

    RCP<Tpetra_Export> exporter;
    RCP<Tpetra_Import> importer;

    void gather_x();
    void scatter_x();

    void gather_f();
    void scatter_f();

    void gather_J();
    void scatter_J();

    void resize(RCP<Albany::AbstractDiscretization> disc, bool have_x_dot);

};

}

#endif
