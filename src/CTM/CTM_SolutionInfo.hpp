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

        // constructor
        SolutionInfo();
        // do not use copy constructor
        SolutionInfo(const SolutionInfo&) = delete;
        // do not use assignment operator
        SolutionInfo& operator=(const SolutionInfo&) = delete;

        // get solution vectors
        Teuchos::RCP<Tpetra_MultiVector> getOwnedMV();
        //
        Teuchos::RCP<Tpetra_MultiVector> getGhostMV();

        // get exporter
        Teuchos::RCP<Tpetra_Export> getExporter();
        // get importer
        Teuchos::RCP<Tpetra_Import> getImporter();

        // scatter
        void scatter_x(
                const Tpetra_Vector& xT, /* note that none are overlapped */
                const Tpetra_Vector* x_dotT,
                const Tpetra_Vector* x_dotdotT);

        // get residual vectors
        Teuchos::RCP<Tpetra_Vector> getOwnedResidual();
        //
        Teuchos::RCP<Tpetra_Vector> getGhostResidual();

        // get Jacobian
        Teuchos::RCP<Tpetra_CrsMatrix> getOwnedJacobian();
        //
        Teuchos::RCP<Tpetra_CrsMatrix> getGhostJacobian();

        void gather_x();
        void scatter_x();

        void gather_f();
        void scatter_f();

        void gather_J();
        void scatter_J();

        void resize(RCP<Albany::AbstractDiscretization> disc, bool have_x_dot);
    private:
        // solution multi vector. First column corresponds to main variable.
        // second column to time derivative of main variable.
        RCP<Tpetra_MultiVector> owned_x;
        RCP<Tpetra_MultiVector> ghost_x;
        // Residual vector.
        RCP<Tpetra_Vector> owned_f;
        RCP<Tpetra_Vector> ghost_f;
        // Jacobians
        RCP<Tpetra_CrsMatrix> owned_J;
        RCP<Tpetra_CrsMatrix> ghost_J;

        RCP<Tpetra_Export> exporter;
        RCP<Tpetra_Import> importer;

    };

}

#endif
