#ifndef CTM_SOLUTION_INFO_HPP
#define CTM_SOLUTION_INFO_HPP

#include <Albany_DataTypes.hpp>

namespace Albany {
class AbstractDiscretization;
}

namespace CTM {

using Teuchos::RCP;

struct LinearObj {
  RCP<Tpetra_Vector> x;
  RCP<Tpetra_Vector> x_dot;
  RCP<Tpetra_Vector> f;
  RCP<Tpetra_CrsMatrix> J;
};

class SolutionInfo {

  public:

    SolutionInfo();

    SolutionInfo(const SolutionInfo&) = delete;
    SolutionInfo& operator=(const SolutionInfo&) = delete;

    void gather_x();
    void gather_x_dot();
    void gather_f();
    void gather_J();

    void scatter_x();
    void scatter_x_dot();
    void scatter_f();
    void scatter_J();

    void resize(RCP<Albany::AbstractDiscretization> disc, bool have_x_dot);

    RCP<LinearObj> owned;
    RCP<LinearObj> ghost;

  private:
    
    RCP<Tpetra_Export> exporter;
    RCP<Tpetra_Import> importer;

};

} // namespace CTM

#endif
