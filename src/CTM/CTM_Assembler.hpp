#ifndef CTM_ASSEMBLER_HPP
#define CTM_ASSEMBLER_HPP

#include <Albany_AbstractProblem.hpp>
#include <PHAL_Workset.hpp>

namespace CTM {

class SolutionInfo;

using Teuchos::RCP;
using Teuchos::ArrayRCP;

class Assembler {

  public:

    Assembler(
        RCP<SolutionInfo> s_info,
        RCP<Albany::AbstractProblem> prob,
        RCP<Albany::AbstractDiscretization> d,
        RCP<Albany::StateManager> sm);

    void assmeble_system(
        const double alpha,
        const double beta,
        const double omega,
        const double t_new,
        const double t_old,
        RCP<Tpetra_Vector> x,
        RCP<Tpetra_Vector> x_dot,
        RCP<Tpetra_Vector> f,
        RCP<Tpetra_CrsMatrix> J);

    void assemble_state(
        const double t_new,
        const double t_old,
        RCP<Tpetra_Vector> x,
        RCP<Tpetra_Vector> x_dot);

  private:

    int neq;

    RCP<SolutionInfo> sol_info;
    RCP<Albany::AbstractProblem> problem;
    RCP<Albany::AbstractDiscretization> disc;
    RCP<Albany::StateManager> state_mgr;

    ArrayRCP<RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > fm;
    ArrayRCP<RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > nfm;
    RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm;
    Teuchos::Array<RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > sfm;

    void loadWorksetBucketInfo(PHAL::Workset& workset, const int ws);

    void loadWorksetSidesetInfo(PHAL::Workset& workset, const int ws);

    void loadBasicWorksetInfo(
        PHAL::Workset& workset, const double t_new, const double t_old);

    void loadWorksetJacobianInfo(
        PHAL::Workset& workset, const double alpha, const double beta,
        const double omega);

    void loadWorksetNodesetInfo(PHAL::Workset& workset);

    void postRegSetup();

};

} // namespace CTM

#endif
