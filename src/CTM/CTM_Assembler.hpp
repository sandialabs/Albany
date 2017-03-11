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

    ArrayRCP<RCP<Albany::MeshSpecsStruct> > mesh_specs;
    ArrayRCP<RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > fm;
    ArrayRCP<RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > nfm;
    RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm;
    Teuchos::Array<RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > sfm;

    void initial_setup();

    void set_initial_conditions();

    void load_ws_basic(
        PHAL::Workset& workset, const double t_new, const double t_old);

    void load_ws_bucket(PHAL::Workset& workset, const int ws);

    void load_ws_jacobian(
        PHAL::Workset& workset, const double alpha, const double beta,
        const double omega);

    void load_ws_nodeset(PHAL::Workset& workset);

    void post_reg_setup();

};

} // namespace CTM

#endif
