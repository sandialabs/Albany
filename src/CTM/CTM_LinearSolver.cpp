#include "CTM_LinearSolver.hpp"

#include <PCU.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_Describable.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_Factory.hpp>
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Albany_Utils.hpp>

namespace CTM {

typedef Tpetra_MultiVector MV;
typedef Tpetra_CrsMatrix MAT;
typedef Tpetra_Operator OP;
typedef Tpetra_RowMatrix RM;
typedef Belos::LinearProblem<ST, MV, OP> LinearProblem;
typedef Belos::SolverManager<ST, MV, OP> Solver;
typedef Belos::BlockGmresSolMgr<ST, MV, OP> GmresSolver;
typedef Tpetra_Operator Prec;
typedef Ifpack2::Preconditioner<ST, LO, GO, KokkosNode> IfpackPrec;

static RCP<ParameterList> get_ifpack2_params() {
  RCP<ParameterList> p = rcp(new ParameterList);
  p->set("fact: drop tolerance", 0.0);
  p->set("fact: ilut level-of-fill", 1.0);
  return p;
}

static RCP<ParameterList> get_muelu_params() {
  RCP<ParameterList> p = rcp(new ParameterList);
  return p;
}

static RCP<ParameterList> get_belos_params(RCP<const ParameterList> in) {
  RCP<ParameterList> p = rcp(new ParameterList);
  int max_iters = in->get<int>("Linear Max Iterations");
  int krylov = in->get<int>("Linear Krylov Size");
  double tol = in->get<double>("Linear Tolerance");
  p->set("Block Size", 1);
  p->set("Num Blocks", krylov);
  p->set("Maximum Iterations", max_iters);
  p->set("Convergence Tolerance", tol);
  p->set("Orthogonalization", "DGKS");
  p->set("Verbosity", 33);
  p->set("Output Style", 1);
  p->set("Output Frequency", 20);
  return p;
}

static RCP<Solver> build_ifpack2_solver(
    RCP<const ParameterList> in, RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x, RCP<Tpetra_Vector> b) {

  auto ifpack2_params = get_ifpack2_params();
  auto belos_params = get_belos_params(in);
  Ifpack2::Factory factory;
  auto P = factory.create<RM>("ILUT", A);
  P->setParameters(*ifpack2_params);
  P->initialize();
  P->compute();
  auto problem = rcp(new LinearProblem(A, x, b));
  problem->setLeftPrec(P);
  problem->setProblem();
  auto solver = rcp(new GmresSolver(problem, belos_params));
  return solver;
}

static RCP<Solver> build_muelu_solver(
    RCP<const ParameterList> in, RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x, RCP<Tpetra_Vector> b) {
  auto muelu_params = get_muelu_params();
  auto belos_params = get_belos_params(in);
  auto AA = (RCP<OP>)A;
  auto M = MueLu::CreateTpetraPreconditioner(AA, *muelu_params);
  auto problem = rcp(new LinearProblem(A, x, b));
  problem->setLeftPrec(M);
  problem->setProblem();
  auto solver = rcp(new GmresSolver(problem, belos_params));
  return solver;
}

static RCP<Solver> build_solver(
    RCP<const ParameterList> in,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b) {
  auto prec = in->get<std::string>("Preconditioner Type");
  RCP<Solver> solver;
  if (prec == "Ifpack2")
    solver = build_ifpack2_solver(in, A, x, b);
  else if (prec == "MueLu")
    solver = build_muelu_solver(in, A, x, b);
  else
    ALBANY_ALWAYS_ASSERT_VERBOSE(false,
        "unknown `Preconditioner Type`");
  return solver;
}

void solve_linear_system(
    RCP<const ParameterList> in,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b) {
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  double t0 = PCU_Time();
  double t1;
  RCP<Solver> solver = build_solver(in, A, x, b);
  solver->solve();
  int iters = solver->getNumIters();
  t1 = PCU_Time();
  if (iters >= in->get<int>("Linear Max Iterations")) {
    *out << "  linear solve failed to converge in " << iters << " iterations" << std::endl;
    *out << "  continuing using the incomplete solve..." << std::endl;
  } else {
    *out << "  linear system solved in " << t1 - t0 << " seconds" << std::endl;
  }
}

} // namespace CTM
