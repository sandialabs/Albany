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

static RCP<ParameterList> get_belos_params(RCP<const ParameterList> in) {
  RCP<ParameterList> p = rcp(new ParameterList);
  int max_iters = in->get<int>("Linear Max. Iterations");
  int krylov = in->get<int>("Linear Krylov Size");
  double tol = in->get<double>("Linear Tolerance");
  p->set("Block Size", 1);
  p->set("Num Blocks", krylov);
  p->set("Maximum Iterations", max_iters);
  p->set("Convergence Tolerance", tol);
  p->set("Orthogonalization", "DGKS");
  return p;
}

static RCP<Prec> build_ifpack2_prec(RCP<Tpetra_CrsMatrix> A) {
  RCP<ParameterList> p = get_ifpack2_params();
  Ifpack2::Factory factory;
  RCP<IfpackPrec> prec = factory.create<RM>("ILUT", A);
  prec->setParameters(*p);
  prec->initialize();
  prec->compute();
  return prec;
}

static RCP<Prec> build_precond(RCP<const ParameterList> p, RCP<Tpetra_CrsMatrix> A) {
  (void) (p);
  return build_ifpack2_prec(A);
}

static RCP<Solver> build_solver(
    RCP<const ParameterList> in,
    RCP<Prec> P,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b) {
  RCP<ParameterList> p = get_belos_params(in);
  RCP<LinearProblem> problem = rcp(new LinearProblem(A, x, b));
  problem->setLeftPrec(P);
  problem->setProblem();
  RCP<Solver> solver = rcp(new GmresSolver(problem, p));
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
  RCP<Prec> P = build_precond(in, A);
  RCP<Solver> solver = build_solver(in, P, A, x, b);
  solver->solve();
  int iters = solver->getNumIters();
  t1 = PCU_Time();
  if (iters >= in->get<int>("Linear Max. Iterations")) {
    *out << "  linear solve failed to converge in " << iters << " iterations" << std::endl;
    *out << "  continuing using the incomplete solve..." << std::endl;
  } else {
    *out << "  linear system solved in " << iters << " iterations" << std::endl;
    *out << "  linear system solved in " << t1 - t0 << " seconds" << std::endl;
  }
}

} // namespace CTM
