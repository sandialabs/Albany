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

static RCP<Solver> build_muelu_solver(
    RCP<const ParameterList> in, RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x, RCP<Tpetra_Vector> b) {
  auto muelu_params = in->sublist("Preconditioner");
  auto belos_params = get_belos_params(in);
  auto AA = (RCP<OP>)A;
  auto M = MueLu::CreateTpetraPreconditioner(AA, muelu_params);
  auto problem = rcp(new LinearProblem(A, x, b));
  problem->setLeftPrec(M);
  problem->setProblem();
  auto solver = rcp(new GmresSolver(problem, belos_params));
  return solver;
}

static void get_inv_row_sum(RCP<Tpetra_CrsMatrix> A, RCP<Tpetra_Vector> s) {
  s->putScalar(0.0);
  auto view = s->get1dViewNonConst();
  for (size_t row = 0; row < s->getLocalLength(); ++row) {
    auto num_entries = A->getNumEntriesInLocalRow(row);
    Teuchos::Array<LO> indices(num_entries);
    Teuchos::Array<ST> values(num_entries);
    A->getLocalRowCopy(row, indices(), values(), num_entries);
    ST sum = 0.0;
    for (size_t j = 0; j < num_entries; ++j)
      sum += std::abs(values[j]);
    if (sum < 1.0e-14)
      view[row] = 0.0;
    else
      view[row] = 1.0/sum;
  }
}

static void scale_system(
    RCP<const ParameterList> in,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> b,
    RCP<Teuchos::FancyOStream> out) {
  bool should_scale = false;
  if (in->isType<bool>("Linear Row Sum Scaling"))
    should_scale = in->get<bool>("Linear Row Sum Scaling");
  if (! should_scale) return;
  *out << "  scaling linear system\n";
  auto map = b->getMap();
  auto scale = rcp(new Tpetra_Vector(map));
  get_inv_row_sum(A, scale);
  b->elementWiseMultiply(1.0, *scale, *b, 0.0);
  A->leftScale(*scale);
}

static RCP<Solver> build_solver(
    RCP<const ParameterList> in,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b) {
  RCP<Solver> solver;
  solver = build_muelu_solver(in, A, x, b);
  return solver;
}

void solve_linear_system(
    RCP<const ParameterList> in,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b) {
  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  double t0 = PCU_Time();
  *out << "  linear system # equations: " <<  x->getGlobalLength() << std::endl;
  scale_system(in, A, b, out);
  RCP<Solver> solver = build_solver(in, A, x, b);
  solver->solve();
  int iters = solver->getNumIters();
  double t1 = PCU_Time();
  if (iters >= in->get<int>("Linear Max Iterations")) {
    *out << "  linear solve failed to converge in " << iters << " iterations" << std::endl;
    *out << "  continuing using the incomplete solve..." << std::endl;
  } else {
    *out << "  linear system solved in " << t1 - t0 << " seconds" << std::endl;
  }
}

} // namespace CTM
