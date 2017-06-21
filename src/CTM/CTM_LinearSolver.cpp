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
#include <Albany_AbstractDiscretization.hpp>
#include <Albany_SimDiscretization.hpp>

namespace CTM {

using Teuchos::rcp_dynamic_cast;

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
    RCP<const ParameterList> in,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b,
    RCP<Tpetra_MultiVector> coords,
    RCP<Tpetra_MultiVector> nullspace) {
  auto muelu_params = in->sublist("Preconditioner");
  auto belos_params = get_belos_params(in);
  auto AA = (RCP<OP>)A;
  auto M = MueLu::CreateTpetraPreconditioner(AA, muelu_params, coords, nullspace);
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

static RCP<Tpetra_MultiVector> get_coords(
    RCP<Albany::AbstractDiscretization> d,
    RCP<Teuchos::FancyOStream> out) {

  *out << "  linear solver: passing coordinates to muelu" << std::endl;

  // get the underlying APF mesh structures
  auto sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(d);
  auto apf_ms = sim_disc->getAPFMeshStruct();
  auto mesh = apf_ms->getMesh();
  auto coord_f = mesh->getCoordinateField();
  auto nodes = sim_disc->getOwnedNodes();

  // create the coord multivector
  int dim = mesh->getDimension();
  auto node_map = d->getNodeMapT();
  auto coord_mv = rcp(new Tpetra_MultiVector(node_map, dim, false));

  // fill in the coord multivector
  for (std::size_t i = 0; i < nodes.getSize(); ++i) {
    auto n = nodes[i];
    double lcoords[3];
    apf::getComponents(coord_f, n.entity, n.node, lcoords);
    for (int j = 0; j < dim; ++j)
      coord_mv->replaceLocalValue(i, j, lcoords[j]);
  }

  return coord_mv;
}

#if 0
static void subtract_centroid(RCP<Tpetra_MultiVector> coord_mv) {

  // get dimension information
  LO num_nodes = coord_mv->getLocalLength();
  int num_dim = coord_mv->getNumVectors();

  // compute the centroid
  ST centroid[3];
  {
    ST sum[3];
    for (int i = 0; i < num_dim; ++i) {
      auto x = coord_mv->getData(i);
      sum[i] = 0.0;
      for (int j = 0; j < num_nodes; ++j)
        sum[i] += x[j];
    }
    Teuchos::reduceAll<int, ST>(
        *coord_mv->getMap()->getComm(), Teuchos::REDUCE_SUM, num_dim, sum, centroid);
    for (int i = 0; i < num_dim; ++i)
      centroid[i] /= num_nodes;
  }

  // subtract the centroid
  for (int i = 0 ; i < num_dim; ++i) {
    auto x = coord_mv->getDataNonConst(i);
    for (int j = 0; j < num_nodes; ++j)
      x[j] -= centroid[i];
  }
}

static double& get_val(
    RCP<Tpetra_MultiVector> x, LO dof, const int i, const int j) {
  auto data = x->getDataNonConst(j);
  return data[dof + i];
}

static void coord_to_rbm(
    RCP<Tpetra_MultiVector> coord_mv,
    RCP<Tpetra_MultiVector> null_mv,
    int num_dofs,
    int num_scalar_dofs,
    int null_space_dim) {

  // get index dimensions
  LO num_nodes = coord_mv->getLocalLength();
  int num_dim = coord_mv->getNumVectors();
  LO vec_length = num_nodes * num_dofs;

  // get coordinate data
  auto x = coord_mv->getData(0);
  auto y = coord_mv->getData(1);
  auto z = coord_mv->getData(2);

  for (int node = 0; node < num_nodes; ++node) {
    int dof = node * num_dofs;
    switch( num_dofs - num_scalar_dofs) {
      case 6:
        for (int i = 3; i < 6 + num_scalar_dofs; ++i)
        for (int j = 0; j < 6 + num_scalar_dofs; ++j)
          get_val(null_mv, dof, i, j) = (i == j) ? 1.0 : 0.0;
        // no break on purpose
      case 3:
        for (int i = 0; i < 3 + num_scalar_dofs; ++i)
        for (int j = 0; j < 3 + num_scalar_dofs; ++j)
          get_val(null_mv, dof, i, j) = (i == j) ? 1.0: 0.0;
        for (int i = 0; i < 3; ++i) {
          for (int j = 3 + num_scalar_dofs; j < 6 + num_scalar_dofs; ++j) {
            if (i == (j-3-num_scalar_dofs)) get_val(null_mv, dof, i, j) =  0.0;
            else {
              if ((i+j) == (4 + num_scalar_dofs)) get_val(null_mv, dof, i, j) =  z[node];
              else if ((i+j) == (5 + num_scalar_dofs)) get_val(null_mv, dof, i, j) =  y[node];
              else if ((i+j) == (6 + num_scalar_dofs)) get_val(null_mv, dof, i, j) = x[node];
              else get_val(null_mv, dof, i, j) = 0.0;
            }
          }
        }
        get_val(null_mv, dof, 0, 5 + num_scalar_dofs) *= -1.0;
        get_val(null_mv, dof, 1, 3 + num_scalar_dofs) *= -1.0;
        get_val(null_mv, dof, 2, 4 + num_scalar_dofs) *= -1.0;
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::logic_error, "coord_to_rbm: unknown num_dofs");
    }
  }
}

static RCP<Tpetra_MultiVector> get_nullspace(
    RCP<Albany::AbstractDiscretization> d,
    RCP<Tpetra_MultiVector> coord_mv,
    RCP<Teuchos::FancyOStream> out) {

  *out << "  linear solver: passing nullspace to MueLu" << std::endl;

  // create the nullspace multivector
  // valid for 3D displacement-based mechanics only!
  auto dof_map = d->getMapT();
  int num_pdes = 3;
  int num_scalar_pdes = 0;
  int null_space_dim = 6;
  auto null_mv = rcp(new Tpetra_MultiVector(dof_map, null_space_dim, false));

  // subtract centroid
  subtract_centroid(coord_mv);

  // coordinate to rigid body mode
  coord_to_rbm(coord_mv, null_mv, num_pdes, num_scalar_pdes, null_space_dim);

  return null_mv;
}
#endif

void solve_linear_system(
    RCP<const ParameterList> in,
    RCP<Tpetra_CrsMatrix> A,
    RCP<Tpetra_Vector> x,
    RCP<Tpetra_Vector> b,
    RCP<Albany::AbstractDiscretization> d) {

  // useful timing info
  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  double t0 = PCU_Time();
  *out << "  linear system # equations: " <<  x->getGlobalLength() << std::endl;

  // scale the linear system if specified
  // not sure this actually helps at all ?
  scale_system(in, A, b, out);

  // get the coordinates and the null space if specified
  RCP<Tpetra_MultiVector> coords;
  RCP<Tpetra_MultiVector> nullspace;
  if (d != Teuchos::null) {
    coords = get_coords(d, out);
  }

  // build the solver and solve
  RCP<Solver> solver = build_muelu_solver(in, A, x, b, coords, nullspace);
  solver->solve();

  // print some final information
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
