//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Unit tests of Albany::getTransposedOp and Albany::transpose(A,At), used by the adjoint
// model (see Albany::ModelEvaluator::create_W_op and evalModelImpl).
//
// The matrix to transpose is a FE matrix built as the Jacobian of the discretizations
// (ThyraCrsMatrixFactory with owned and overlapped spaces), assembled as in
// Application::computeGlobalJacobian (beginFEAssembly, element contributions, endFEAssembly),
// with Dirichlet-like rows (beginModify, row set to the identity, endModify).
// Run also in parallel: with one rank, the owned and the overlapped matrices of a
// Tpetra::FECrsMatrix are the same, and the transpose cannot break their sharing.

#include "Albany_CommUtils.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_TpetraTypes.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace {

// Structured 2D mesh of quads, with 2 dofs per node. The rows of elements are split among
// the ranks, and each rank owns the nodes of the lower side of its elements (and the last
// rank also those of the top side of the mesh).
struct QuadMesh
{
  static constexpr int numComps = 2;
  int nex, ney, nx;
  std::vector<int> elems;   // Local elements
  Teuchos::RCP<const Thyra_VectorSpace> vs, ov_vs;

  explicit QuadMesh (const Teuchos::RCP<const Teuchos_Comm>& comm)
  {
    const int numRanks = comm->getSize();
    const int rank     = comm->getRank();
    nex = 5;
    ney = 2*numRanks+1;
    nx  = nex+1;

    const auto firstElemRow = [&](const int r) { return r*ney/numRanks; };
    for (int j=firstElemRow(rank); j<firstElemRow(rank+1); ++j) {
      for (int i=0; i<nex; ++i) {
        elems.push_back(j*nex+i);
      }
    }

    const int firstNodeRow = firstElemRow(rank);
    const int lastNodeRow  = rank==numRanks-1 ? ney : firstElemRow(rank+1)-1;
    std::vector<GO> owned;
    for (int j=firstNodeRow; j<=lastNodeRow; ++j) {
      for (int i=0; i<nx; ++i) {
        for (int c=0; c<numComps; ++c) {
          owned.push_back(dof(j*nx+i,c));
        }
      }
    }
    // The overlapped space starts with the owned dofs, as the overlapped spaces of the
    // discretizations
    std::set<GO> ghosted;
    for (const int e : elems) {
      for (const GO g : elemDofs(e)) {
        if (std::find(owned.begin(),owned.end(),g)==owned.end()) {
          ghosted.insert(g);
        }
      }
    }
    std::vector<GO> overlapped = owned;
    overlapped.insert(overlapped.end(),ghosted.begin(),ghosted.end());

    vs    = Albany::createVectorSpace(comm,Teuchos::ArrayView<const GO>(owned));
    ov_vs = Albany::createVectorSpace(comm,Teuchos::ArrayView<const GO>(overlapped));
  }

  static GO dof (const int node, const int comp) { return numComps*static_cast<GO>(node)+comp; }

  std::vector<GO> elemDofs (const int e) const {
    const int i = e % nex;
    const int j = e / nex;
    const int nodes[4] = { j*nx+i, j*nx+i+1, (j+1)*nx+i+1, (j+1)*nx+i };
    std::vector<GO> dofs;
    for (const int n : nodes) {
      for (int c=0; c<numComps; ++c) {
        dofs.push_back(dof(n,c));
      }
    }
    return dofs;
  }

  bool onBottomEdge (const GO dof) const { return dof/numComps < nx; }

  // The Jacobian op, as the discretizations create it
  Teuchos::RCP<Thyra_LinearOp> createJacobianOp () const {
    Albany::ThyraCrsMatrixFactory factory(vs,vs,ov_vs,ov_vs);
    for (const int e : elems) {
      const auto dofs = elemDofs(e);
      factory.insertGlobalIndices(dofs,dofs,true);
    }
    factory.fillComplete();
    return factory.createOp();
  }

  // Assemble nonsymmetric element matrices depending on seed, then replace the rows of the
  // dofs on the bottom edge with the identity, as weak Dirichlet conditions do
  void assemble (const Teuchos::RCP<Thyra_LinearOp>& jac, const int seed) const {
    Albany::beginFEAssembly(jac);
    Albany::assign(jac,0.0);
    auto fe_mat = Albany::getTpetraFECrsMatrix(jac,true);
    for (const int e : elems) {
      const auto dofs = elemDofs(e);
      const std::vector<Tpetra_GO> cols(dofs.begin(),dofs.end());
      std::vector<ST> vals(cols.size());
      for (size_t a=0; a<cols.size(); ++a) {
        for (size_t b=0; b<cols.size(); ++b) {
          vals[b] = std::sin(0.37*seed + 1.3*a - 0.71*b + 0.05*e) + 0.1*a*b*seed + (a==b ? 4.0 : 0.0);
        }
        fe_mat->sumIntoGlobalValues(cols[a],Teuchos::ArrayView<const Tpetra_GO>(cols),
                                    Teuchos::ArrayView<const ST>(vals));
      }
    }
    Albany::endFEAssembly(jac);

    Albany::beginModify(jac);
    auto mat = Albany::getTpetraMatrix(jac,true);
    const auto rowMap = mat->getRowMap();
    const auto colMap = mat->getColMap();
    for (LO lrow=0; lrow<static_cast<LO>(rowMap->getLocalNumElements()); ++lrow) {
      const Tpetra_GO grow = rowMap->getGlobalElement(lrow);
      if (!onBottomEdge(grow)) {
        continue;
      }
      typename Tpetra_CrsMatrix::local_inds_host_view_type lcols;
      typename Tpetra_CrsMatrix::values_host_view_type     lvals;
      mat->getLocalRowView(lrow,lcols,lvals);
      std::vector<LO> indices(lcols.extent(0));
      std::vector<ST> values(lcols.extent(0));
      for (size_t k=0; k<indices.size(); ++k) {
        indices[k] = lcols(k);
        values[k]  = colMap->getGlobalElement(lcols(k))==grow ? 1.0 : 0.0;
      }
      Albany::setLocalRowValues(jac,lrow,Teuchos::ArrayView<const LO>(indices),
                                Teuchos::ArrayView<const ST>(values));
    }
    Albany::endModify(jac);
  }
};

// max_j ||At*x_j - A^T*x_j||_inf / ||A^T*x_j||_inf, for a few random x_j
double transposeError (const Teuchos::RCP<const Thyra_LinearOp>& At,
                       const Teuchos::RCP<const Thyra_LinearOp>& A)
{
  const int numVecs = 3;
  auto x  = Thyra::createMembers(A->range(),numVecs);
  auto y  = Thyra::createMembers(A->domain(),numVecs);
  auto yt = Thyra::createMembers(A->domain(),numVecs);
  Thyra::seed_randomize<ST>(42);
  Thyra::randomize(-1.0,1.0,x.ptr());
  At->apply(Thyra::NOTRANS,*x,yt.ptr(),1.0,0.0);
  A->apply(Thyra::TRANS,*x,y.ptr(),1.0,0.0);
  std::vector<ST> normsY(numVecs);
  Thyra::norms_inf(*y,Teuchos::arrayViewFromVector(normsY));
  Thyra::update(-1.0,*y,yt.ptr());
  std::vector<ST> normsDiff(numVecs);
  Thyra::norms_inf(*yt,Teuchos::arrayViewFromVector(normsDiff));
  double err = 0;
  for (int j=0; j<numVecs; ++j) {
    err = std::max(err,normsDiff[j]/normsY[j]);
  }
  return err;
}

bool compatibleSpaces (const Teuchos::RCP<const Thyra_LinearOp>& At,
                       const Teuchos::RCP<const Thyra_LinearOp>& A)
{
  return At->range()->isCompatible(*A->domain()) && At->domain()->isCompatible(*A->range());
}

constexpr double tol = 1e-13;

} // anonymous namespace

TEUCHOS_UNIT_TEST(TransposeUtils, GetTransposedOp)
{
  const auto comm = Albany::getDefaultComm();
  const QuadMesh mesh(comm);
  auto jac = mesh.createJacobianOp();
  mesh.assemble(jac,1);

  const auto jac_t = Albany::getTransposedOp(jac.getConst());

  // A regular matrix with a static graph, even if the input is a FE matrix
  TEST_ASSERT (Albany::getTpetraFECrsMatrix(jac_t,false).is_null());
  TEST_ASSERT (Albany::getTpetraMatrix(jac_t,true)->isStaticGraph());
  TEST_ASSERT (compatibleSpaces(jac_t,jac));
  TEST_COMPARE (transposeError(jac_t,jac), <, tol);
}

TEUCHOS_UNIT_TEST(TransposeUtils, RepeatedAssembly)
{
  // As in the adjoint model: W is created once (create_W_op) from the empty Jacobian, and
  // receives the transpose of each new Jacobian assembled in the same FE matrix
  const auto comm = Albany::getDefaultComm();
  const QuadMesh mesh(comm);
  auto jac = mesh.createJacobianOp();
  const auto W = Albany::getTransposedOp(jac.getConst());
  const auto W_mat   = Albany::getTpetraMatrix(W,true);
  const auto W_graph = W_mat->getCrsGraph();

  for (int seed=1; seed<=4; ++seed) {
    out << "Assembly " << seed << "\n";
    mesh.assemble(jac,seed);
    Albany::transpose(jac.getConst(),W);
    TEST_COMPARE (transposeError(W,jac), <, tol);
    TEST_ASSERT (compatibleSpaces(W,jac));

    // Only the values changed: same matrix, same graph
    const auto mat = Albany::getTpetraMatrix(W,true);
    TEST_EQUALITY (mat.get(), W_mat.get());
    TEST_EQUALITY (mat->getCrsGraph().get(), W_graph.get());
  }
}

TEUCHOS_UNIT_TEST(TransposeUtils, StructureChange)
{
  // W created for a matrix with another graph (the identity on the same space): its
  // matrix is replaced by the first transpose, then only its values change
  const auto comm = Albany::getDefaultComm();
  const QuadMesh mesh(comm);
  auto jac = mesh.createJacobianOp();

  Albany::ThyraCrsMatrixFactory factory(mesh.vs,mesh.vs);
  for (const GO g : Albany::getGlobalElements(mesh.vs)) {
    factory.insertGlobalIndices(g,g);
  }
  factory.fillComplete();
  auto id = factory.createOp();
  Albany::assign(id,1.0);
  const auto W = Albany::getTransposedOp(id.getConst());
  const auto id_graph = Albany::getTpetraMatrix(W,true)->getCrsGraph();

  mesh.assemble(jac,1);
  Albany::transpose(jac.getConst(),W);
  TEST_COMPARE (transposeError(W,jac), <, tol);
  TEST_ASSERT (compatibleSpaces(W,jac));
  const auto W_graph = Albany::getTpetraMatrix(W,true)->getCrsGraph();
  TEST_INEQUALITY (W_graph.get(), id_graph.get());
  TEST_ASSERT (Albany::getTpetraMatrix(W,true)->isStaticGraph());

  mesh.assemble(jac,2);
  Albany::transpose(jac.getConst(),W);
  TEST_COMPARE (transposeError(W,jac), <, tol);
  TEST_EQUALITY (Albany::getTpetraMatrix(W,true)->getCrsGraph().get(), W_graph.get());
}

TEUCHOS_UNIT_TEST(TransposeUtils, FEMatrixOutput)
{
  // The values of a FE matrix cannot be replaced: refused as output
  const auto comm = Albany::getDefaultComm();
  const QuadMesh mesh(comm);
  auto jac = mesh.createJacobianOp();
  mesh.assemble(jac,1);
  auto other = mesh.createJacobianOp();
  TEST_THROW (Albany::transpose(jac.getConst(),other), std::logic_error);
}
