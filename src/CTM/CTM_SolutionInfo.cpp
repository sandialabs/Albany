#include "CTM_SolutionInfo.hpp"

#include <PCU.h>
#include <Albany_AbstractDiscretization.hpp>

namespace CTM {

using Teuchos::rcp;

SolutionInfo::SolutionInfo() {
  owned = rcp(new LinearObj);
  ghost = rcp(new LinearObj);
}

Teuchos::RCP<Tpetra_MultiVector> SolutionInfo::getOwnedMV() {
  return owned_x;
}

Teuchos::RCP<Tpetra_MultiVector> SolutionInfo::getGhostMV() {
  return ghost_x;
}

Teuchos::RCP<Tpetra_Export> SolutionInfo::getExporter() {
  return exporter;
}

Teuchos::RCP<Tpetra_Import> SolutionInfo::getImporter() {
  return importer;
}

void SolutionInfo::gather_x() {
  owned_x->doExport(*ghost_x, *exporter, Tpetra::INSERT);
}

void SolutionInfo::scatter_x() {
  ghost_x->doImport(*owned_x, *importer, Tpetra::INSERT);
}

void SolutionInfo::gather_f() {
  owned->f->doExport(*(ghost->f), *exporter, Tpetra::ADD);
}

void SolutionInfo::scatter_f() {
  ghost->f->doExport(*(owned->f), *importer, Tpetra::INSERT);
}

void SolutionInfo::gather_J() {
  owned->J->doExport(*(ghost->J), *exporter, Tpetra::ADD);
}

void SolutionInfo::scatter_J() {
  ghost->J->doImport(*(owned->J), *importer, Tpetra::INSERT);
}

void SolutionInfo::scatter_x(
    const Tpetra_Vector& xT,
    const Tpetra_Vector* x_dotT,
    const Tpetra_Vector* x_dotdotT) {

  ghost_x->getVectorNonConst(0)->doImport(xT, *importer, Tpetra::INSERT);

  if (x_dotT) {
    TEUCHOS_TEST_FOR_EXCEPTION(ghost_x->getNumVectors() < 2, std::logic_error,
        "AdaptiveSolutionManager error: x_dotT defined but only a single solution vector is available");
    ghost_x->getVectorNonConst(1)->doImport(*x_dotT, *importer, Tpetra::INSERT);
  }

  if (x_dotdotT) {
    TEUCHOS_TEST_FOR_EXCEPTION(ghost_x->getNumVectors() < 3, std::logic_error,
        "AdaptiveSolutionManager error: x_dotdotT defined but xDotDot isn't defined in the multivector");
    ghost_x->getVectorNonConst(2)->doImport(*x_dotdotT, *importer, Tpetra::INSERT);
  }

}

void SolutionInfo::resize(RCP<Albany::AbstractDiscretization> d, bool have_x_dot) {
  auto t0 = PCU_Time();
  int number_vectors = 1;
  if (have_x_dot) number_vectors = 2;
  auto owned_map = d->getMapT();
  auto ghost_map = d->getOverlapMapT();
  auto owned_graph = d->getJacobianGraphT();
  auto ghost_graph = d->getOverlapJacobianGraphT();
  exporter = rcp(new Tpetra_Export(ghost_map, owned_map));
  importer = rcp(new Tpetra_Import(owned_map, ghost_map));
  owned_x = rcp(new Tpetra_MultiVector(owned_map, number_vectors));
  ghost_x = rcp(new Tpetra_MultiVector(ghost_map, number_vectors));

  owned->x = rcp(new Tpetra_Vector(owned_map));
  owned->x_dot = rcp(new Tpetra_Vector(owned_map));
  owned->f = rcp(new Tpetra_Vector(owned_map));
  owned->J = rcp(new Tpetra_CrsMatrix(owned_graph));

  ghost->x = rcp(new Tpetra_Vector(ghost_map));
  ghost->x_dot = rcp(new Tpetra_Vector(ghost_map));
  ghost->f = rcp(new Tpetra_Vector(ghost_map));
  ghost->J = rcp(new Tpetra_CrsMatrix(ghost_graph));

  auto t1 = PCU_Time();
  if (!PCU_Comm_Self())
    printf("Solution containers resized in %f seconds\n", t1 - t0);
}

} // namespace CTM
