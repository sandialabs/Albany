#include "CTM_SolutionInfo.hpp"

#include <PCU.h>
#include <Albany_AbstractDiscretization.hpp>

namespace CTM {

using Teuchos::rcp;

SolutionInfo::SolutionInfo() {
  owned = rcp(new LinearObj);
  ghost = rcp(new LinearObj);
}

void SolutionInfo::gather_x() {
  owned->x->doExport(*(ghost->x), *exporter, Tpetra::INSERT);
}

void SolutionInfo::gather_x_dot() {
  if (owned->x_dot != Teuchos::null)
    owned->x_dot->doExport(*(ghost->x_dot), *exporter, Tpetra::INSERT);
}

void SolutionInfo::gather_f() {
  owned->f->doExport(*(ghost->f), *exporter, Tpetra::ADD);
}

void SolutionInfo::gather_J() {
  owned->J->doExport(*(ghost->J), *exporter, Tpetra::ADD);
}

void SolutionInfo::scatter_x() {
  ghost->x->doImport(*(owned->x), *importer, Tpetra::INSERT);
}

void SolutionInfo::scatter_x_dot() {
  if (ghost->x_dot != Teuchos::null)
    ghost->x_dot->doImport(*(owned->x_dot), *importer, Tpetra::INSERT);
}

void SolutionInfo::scatter_f() {
  ghost->f->doExport(*(owned->f), *importer, Tpetra::INSERT);
}

void SolutionInfo::scatter_J() {
  ghost->J->doImport(*(owned->J), *importer, Tpetra::INSERT);
}

void SolutionInfo::resize(RCP<Albany::AbstractDiscretization> d, bool have_x_dot) {
  auto t0 = PCU_Time();
  
  auto owned_map = d->getMapT();
  auto ghost_map = d->getOverlapMapT();
  auto owned_graph = d->getJacobianGraphT();
  auto ghost_graph = d->getOverlapJacobianGraphT();
  
  exporter = rcp(new Tpetra_Export(ghost_map, owned_map));
  importer = rcp(new Tpetra_Import(owned_map, ghost_map));

  owned->x = rcp(new Tpetra_Vector(owned_map));
  owned->f = rcp(new Tpetra_Vector(owned_map));
  owned->J = rcp(new Tpetra_CrsMatrix(owned_graph));
  if (have_x_dot)
    owned->x_dot = rcp(new Tpetra_Vector(owned_map));

  ghost->x = rcp(new Tpetra_Vector(ghost_map));
  ghost->f = rcp(new Tpetra_Vector(ghost_map));
  ghost->J = rcp(new Tpetra_CrsMatrix(ghost_graph));
  if (have_x_dot)
    ghost->x_dot = rcp(new Tpetra_Vector(ghost_map));

  auto t1 = PCU_Time();
  if (!PCU_Comm_Self())
    printf("Solution containers resized in %f seconds\n", t1 - t0);
}

} // namespace CTM
