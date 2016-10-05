#include "CTM_SolutionInfo.hpp"
#include <Albany_AbstractDiscretization.hpp>
#include <PCU.h>

namespace CTM {

SolutionInfo::SolutionInfo() {
}

void SolutionInfo::gather_x() {
  owned_x->doExport(*ghost_x, *exporter, Tpetra::INSERT);
}

void SolutionInfo::scatter_x() {
  ghost_x->doImport(*owned_x, *importer, Tpetra::INSERT);
}

void SolutionInfo::gather_f() {
  owned_f->doExport(*ghost_f, *exporter, Tpetra::ADD);
}

void SolutionInfo::scatter_f() {
  ghost_f->doImport(*owned_f, *importer, Tpetra::INSERT);
}

void SolutionInfo::gather_J() {
  owned_J->doExport(*ghost_J, *exporter, Tpetra::ADD);
}

void SolutionInfo::scatter_J() {
  ghost_J->doImport(*owned_J, *importer, Tpetra::INSERT);
}

void SolutionInfo::resize(RCP<Albany::AbstractDiscretization> d, bool dt) {
  auto t0 = PCU_Time();
  int nv = 1;
  if (dt) nv = 2;
  auto map = d->getMapT();
  auto ghost_map = d->getOverlapMapT();
  auto graph = d->getJacobianGraphT();
  auto ghost_graph = d->getOverlapJacobianGraphT();
  exporter = rcp(new Tpetra_Export(ghost_map, map));
  importer = rcp(new Tpetra_Import(map, ghost_map));
  owned_x = rcp(new Tpetra_MultiVector(map, nv));
  ghost_x = rcp(new Tpetra_MultiVector(ghost_map, nv));
  owned_f = rcp(new Tpetra_Vector(map));
  ghost_f = rcp(new Tpetra_Vector(ghost_map));
  owned_J = rcp(new Tpetra_CrsMatrix(graph));
  ghost_J = rcp(new Tpetra_CrsMatrix(ghost_graph));
  auto t1 = PCU_Time();
  if (!PCU_Comm_Self())
    printf("Solution containers resized in %f seconds", t1-t0);
}

}
