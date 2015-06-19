//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_LinearSystem.hpp"
#include "GOAL_Discretization.hpp"
#include "PHAL_Workset.hpp"
#include "Albany_APFDiscretization.hpp"

using Teuchos::rcp;
using Teuchos::RCP;

namespace GOAL {

LinearSystem::LinearSystem(RCP<Discretization>& d) :
  disc(d)
{
  init();
  fillSolution();
}

LinearSystem::~LinearSystem()
{
}

void LinearSystem::init()
{
  RCP<Albany::APFDiscretization> d = disc->getAPFDisc();
  RCP<const Tpetra_Map> m = d->getMapT();
  RCP<const Tpetra_Map> om = d->getOverlapMapT();
  RCP<const Tpetra_CrsGraph> g = d->getJacobianGraphT();
  RCP<const Tpetra_CrsGraph> og = d->getOverlapJacobianGraphT();
  x = rcp( new Tpetra_Vector(m) );
  xdot = rcp( new Tpetra_Vector(m) );
  xdotdot = rcp( new Tpetra_Vector(m) );
  rhs = rcp( new Tpetra_Vector(m) );
  jac = rcp( new Tpetra_CrsMatrix(g) );
  overlapX = rcp( new Tpetra_Vector(om) );
  overlapXdot = rcp( new Tpetra_Vector(om) );
  overlapXdotdot = rcp( new Tpetra_Vector(om) );
  overlapRhs = rcp( new Tpetra_Vector(om) );
  overlapJac = rcp( new Tpetra_CrsMatrix(og) );
  importer = rcp(new Tpetra_Import(m,om));
  exporter = rcp(new Tpetra_Export(om,m));
}

void LinearSystem::fillSolution()
{
  disc->fillSolution(x);
  overlapX->doImport(*x,*importer,Tpetra::INSERT);
}

void LinearSystem::setWorksetSolutionInfo(PHAL::Workset& workset)
{
  workset.xT = overlapX;
  workset.xdotT = overlapXdot;
  workset.xdotdotT = overlapXdotdot;
  workset.JacT = overlapJac;
  workset.x_importerT = importer;
  workset.comm = x->getMap()->getComm();
}

void LinearSystem::completeJacobianFill()
{
  jac->doExport(*(overlapJac),*(exporter),Tpetra::ADD);
  jac->fillComplete();
}

static void writeJacobian(const char* n, RCP<Tpetra_CrsMatrix>& j)
{
  Tpetra_MatrixMarket_Writer::writeSparseFile(n,j);
}

void LinearSystem::writeLinearSystem(int ctr)
{
  char n[25];
  sprintf(n,"g_jac_%i.mm",ctr);
  writeJacobian(n,jac);
}

}
