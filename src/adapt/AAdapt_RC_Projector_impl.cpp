//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DataTypes.hpp"

#include <BelosBlockCGSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_RILUK.hpp>

namespace AAdapt {
namespace rc {

Teuchos::RCP<Tpetra_MultiVector>
solve (const Teuchos::RCP<const Tpetra_CrsMatrix>& A,
       Teuchos::RCP<Tpetra_Operator>& P,
       const Teuchos::RCP<const Tpetra_MultiVector>& b,
       Teuchos::ParameterList& pl) {
  typedef Tpetra_MultiVector MV;
  typedef Tpetra_Operator Op;
  typedef Belos::SolverManager<RealType, MV, Op> SolverManager;
  typedef Belos::LinearProblem<RealType, MV, Op> LinearProblem;

  const int nrhs = b->getNumVectors();
  Teuchos::RCP<Tpetra_MultiVector> x = Teuchos::rcp(
    new Tpetra_MultiVector(A->getDomainMap(), nrhs, true));
    
  if (P.is_null()) {
    Teuchos::ParameterList pl;
    pl.set<int>("fact: iluk level-of-fill", 0);
    Teuchos::RCP< Ifpack2::RILUK<Tpetra_RowMatrix> >
      prec = Teuchos::rcp(new Ifpack2::RILUK<Tpetra_RowMatrix>(A));
    prec->setParameters(pl);
    prec->initialize();
    prec->compute();
    P = prec;
  };

  Teuchos::RCP<LinearProblem>
    problem = Teuchos::rcp(new LinearProblem(A, x, b));
  problem->setRightPrec(P);
  problem->setProblem();

  Belos::BlockCGSolMgr<RealType, MV, Op>
    solver(problem, Teuchos::rcp(&pl, false));
  solver.solve();

  return x;
}

} // namespace rc
} // namespace AAdapt
