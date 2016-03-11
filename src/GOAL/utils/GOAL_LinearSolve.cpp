//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_LinearSolve.hpp"

#include <Thyra_VectorBase.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#ifdef ALBANY_IFPACK2
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#endif
#include <Teuchos_AbstractFactoryStd.hpp>
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>

namespace GOAL {

void solveLinearSystem(
    Teuchos::RCP<Albany::Application> app,
    Teuchos::RCP<Tpetra_CrsMatrix> Amat,
    Teuchos::RCP<Tpetra_Vector> xvec,
    Teuchos::RCP<Tpetra_Vector> bvec)
{

  // linear solver builder object
  Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;

  // set up ifpack2 preconditioner if it's available
#ifdef ALBANY_IFPACK2
  {
    typedef Thyra::PreconditionerFactoryBase<ST> Base;
    typedef Thyra::Ifpack2PreconditionerFactory<Tpetra_CrsMatrix> Impl;
    linearSolverBuilder.setPreconditioningStrategyFactory(
        Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
  }
#endif

  // create a parameterlist with the solver options
  Teuchos::RCP<Teuchos::ParameterList> sl =
    Teuchos::rcp(new Teuchos::ParameterList);
  sl->set<std::string>("Linear Solver Type", "Belos");
  Teuchos::ParameterList& solverTypes = sl->sublist("Linear Solver Types");
  Teuchos::ParameterList& belosTypes = solverTypes.sublist("Belos");
  Teuchos::ParameterList& verboseObject = belosTypes.sublist("VerboseObject");
  verboseObject.set<std::string>("Verbosity Level", "medium");
  belosTypes.set<std::string>("Solver Type", "Block GMRES");
  Teuchos::ParameterList& solver =
    belosTypes.sublist("Solver Types").sublist("Block GMRES");
  solver.set<double>("Convergence Tolerance", 1.0e-8);
  solver.set<int>("Output Frequency", 25);
  solver.set<int>("Output Style", 1);
  solver.set<int>("Verbosity", 33);
  solver.set<int>("Maximum Iterations", 1200);
  solver.set<int>("Block Size", 1);
  solver.set<int>("Num Blocks", 1200);
  solver.set<std::string>("Orthogonalization", "DGKS");
#ifdef ALBANY_IFPACK2
  sl->set<std::string>("Preconditioner Type", "Ifpack2");
  Teuchos::ParameterList& precTypes = sl->sublist("Preconditioner Types");
  Teuchos::ParameterList& ifpackTypes = precTypes.sublist("Ifpack2");
  ifpackTypes.set<int>("Overlap", 0);
  ifpackTypes.set<std::string>("Prec Type", "RILUK");
  Teuchos::ParameterList& ifpackSettings =
    ifpackTypes.sublist("Ifpack2 Settings");
  ifpackSettings.set<int>("fact: iluk level-of-fill", 0);
#else
  sl->set<std::string>("Preconditioner Type", "None");
#endif

  linearSolverBuilder.setParameterList(sl);

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory =
    createLinearSolveStrategy(linearSolverBuilder);
  lowsFactory->setVerbLevel(Teuchos::VERB_LOW);

  
  const Teuchos::RCP<Tpetra_Operator> Aop = Amat;
  const Teuchos::RCP<Thyra::LinearOpBase<ST> > A =
    Thyra::createLinearOp(Aop);
  Teuchos::RCP<Thyra::LinearOpWithSolveBase<ST> >
    nsA = lowsFactory->createOp();
  Thyra::initializeOp<ST>(*lowsFactory, A, nsA.ptr());
  Teuchos::RCP<Thyra::MultiVectorBase<ST> >
    x = Thyra::createVector(xvec);
  Teuchos::RCP<Thyra::MultiVectorBase<ST> >
    b = Thyra::createVector(bvec);

  Thyra::SolveStatus<ST> solveStatus =
    Thyra::solve(*nsA, Thyra::NOTRANS, *b, x.ptr());

}

}
