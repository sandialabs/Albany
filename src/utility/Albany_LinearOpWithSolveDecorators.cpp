#include "Albany_LinearOpWithSolveDecorators.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#include "Stratimikos_MueLuHelpers.hpp"
#include "Albany_TpetraTypes.hpp"

namespace Albany
{

  MatrixBased_LOWS::
      MatrixBased_LOWS(
          const Teuchos::RCP<Thyra_LinearOp> &matrix) : mat_(matrix) {}

  MatrixBased_LOWS::
      ~MatrixBased_LOWS() {}

  Teuchos::RCP<const Thyra_VectorSpace>
  MatrixBased_LOWS::
      domain() const
  {
    return mat_->domain();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  MatrixBased_LOWS::
      range() const
  {
    return mat_->range();
  }

  Teuchos::RCP<Thyra_LinearOp>
  MatrixBased_LOWS::
      getMatrix()
  {
    return mat_;
  }

  void
  MatrixBased_LOWS::
      initializeSolver(Teuchos::RCP<Teuchos::ParameterList> solverParamList)
  {
    std::string solverType = solverParamList->get<std::string>("Linear Solver Type");
    Stratimikos::DefaultLinearSolverBuilder strat;
#ifdef ALBANY_MUELU
    Stratimikos::enableMueLu<double, LO, Tpetra_GO, KokkosNode>(strat);
#endif
    strat.setParameterList(solverParamList);
    auto lows_factory = strat.createLinearSolveStrategy(solverType);
    solver_ = lows_factory->createOp();
    solver_transp_ = lows_factory->createOp();
    auto prec_factory =  lows_factory->getPreconditionerFactory();  
    if(Teuchos::nonnull(prec_factory)) {
      auto prec = prec_factory->createPrec();
      prec_factory->initializePrec(Teuchos::rcp(new ::Thyra::DefaultLinearOpSource<double>(mat_)), prec.get());
      Thyra::initializePreconditionedOp<double>(*lows_factory,
          mat_,
          prec,
          solver_.ptr(),
          Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
      Thyra::initializePreconditionedOp<double>(*lows_factory,
          Thyra::transpose<double>(mat_),
          Thyra::unspecifiedPrec<double>(::Thyra::transpose<double>(prec->getUnspecifiedPrecOp())),
          solver_transp_.ptr(),
          Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
    } else {
      Thyra::initializeOp<double>(*lows_factory, mat_, solver_.ptr(), Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
      Thyra::initializeOp<double>(*lows_factory, Thyra::transpose<double>(mat_), solver_transp_.ptr(),Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
    }
  }

  bool
  MatrixBased_LOWS::
      opSupportedImpl(Thyra::EOpTransp M_trans) const
  {
    return mat_->opSupported(M_trans);
  }

  void
  MatrixBased_LOWS::
      applyImpl(const Thyra::EOpTransp M_trans,
                const Thyra_MultiVector &X,
                const Teuchos::Ptr<Thyra_MultiVector> &Y,
                const ST alpha,
                const ST beta) const
  {
    mat_->apply(M_trans, X, Y, alpha, beta);
  }

  Thyra::SolveStatus<double>
  MatrixBased_LOWS::
      solveImpl(
          const Thyra::EOpTransp transp,
          const Thyra_MultiVector &B,
          const Teuchos::Ptr<Thyra_MultiVector> &X,
          const Teuchos::Ptr<const Thyra::SolveCriteria<ST>> solveCriteria) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(solver_), std::runtime_error, "Error! MatrixBased_LOWS::solveImpl, Solver not initialized, call initializeSolver first.\n");
    if (transp == Thyra::EOpTransp::NOTRANS)
      return solver_->solve(Thyra::EOpTransp::NOTRANS, B, X, solveCriteria);
    else
      return solver_transp_->solve(Thyra::EOpTransp::NOTRANS, B, X, solveCriteria);
  }

} // namespace Albany
