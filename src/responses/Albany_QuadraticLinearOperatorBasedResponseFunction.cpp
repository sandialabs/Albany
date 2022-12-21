//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_QuadraticLinearOperatorBasedResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Tpetra_Core.hpp"
#include "MatrixMarket_Tpetra.hpp"


Albany::QuadraticLinearOperatorBasedResponseFunction::
QuadraticLinearOperatorBasedResponseFunction(const Teuchos::RCP<const Albany::Application> &app,
    Teuchos::ParameterList &responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app)
{
  auto coeff = responseParams.get<double>("Scaling Coefficient");
  field_name_ = responseParams.get<std::string>("Field Name");
  auto file_name_A = responseParams.get<std::string>("Linear Operator File Name");
  auto file_name_D = responseParams.get<std::string>("Diagonal Scaling File Name");
  twoAtDinvA_ = Teuchos::rcp(new AtDinvA_LOWS(file_name_A,file_name_D,2.0*coeff));
}

Albany::QuadraticLinearOperatorBasedResponseFunction::
~QuadraticLinearOperatorBasedResponseFunction()
{
}

unsigned int
Albany::QuadraticLinearOperatorBasedResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const Teuchos::RCP<Thyra_Vector>& g)
{  
  Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
  twoAtDinvA_->setupFwdOp(field->space());

  //  coeff p' A' inv(D) A p
  g->assign(0.5*twoAtDinvA_->quadraticForm(*field));

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());
  g_->assign(*g);
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateTangent(const double alpha, 
		const double /*beta*/,
		const double /*omega*/,
		const double /*current_time*/,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		Teuchos::Array<ParamVec>& /*p*/,
    const int  /*parameter_index*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vx*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  if (!g.is_null()) {
    if (g_.is_null()) {
      Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
      twoAtDinvA_->setupFwdOp(field->space());
      
      //  coeff p' A' inv(D) A p
      g->assign(0.5*twoAtDinvA_->quadraticForm(*field));

      g_ = Thyra::createMember(g->space());
      g_->assign(*g);
    } else
      g->assign(*g_);
  }

  if (!gx.is_null()) {
    gx->assign(0);
  }

  if (!gp.is_null()) {
    gp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const int  /*parameter_index*/,
		const Teuchos::RCP<Thyra_Vector>& g,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!g.is_null()) {
    if (g_.is_null()) {
      Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
      twoAtDinvA_->setupFwdOp(field->space());
      
      //  coeff p' A' inv(D) A p
      g->assign(0.5*twoAtDinvA_->quadraticForm(*field));

      g_ = Thyra::createMember(g->space());
      g_->assign(*g);
    } else
      g->assign(*g_);
  }
  
  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    // V_StV stands for V_out = Scalar * V_in
    dg_dx->assign(0.0);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!dg_dp.is_null()) {
    if(dist_param_name == field_name_) {
      Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
      twoAtDinvA_->setupFwdOp(field->space());

      //  2 coeff A' inv(D) A p
      twoAtDinvA_->apply(Thyra::EOpTransp::NOTRANS, *field, dg_dp.ptr(), 1.0, 0.0);
    } else
      dg_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_xx(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& param_array,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dxdx)
{
  if (!Hv_dxdx.is_null()) {
    Hv_dxdx->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_xp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& /*v*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_direction_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_px(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& /*v*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_pp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& dist_param_name,
    const std::string& dist_param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    if((dist_param_name == field_name_) && (dist_param_direction_name == field_name_)) {
      twoAtDinvA_->setupFwdOp(app_->getDistributedParameterLibrary()->get(field_name_)->vector_space());

      // 2 coeff A' inv(D) A v
      twoAtDinvA_->apply(Thyra::EOpTransp::NOTRANS, *v, Hv_dp.ptr(), 1.0, 0.0);
    }
    else
      Hv_dp->assign(0.0);
  }
}

Teuchos::RCP<Thyra_LinearOp>
Albany::QuadraticLinearOperatorBasedResponseFunction::
get_Hess_pp_operator(const std::string& param_name)
{
  TEUCHOS_TEST_FOR_EXCEPTION (field_name_ != param_name, std::runtime_error, "Error! The parameter name should be the same as the field name.\n");
  twoAtDinvA_->setupFwdOp(app_->getDistributedParameterLibrary()->get(field_name_)->vector_space());
  return twoAtDinvA_;
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
printResponse(Teuchos::RCP<Teuchos::FancyOStream> out)
{
  if (g_.is_null()) {
    *out << " the response has not been evaluated yet!";
    return;
  }

  std::size_t precision = 8;
  std::size_t value_width = precision + 4;
  int gsize = g_->space()->dim();

  for (int j = 0; j < gsize; j++) {
    *out << std::setw(value_width) << Thyra::get_ele(*g_,j);
    if (j < gsize-1)
      *out << ", ";
  }
}

//***************** Implementation of AtDinvA_LOWS **********************


// Constructor
Albany::AtDinvA_LOWS::
AtDinvA_LOWS(
  const std::string& file_name_A,
  const std::string& file_name_D,
  const double& coeff) :
  file_name_A_(file_name_A),
  file_name_D_(file_name_D),
  coeff_(coeff) {};


//! Destructor
Albany::AtDinvA_LOWS::
~AtDinvA_LOWS() {}


Teuchos::RCP<const Thyra_VectorSpace>
Albany::AtDinvA_LOWS::
domain() const {
  return vec_space_;
}


Teuchos::RCP<const Thyra_VectorSpace>
Albany::AtDinvA_LOWS::
range() const {
  return vec_space_;
}


void
Albany::AtDinvA_LOWS::
setupFwdOp(const Teuchos::RCP<const Thyra_VectorSpace>& vec_space)
{
  if(A_.is_null()) {
    vec_space_ = vec_space;
    AtDinvA_LOWS::loadLinearOperators();
  }
}


//  coeff X' A' inv(D) A  X
ST 
Albany::AtDinvA_LOWS::
quadraticForm(const Thyra_MultiVector& X) {
    // A X
    A_->apply(Thyra::EOpTransp::NOTRANS, X, vec1_.ptr(), 1.0, 0.0);

    // coeff inv(D) A p
    vec2_->assign(0.0);
    Thyra::ele_wise_divide( coeff_, *vec1_, *D_, vec2_.ptr() );

    //  coeff p' A' inv(D) A p
    return Thyra::dot(*vec1_,*vec2_);
}


void
Albany::AtDinvA_LOWS::
initializeSolver(Teuchos::RCP<Teuchos::ParameterList> solverParamList) {
  if(A_.is_null())
    AtDinvA_LOWS::loadLinearOperators();

  std::string solverType = solverParamList->get<std::string>("Linear Solver Type");

  Stratimikos::DefaultLinearSolverBuilder strat;

  #ifdef ALBANY_MUELU
    Stratimikos::enableMueLu<double, LO, Tpetra_GO, KokkosNode>(strat);
  #endif

  #ifdef ALBANY_IFPACK2
    strat.setPreconditioningStrategyFactory(
      Teuchos::abstractFactoryStd<Thyra::PreconditionerFactoryBase<ST>,
      Thyra::Ifpack2PreconditionerFactory<Tpetra_CrsMatrix>>(),
      "Ifpack2", true
      );
  #endif

  strat.setParameterList(solverParamList);
  auto lows_factory = strat.createLinearSolveStrategy(solverType);
  A_solver_ = lows_factory->createOp();
  A_transSolver_ = lows_factory->createOp();

  auto prec_factory =  lows_factory->getPreconditionerFactory();  
  if(Teuchos::nonnull(prec_factory)) {
    auto prec = prec_factory->createPrec();
    prec_factory->initializePrec(Teuchos::rcp(new ::Thyra::DefaultLinearOpSource<double>(A_)), prec.get());
    Thyra::initializePreconditionedOp<double>(*lows_factory,
          A_,
          prec,
          A_solver_.ptr(),
          Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
    
    Thyra::initializePreconditionedOp<double>(*lows_factory,
          Thyra::transpose<double>(A_),
          Thyra::unspecifiedPrec<double>(::Thyra::transpose<double>(prec->getUnspecifiedPrecOp())),
          A_transSolver_.ptr(),
          Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
  } else {
    Thyra::initializeOp<double>(*lows_factory, A_, A_solver_.ptr(),Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
    Thyra::initializeOp<double>(*lows_factory, Thyra::transpose<double>(A_), A_transSolver_.ptr(),Thyra::SUPPORT_SOLVE_FORWARD_ONLY);
  }
}


void
Albany::AtDinvA_LOWS::
loadLinearOperators() {
  Teuchos::RCP<const Tpetra_Map> rowMap = Albany::getTpetraMap(vec_space_);
  Teuchos::RCP<const Tpetra_Map> colMap;
  Teuchos::RCP<const Tpetra_Map> domainMap = rowMap;
  Teuchos::RCP<const Tpetra_Map> rangeMap = rowMap;
  typedef Tpetra::MatrixMarket::Reader<Tpetra_CrsMatrix> reader_type;

  bool mapIsContiguous =
      (static_cast<Tpetra_GO>(rowMap->getMaxAllGlobalIndex()+1-rowMap->getMinAllGlobalIndex()) ==
        static_cast<Tpetra_GO>(rowMap->getGlobalNumElements()));

  TEUCHOS_TEST_FOR_EXCEPTION (!mapIsContiguous, std::runtime_error,
                              "Error! Row Map needs to be contiguous for the Matrix reader to work.\n");

  auto tpetra_mat =
      reader_type::readSparseFile (file_name_A_, rowMap, colMap, domainMap, rangeMap);

  auto tpetra_diag_mat =
      reader_type::readSparseFile (file_name_D_, rowMap, colMap, domainMap, rangeMap);
  Teuchos::RCP<Tpetra_Vector> tpetra_diag_vec = Teuchos::rcp(new Tpetra_Vector(rowMap));
  tpetra_diag_mat->getLocalDiagCopy (*tpetra_diag_vec);

  A_ = Albany::createThyraLinearOp(tpetra_mat);
  D_ = Albany::createThyraVector(tpetra_diag_vec);
  vec1_ = Thyra::createMember(A_->range());
  vec2_ = Thyra::createMember(A_->range());
}


bool
Albany::AtDinvA_LOWS::
opSupportedImpl(Thyra::EOpTransp /*M_trans*/) const {
  return true;
}


void
Albany::AtDinvA_LOWS::
applyImpl (const Thyra::EOpTransp /*M_trans*/, //operator is symmetric by construction
                const Thyra_MultiVector& X,
                const Teuchos::Ptr<Thyra_MultiVector>& Y,
                const ST alpha,
                const ST beta) const {
  //A X
  A_->apply(Thyra::EOpTransp::NOTRANS, X, vec1_.ptr(), 1.0, 0.0);

  // coeff inv(D) A X
  vec2_->assign(0.0);
  Thyra::ele_wise_divide( coeff_, *vec1_, *D_, vec2_.ptr() );

  // Y = alpha coeff A' inv(D) A X + beta Y
  A_->apply(Thyra::EOpTransp::TRANS, *vec2_, Y, alpha, beta);
}

// returns X = coeff^{-1} A^{-1} D A^{-T} B 
Thyra::SolveStatus<double>
Albany::AtDinvA_LOWS::
solveImpl(
  const Thyra::EOpTransp transp,
  const Thyra_MultiVector &B,
  const Teuchos::Ptr<Thyra_MultiVector> &X,
  const Teuchos::Ptr<const Thyra::SolveCriteria<ST> > solveCriteria
  ) const {
  Thyra::SolveStatus<double> solveStatus;
  
  TEUCHOS_TEST_FOR_EXCEPTION (Teuchos::is_null(A_solver_) || Teuchos::is_null(A_transSolver_), std::runtime_error, "Error! AtDinvA_LOWS::solveImpl, Solvers not initialized, call initializeSolver first.\n");

  Thyra::SolveStatus<double> solveStatus1, solveStatus2;
  solveStatus1 = A_transSolver_->solve(Thyra::EOpTransp::NOTRANS, B, vec1_.ptr(), solveCriteria);
  vec2_->assign(0.0);
  Thyra::ele_wise_prod( 1.0/coeff_, *vec1_, *D_, vec2_.ptr() );
  solveStatus2 = A_solver_->solve(Thyra::EOpTransp::NOTRANS, *vec2_, X, solveCriteria);

  if((solveStatus1.solveStatus == Thyra::SOLVE_STATUS_CONVERGED) && (solveStatus2.solveStatus == Thyra::SOLVE_STATUS_CONVERGED))
    solveStatus.solveStatus =  Thyra::SOLVE_STATUS_CONVERGED;
  else if ((solveStatus1.solveStatus == Thyra::SOLVE_STATUS_UNCONVERGED) || (solveStatus2.solveStatus == Thyra::SOLVE_STATUS_UNCONVERGED))
    solveStatus.solveStatus =  Thyra::SOLVE_STATUS_UNCONVERGED;

  return solveStatus;
}
