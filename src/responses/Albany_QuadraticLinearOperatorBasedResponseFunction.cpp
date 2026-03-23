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
#include "Albany_ThyraUtils.hpp"
#include "Tpetra_Core.hpp"
#include "MatrixMarket_Tpetra.hpp"


Albany::QuadraticLinearOperatorBasedResponseFunction::
QuadraticLinearOperatorBasedResponseFunction(const Teuchos::RCP<const Albany::Application> &app,
    const Teuchos::ParameterList &responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app)
{
  auto coeff = responseParams.get<double>("Scaling Coefficient");
  field_name_ = responseParams.get<std::string>("Field Name");
  bool isMisfit = responseParams.isParameter("Target Field Name");
  target_name_ = isMisfit ? responseParams.get<std::string>("Target Field Name") : "";
  auto file_name_A = responseParams.get<std::string>("Matrix A File Name");
  auto file_name_D = responseParams.get<std::string>("Matrix D File Name");
  bool symmetricA = responseParams.isParameter("Matrix A Is Symmetric") ?  responseParams.get<bool>("Matrix A Is Symmetric") : false;
  bool diagonalD = responseParams.isParameter("Matrix D Is Diagonal") ? responseParams.get<bool>("Matrix D Is Diagonal") : false;
  bool verbose = responseParams.isParameter("Verbose") ? responseParams.get<bool>("Verbose") : true;
  Teuchos::RCP<Teuchos::ParameterList> solverParamList = responseParams.isSublist("D Solver Settings") ? 
      Teuchos::rcp(new Teuchos::ParameterList(responseParams.sublist("D Solver Settings"))) : Teuchos::null;
  twoAtDinvA_ = Teuchos::rcp(new AtDinvA_LOWS(file_name_A,file_name_D,2.0*coeff,solverParamList,symmetricA,diagonalD,verbose));
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
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const Teuchos::RCP<Thyra_Vector>& g)
{  
  Teuchos::RCP<const Thyra_Vector> field;
  if(field_name_=="solution") 
    field = x;
  else 
    field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
  twoAtDinvA_->setupFwdOp(field->space());

  // coeff p' A' inv(D) A p
  if(target_name_ == "")
    g->assign(0.5*twoAtDinvA_->quadraticForm(*field));
  else {
    Teuchos::RCP<Thyra_Vector> diff_field = field->clone_v();
    diff_field->update(-1.0, *app_->getDistributedParameterLibrary()->get(target_name_)->vector());
    g->assign(0.5*twoAtDinvA_->quadraticForm(*diff_field));
  }

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());
  g_->assign(*g);
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateTangent(const double /* alpha */,
		const double /*beta*/,
		const double /*omega*/,
		const double /*current_time*/,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
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
      Teuchos::RCP<const Thyra_Vector> field;
      if(field_name_=="solution") 
        field = x;
      else 
        field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
        
      twoAtDinvA_->setupFwdOp(field->space());
      
      // coeff p' A' inv(D) A p
      if(target_name_ == "")
        g->assign(0.5*twoAtDinvA_->quadraticForm(*field));
      else {
        Teuchos::RCP<Thyra_Vector> diff_field = field->clone_v();
        diff_field->update(-1.0, *app_->getDistributedParameterLibrary()->get(target_name_)->vector());
        g->assign(0.5*twoAtDinvA_->quadraticForm(*diff_field));
      }

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
    const Teuchos::RCP<const Thyra_Vector>& x,
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
      Teuchos::RCP<const Thyra_Vector> field;
      if(field_name_=="solution") 
        field = x;
      else 
        field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
      twoAtDinvA_->setupFwdOp(field->space());
      
      //  coeff p' A' inv(D) A p
      if(target_name_ == "")
        g->assign(0.5*twoAtDinvA_->quadraticForm(*field));
      else {
        Teuchos::RCP<Thyra_Vector> diff_field = field->clone_v();
        diff_field->update(-1.0, *app_->getDistributedParameterLibrary()->get(target_name_)->vector());
        g->assign(0.5*twoAtDinvA_->quadraticForm(*diff_field));
      }
      

      g_ = Thyra::createMember(g->space());
      g_->assign(*g);
    } else
      g->assign(*g_);
  }
  
  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    if(field_name_ == "solution") {
      Teuchos::RCP<const Thyra_Vector> field = x;
      twoAtDinvA_->setupFwdOp(field->space());

      //  2 coeff A' inv(D) A p
      if(target_name_ == "")
        twoAtDinvA_->apply(Thyra::EOpTransp::NOTRANS, *field, dg_dx.ptr(), 1.0, 0.0);
      else {
        Teuchos::RCP<Thyra_Vector> diff_field = field->clone_v();
        diff_field->update(-1.0, *app_->getDistributedParameterLibrary()->get(target_name_)->vector());
        twoAtDinvA_->apply(Thyra::EOpTransp::NOTRANS, *diff_field, dg_dx.ptr(), 1.0, 0.0);
      }
    }
    else
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
      if(target_name_ == "")
        twoAtDinvA_->apply(Thyra::EOpTransp::NOTRANS, *field, dg_dp.ptr(), 1.0, 0.0);
      else {
        Teuchos::RCP<Thyra_Vector> diff_field = field->clone_v();
        diff_field->update(-1.0, *app_->getDistributedParameterLibrary()->get(target_name_)->vector());
        twoAtDinvA_->apply(Thyra::EOpTransp::NOTRANS, *diff_field, dg_dp.ptr(), 1.0, 0.0);
      }
    } else
      dg_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_xx(
    const double /* current_time */,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /* param_array */,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dxdx)
{
  if (!Hv_dxdx.is_null()) {
    if(field_name_ == "solution") {
      twoAtDinvA_->setupFwdOp(x->space());

      // 2 coeff A' inv(D) A v
      twoAtDinvA_->apply(Thyra::EOpTransp::NOTRANS, *v, Hv_dxdx.ptr(), 1.0, 0.0);
    }
    else
      Hv_dxdx->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_xp(
    const double /* current_time */,
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
    const double /* current_time */,
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
    const double /* current_time */,
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
  TEUCHOS_TEST_FOR_EXCEPTION (field_name_ != param_name, std::runtime_error, "Error! The parameter name \"" << param_name << "\" should be the same as the field name \"" << field_name_ << "\".\n");
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
  const double& coeff,
  const Teuchos::RCP<Teuchos::ParameterList> solverParameterList,
  const bool symmetricA,
  const bool diagonalD,
  const bool verbose) :
  file_name_A_(file_name_A),
  file_name_D_(file_name_D),
  coeff_(coeff),
  symmetricA_(symmetricA),
  diagonalD_(diagonalD),
  verbose_(verbose),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
  {
    AequalsD_ = (file_name_A == file_name_D);
    TEUCHOS_TEST_FOR_EXCEPTION (AequalsD_ && !(symmetricA || diagonalD), std::runtime_error, 
       "Error! AtDinvA_LOWS::AtDinvA_LOWS, when A equals D, A must be symmetric, as D is\n");   

    if(!(diagonalD || AequalsD_)) {
      TEUCHOS_TEST_FOR_EXCEPTION (Teuchos::is_null(solverParameterList), std::runtime_error, 
       "Error! AtDinvA_LOWS::AtDinvA_LOWS, Solver settings for inverting D must be available when D is not diagonal.\n");
      fwdSolverParameterList_=solverParameterList;
    }
  };


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
    if(!(diagonalD_ || AequalsD_)) {
      D_->initializeSolver(fwdSolverParameterList_);
    }
  }
}


//  coeff X' A' inv(D) A  X, or coeff X' A  X in the simplified case A==D
ST 
Albany::AtDinvA_LOWS::
quadraticForm(const Thyra_MultiVector& X) {
    // A X
    A_->apply(Thyra::EOpTransp::NOTRANS, X, vec1_.ptr(), 1.0, 0.0);

    if(AequalsD_) {
      vec2_->assign(X);
      return coeff_*Thyra::dot(*vec2_,*vec1_);
    }

    // inv(D) A X
    vec2_->assign(0.0);
    if(diagonalD_)
      Thyra::ele_wise_divide( 1.0, *vec1_, *vecD_, vec2_.ptr() );
    else {
      TEUCHOS_TEST_FOR_EXCEPTION (!D_->isInitialized(), std::runtime_error, "Error! AtDinvA_LOWS::quadraticForm, D solver not initialized.\n");
      if (verbose_)
        *out << "\nAtDinvA_LOWS::quadraticForm: linear solve for D\n" << std::endl;

      D_->solve(Thyra::EOpTransp::NOTRANS, *vec1_, vec2_.ptr());
    }

    //  coeff X' A' inv(D) A X
    return coeff_*Thyra::dot(*vec1_,*vec2_);
}

void
Albany::AtDinvA_LOWS::
initializeSolver(Teuchos::RCP<Teuchos::ParameterList> solverParamList) {
  A_->initializeSolver(solverParamList);
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

  auto tpetra_A_mat =
      reader_type::readSparseFile (file_name_A_, rowMap, colMap, domainMap, rangeMap);

  auto tpetra_D_mat =
      reader_type::readSparseFile (file_name_D_, rowMap, colMap, domainMap, rangeMap);
  
  if(diagonalD_) {
    Teuchos::RCP<Tpetra_Vector> tpetra_diag_vec = Teuchos::rcp(new Tpetra_Vector(rowMap));
    tpetra_D_mat->getLocalDiagCopy (*tpetra_diag_vec);
    vecD_ = Albany::createThyraVector(tpetra_diag_vec);
  } else {
    D_ = Teuchos::rcp(new MatrixBased_LOWS(Albany::createThyraLinearOp(tpetra_D_mat), true));
  }

  A_ = Teuchos::rcp(new MatrixBased_LOWS(Albany::createThyraLinearOp(tpetra_A_mat), symmetricA_));  
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
  if(AequalsD_) { 
    // Y = alpha coeff A X  + beta Y in the simplified case (A==D)
    A_->apply(Thyra::EOpTransp::NOTRANS, X, Y, alpha*coeff_, beta);
  } else { // Y = alpha coeff A' inv(D) A X + beta Y

    //A X
    Thyra::SolveStatus<double> solveStatus;
    A_->apply(Thyra::EOpTransp::NOTRANS, X, vec1_.ptr(), coeff_, 0.0);

    // coeff inv(D) A X
    vec2_->assign(0.0);
    if(diagonalD_)
      Thyra::ele_wise_divide( 1.0, *vec1_, *vecD_, vec2_.ptr() );
    else {
      TEUCHOS_TEST_FOR_EXCEPTION (!D_->isInitialized(), std::runtime_error, "Error! AtDinvA_LOWS::applyImpl, D Solver not initialized, call initializeFwdSolver first.\n");

      if (verbose_)
        *out << "\nAtDinvA_LOWS::applyImpl: linear solve for D\n" << std::endl;
      
      D_->solve(Thyra::EOpTransp::NOTRANS, *vec1_, vec2_.ptr());
    }

    // Y = alpha coeff A' inv(D) A X + beta Y
    A_->apply(Thyra::EOpTransp::TRANS, *vec2_, Y, alpha, beta);
  }
}

// returns X = coeff^{-1} A^{-1} D A^{-T} B, or X = coeff^{-1} A^{-1} B, when A==D
Thyra::SolveStatus<double>
Albany::AtDinvA_LOWS::
solveImpl(
  const Thyra::EOpTransp /* transp */,
  const Thyra_MultiVector &B,
  const Teuchos::Ptr<Thyra_MultiVector> &X,
  const Teuchos::Ptr<const Thyra::SolveCriteria<ST> > solveCriteria
  ) const {
  Thyra::SolveStatus<double> solveStatus;


  TEUCHOS_TEST_FOR_EXCEPTION (!A_->isInitialized(), std::runtime_error, "Error! AtDinvA_LOWS::solveImpl, Solvers not initialized, call initializeSolver first.\n");

  Thyra::SolveStatus<double> solveStatus1, solveStatus2;

  if(AequalsD_) { //X = coeff^{-1} A^{-1} 
    if (verbose_)
      *out << "\nAtDinvA_LOWS::solveImpl: linear solve for A\n" << std::endl;

    solveStatus1 = A_->solve(Thyra::EOpTransp::NOTRANS, B, X, solveCriteria);
    X->scale(1.0/coeff_);
    return solveStatus1;
  }  
  if (verbose_)
    *out << "\nAtDinvA_LOWS::solveImpl: first linear solve for A'\n" << std::endl;

  vec1_->assign(0.0);
  // v1 = A^{-T} B
  solveStatus1 = A_->solve(Thyra::EOpTransp::TRANS, B, vec1_.ptr(), solveCriteria);
  
  // v2 = coeff^{-1} D A^{-T} B
  vec2_->assign(0.0);
  if(diagonalD_) {
    Thyra::ele_wise_prod( 1.0/coeff_, *vec1_, *vecD_, vec2_.ptr() );
  } else {
    D_->apply(Thyra::EOpTransp::NOTRANS, *vec1_, vec2_.ptr(), 1.0/coeff_, 0.0);
  }
  if (verbose_)
    *out << "\n\nAtDinvA_LOWS::solveImpl: second linear solve for A\n" << std::endl;

  // X = coeff^{-1} A^{-1} D A^{-T} B
  solveStatus2 = A_->solve(Thyra::EOpTransp::NOTRANS, *vec2_, X, solveCriteria);

  if((solveStatus1.solveStatus == Thyra::SOLVE_STATUS_CONVERGED) && (solveStatus2.solveStatus == Thyra::SOLVE_STATUS_CONVERGED))
    solveStatus.solveStatus =  Thyra::SOLVE_STATUS_CONVERGED;
  else if ((solveStatus1.solveStatus == Thyra::SOLVE_STATUS_UNCONVERGED) || (solveStatus2.solveStatus == Thyra::SOLVE_STATUS_UNCONVERGED))
    solveStatus.solveStatus =  Thyra::SOLVE_STATUS_UNCONVERGED;

  return solveStatus;
}
