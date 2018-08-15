//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Aeras_HVDecorator.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_ModelFactory.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include <sstream>

#include "Albany_TpetraThyraUtils.hpp"

//uncomment the following to write stuff out to matrix market to debug
//#define WRITE_TO_MATRIX_MARKET_TO_MM_FILE

#ifdef WRITE_TO_MATRIX_MARKET_TO_MM_FILE
static
int mm_counter = 0;
#include "TpetraExt_MMHelpers.hpp"
#endif // WRITE_TO_MATRIX_MARKET

//#define OUTPUT_TO_SCREEN 

namespace {
// Got hints from Tpetra::CrsMatrix::clone.
Teuchos::RCP<Tpetra_CrsMatrix> alloc (const Teuchos::RCP<Tpetra_CrsMatrix>& A) {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayRCP;

  ArrayRCP<const std::size_t> per_local_row;
  std::size_t all_local_rows = 0;
  bool bound_same = false;
  A->getCrsGraph()->getNumEntriesPerLocalRowUpperBound(per_local_row, all_local_rows, bound_same);

  RCP<Tpetra_CrsMatrix> B;
  if (bound_same)
    B = rcp(new Tpetra_CrsMatrix(A->getRowMap(), A->getColMap(), all_local_rows,
                                 Tpetra::StaticProfile));
  else
    B = rcp(new Tpetra_CrsMatrix(A->getRowMap(), A->getColMap(), per_local_row,
                                 Tpetra::StaticProfile));

  return B;
}

Teuchos::RCP<Thyra_LinearOp> getOnlyNonzeros (const Teuchos::RCP<Thyra_LinearOp>& A_op) {
  using Teuchos::RCP;
  using Teuchos::Array;
  using Teuchos::ArrayView;

  auto A = Albany::getTpetraMatrix(A_op);

  TEUCHOS_ASSERT(A->hasColMap());
  TEUCHOS_ASSERT(A->isLocallyIndexed());

  RCP<Tpetra_CrsMatrix> B = alloc(A);
  const RCP<const Tpetra_Map> row_map = B->getRowMap();

  ArrayView<const LO> Ainds;
  ArrayView<const ST> Avals;
  Array<LO> Binds;
  Array<ST> Bvals;
  for (LO lrow = row_map->getMinLocalIndex(), lmax = row_map->getMaxLocalIndex(); lrow <= lmax; ++lrow) {
    A->getLocalRowView(lrow, Ainds, Avals);
    if (Ainds.size()) {
      Binds.clear();
      Bvals.clear();
      for (std::size_t i = 0, ilim = Ainds.size(); i < ilim; ++i)
        if (Avals[i] != 0) {
          Binds.push_back(Ainds[i]);
          Bvals.push_back(Avals[i]);
        }
      B->insertLocalValues(lrow, Binds, Bvals);
    }
  }
  B->fillComplete();
  
  return Albany::createThyraLinearOp(B);
}
} // namespace

Aeras::HVDecorator::HVDecorator(
    const Teuchos::RCP<Albany::Application>& app_,
    const Teuchos::RCP<Teuchos::ParameterList>& appParams)
    :Albany::ModelEvaluatorT(app_,appParams)
{

#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  std::cout << "In HVDecorator app name: " << app->getProblemPL()->get("Name", "") << std::endl;
#endif

  // Create and store mass and Laplacian operators (in CrsMatrix form). 
  Teuchos::RCP<Tpetra_CrsMatrix> mass = Albany::getTpetraMatrix(createOperatorDiag(1.0, 0.0, 0.0));
  Teuchos::RCP<Thyra_LinearOp> laplace = createOperator(0.0, 0.0, 1.0);

  Teuchos::RCP<const Thyra_VectorSpace> mass_vs = Albany::createThyraVectorSpace(mass->getRowMap());

  // Do some preprocessing to speed up subsequent residual calculations.
  // 1. Store the lumped mass diag reciprocal.
  inv_mass_diag_ = Thyra::createMember(mass_vs);
  mass->getLocalDiagCopy(*Albany::getTpetraVector(inv_mass_diag_));
  Thyra::reciprocal(*inv_mass_diag_,inv_mass_diag_.ptr());
  // 2. Create a work vector in advance.
  wrk_ = Thyra::createMember(mass_vs);
  // 3. Remove the structural zeros, numerical zeros, from the Laplace
  // operator.
  laplace_ = getOnlyNonzeros(laplace);
  xtilde = Thyra::createMember(mass_vs);

//OG In case of a parallel run by some reason laplace.mm file contains indices
//out of range with non-trivial entries. I haven't debugged this yet. AB suggested to
//compare the product L*x (L is the Laplace, x is an arbitrary vector)
//in case of a parallel and serial run.
#ifdef WRITE_TO_MATRIX_MARKET_TO_MM_FILE
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile("mass.mm", mass);
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeSparseFile("laplace.mm", Albany::getTpetraMatrix(laplace_));
#endif
}
 
//IKT: the following function creates either the mass or Laplacian operator, to be 
//stored as a member function and used in evalModelImpl to perform the update for the auxiliary 
//utilde/htilde variables when integrating the hyperviscosity system in time using 
//an explicit scheme. 
Teuchos::RCP<Thyra_LinearOp> 
Aeras::HVDecorator::createOperator(double alpha, double beta, double omega)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  double curr_time = 0.0;
  //Get implicit_graphT from discretization object
  Teuchos::RCP<const Tpetra_CrsGraph> implicit_graphT = 
    app->getDiscretization()->getImplicitJacobianGraphT();  
  //Define operator Op from implicit_graphT
  const Teuchos::RCP<Thyra_LinearOp> Op =
    Teuchos::nonnull(implicit_graphT) ? 
    Albany::createThyraLinearOp(Teuchos::rcp(new Tpetra_CrsMatrix(implicit_graphT))) :
    Teuchos::null; 
  auto args = this->getNominalValues();
  auto f = Thyra::createMember(args.get_x()->space());
  app->computeGlobalJacobian(alpha, beta, omega, curr_time,
                             args.get_x(),
                             (supports_xdot ? args.get_x_dot() : Teuchos::null),
                             (supports_xdotdot ? args.get_x_dot_dot() : Teuchos::null),
                             sacado_param_vec, f, Op);
  return Op; 
}

Teuchos::RCP<Thyra_LinearOp> 
Aeras::HVDecorator::createOperatorDiag(double alpha, double beta, double omega)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  double curr_time = 0.0;
  const Teuchos::RCP<Thyra_LinearOp> op = this->create_W_op();

  auto args = this->getNominalValues();
  auto f = Thyra::createMember(args.get_x()->space());
  app->computeGlobalJacobian(alpha, beta, omega, curr_time,
                             args.get_x(),
                             (supports_xdot ? args.get_x_dot() : Teuchos::null),
                             (supports_xdotdot ? args.get_x_dot_dot() : Teuchos::null),
                             sacado_param_vec, f, op);
  return op; 
}

//IKT: the following function returns laplace_*mass_^(-1)*laplace_*x_in.  It is to be called 
//in evalModelImpl after the last computeGlobalResidual call.
//Note that it is more efficient to implement an apply method like is done here, than 
//to form a sparse CrsMatrix laplace_*mass_^(-1)*laplace_ and store it.  
void
Aeras::HVDecorator::applyLinvML(Teuchos::RCP<const Thyra_Vector> x_in, Teuchos::RCP<Thyra_Vector> x_out)
const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif

  // x_out = laplace_ * x_in
  laplace_->apply(Thyra::NOTRANS, *x_in, wrk_.ptr(), 1.0, 0.0);

  // x_out = inv(M) * x_out
  Thyra::ele_wise_scale(*inv_mass_diag_, wrk_.ptr());

  // x_out = laplace*wrk_ = laplace * inv(M) * laplace * x_in
  laplace_->apply(Thyra::NOTRANS, *wrk_, x_out.ptr(), 1.0, 0.0);

  //Teuchos::ArrayRCP<const ST> inv_mass_diag_constView = inv_mass_diag->get1dView(); 
  /*//create CrsMatrix for Mass^(-1)
  Teuchos::RCP<Tpetra_CrsMatrix> inv_mass = Teuchos::rcp(new Tpetra_CrsMatrix(mass_->getRowMap(), 1)); 
  for (LO row=0; row<mass_->getNodeNumRows(); row++) {
    ST val = inv_mass_diag_constView[row];  
    inv_mass->sumIntoLocalValues(row, Teuchos::arrayView(&row,1), Teuchos::arrayView(&val,1)); 
  }
  inv_mass->fillComplete(); 
  //allocate l_minv_l
  Teuchos::RCP<Tpetra_CrsMatrix> l_minv_l = Teuchos::rcp(new Tpetra_CrsMatrix(laplace_->getRowMap(), laplace_->getGlobalMaxNumRowEntries()));
  l_minv_l->fillComplete();  
  //l_minv_l = mass_inv*laplace_ 
  Tpetra::MatrixMatrix::Multiply(*inv_mass, false, *laplace_, false, *l_minv_l); 
  //l_minv_l = laplace_*mass_inv
  Tpetra::MatrixMatrix::Multiply(*laplace_, false, *l_minv_l, false, *l_minv_l);
  return l_minv_l;  
  */
}


//og: do I have to copy/paste this from AMET.cpp?
namespace {
// As of early Jan 2015, it seems there is some conflict between Thyra's use of
// NaN to initialize certain quantities and Tpetra's v.update(alpha, x, 0)
// implementation. In the past, 0 as the third argument seemed to trigger a code
// path that does a set v <- alpha x rather than an accumulation v <- alpha x +
// beta v. Hence any(isnan(v(:))) was not a problem if beta == 0. That seems not
// to be entirely true now. For some reason, this problem occurs only in DEBUG
// builds in the sensitivities. I have not had time to fully dissect this
// problem to determine why the problem occurs only there, but the solution is
// nonetheless quite suggestive: sanitize v before calling update. I do this at
// the highest level, here, rather than in the responses.
void sanitize_nans (const Thyra_Derivative& v) {
  if ( ! v.isEmpty() && Teuchos::nonnull(v.getMultiVector())) {
    v.getMultiVector()->assign(0.0);
  }
}
} // namespace


// hide the original parental method AMET->evalModelImpl():
void
Aeras::HVDecorator::evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgsT,
    const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgsT) const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG WHICH HVDecorator: " << __PRETTY_FUNCTION__ << "\n";
#endif
	
  Teuchos::TimeMonitor Timer(*timer); //start timer

  //
  // Get the input arguments
  //
  // Thyra vectors
  const Teuchos::RCP<const Thyra_Vector> x = inArgsT.get_x();
  const Teuchos::RCP<const Thyra_Vector> x_dot =
      (supports_xdot ? inArgsT.get_x_dot() : Teuchos::null);
  const Teuchos::RCP<const Thyra_Vector> x_dotdot =
      (supports_xdotdot ? inArgsT.get_x_dot_dot() : Teuchos::null);

  const double alpha = (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ? inArgsT.get_alpha() : 0.0;
  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  // const double omega = (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ? inArgsT.get_omega() : 0.0;
  const double omega = 0.0;
  const double beta = (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ? inArgsT.get_beta() : 1.0;
  const double curr_time = (Teuchos::nonnull(x_dot) || Teuchos::nonnull(x_dotdot)) ? inArgsT.get_t() : 0.0;

  for (int l = 0; l < inArgsT.Np(); ++l) {
    const Teuchos::RCP<const Thyra_Vector> p = inArgsT.get_p(l);
    if (Teuchos::nonnull(p)) {
      const Teuchos::RCP<const Tpetra_Vector> pT = Albany::getConstTpetraVector(p);
      const Teuchos::ArrayRCP<const ST> pT_constView = pT->get1dView();

      ParamVec &sacado_param_vector = sacado_param_vec[l];
      for (unsigned int k = 0; k < sacado_param_vector.size(); ++k) {
        sacado_param_vector[k].baseValue = pT_constView[k];
      }
    }
  }

  //
  // Get the output arguments
  //
  auto f    = outArgsT.get_f();
  auto W_op = outArgsT.get_W_op();

  //
  // Compute the functions
  //
  bool f_already_computed = false;

  // W matrix
  if (Teuchos::nonnull(W_op)) {
    app->computeGlobalJacobian(
        alpha, beta, omega, curr_time, x, x_dot, x_dotdot,
        sacado_param_vec, f, W_op);
    f_already_computed = true;
  }

  // df/dp
  for (int l = 0; l < outArgsT.Np(); ++l) {
    const Teuchos::RCP<Thyra_MultiVector> df_dp = outArgsT.get_DfDp(l).getMultiVector();

    if (Teuchos::nonnull(df_dp)) {
      const Teuchos::RCP<ParamVec> p_vec = Teuchos::rcpFromRef(sacado_param_vec[l]);

      app->computeGlobalTangent(
          0.0, 0.0, 0.0, curr_time, false, x, x_dot, x_dotdot,
          sacado_param_vec, p_vec.get(),
          Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null,
          f, Teuchos::null, df_dp);

      f_already_computed = true;
    }
  }

  // f
  if (app->is_adjoint) {
    const Thyra_Derivative f_deriv(f, Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW);
    const Thyra_Derivative dummy_deriv;

    const int response_index = 0; // need to add capability for sending this in
    app->evaluateResponseDerivative(
        response_index, curr_time, x, x_dot, x_dotdot,
        sacado_param_vec, NULL,
        Teuchos::null, f_deriv, dummy_deriv, dummy_deriv, dummy_deriv);
  } else {
    if (Teuchos::nonnull(f) && !f_already_computed) {
      app->computeGlobalResidual(
          curr_time, x, x_dot, x_dotdot,
          sacado_param_vec, f);
    }
  }

  //compute xtilde 
  applyLinvML(x, xtilde); 

#ifdef WRITE_TO_MATRIX_MARKET_TO_MM_FILE
  //writing to MatrixMarket for debug
  char name[100];  //create string for file name
  sprintf(name, "xT_%i.mm", mm_counter);
  const Teuchos::RCP<const Tpetra_Vector> xT = Albany::getConstTpetraVector(x);
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(name, xT);
  sprintf(name, "xtildeT_%i.mm", mm_counter);
  const Teuchos::RCP<const Tpetra_Vector> xtildeT = Albany::getConstTpetraVector(xtilde);
  Tpetra::MatrixMarket::Writer<Tpetra_CrsMatrix>::writeDenseFile(name, xtildeT);
  mm_counter++; 
#endif  

  if (supports_xdot && Teuchos::nonnull(inArgsT.get_x_dot()) && Teuchos::nonnull(f)){
#ifdef OUTPUT_TO_SCREEN
    std::cout <<"in the if-statement for the update" <<std::endl;
#endif
    f->update(1.0,*xtilde);
  }

  // Response functions
  for (int j = 0; j < outArgsT.Ng(); ++j) {
    Teuchos::RCP<Thyra_Vector> g = outArgsT.get_g(j);

    const Thyra_Derivative dg_dx = outArgsT.get_DgDx(j);
    const Thyra_Derivative dg_dxdot = outArgsT.get_DgDx_dot(j);
    // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
    const Thyra_Derivative dg_dxdotdot;
    sanitize_nans(dg_dx);
    sanitize_nans(dg_dxdot);
    sanitize_nans(dg_dxdotdot);

    // dg/dx, dg/dxdot
    if (!dg_dx.isEmpty() || !dg_dxdot.isEmpty()) {
      const Thyra_Derivative dummy_deriv;
      app->evaluateResponseDerivative(
          j, curr_time, x, x_dot, x_dotdot,
          sacado_param_vec, NULL,
          g, dg_dx, dg_dxdot, dg_dxdotdot, dummy_deriv);
      // Set g to null to indicate the response was evaluated.
      g= Teuchos::null;
    }

    // dg/dp
    for (int l = 0; l < outArgsT.Np(); ++l) {
      const Teuchos::RCP<Thyra_MultiVector> dg_dp =  outArgsT.get_DgDp(j, l).getMultiVector();

      if (Teuchos::nonnull(dg_dp)) {
        const Teuchos::RCP<ParamVec> p_vec = Teuchos::rcpFromRef(sacado_param_vec[l]);
        app->evaluateResponseTangent(
            j, alpha, beta, omega, curr_time, false,
            x, x_dot, x_dotdot, sacado_param_vec, p_vec.get(),
            Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null,
            g, Teuchos::null, dg_dp);
        g = Teuchos::null;
      }
    }

    // If response was not yet evaluated, do it now.
    if (Teuchos::nonnull(g)) {
      app->evaluateResponse(
          j, curr_time,
          x, x_dot, x_dotdot,
          sacado_param_vec, g);
    }
  }
}
