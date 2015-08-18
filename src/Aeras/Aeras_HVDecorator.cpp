//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Aeras_HVDecorator.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_ModelFactory.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include <sstream>
#include "TpetraExt_MatrixMatrix.hpp"

//uncomment the following to write stuff out to matrix market to debug
#define WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
static
int mm_counter = 0;
#endif // WRITE_TO_MATRIX_MARKET

#define OUTPUT_TO_SCREEN 

Aeras::HVDecorator::HVDecorator(
    const Teuchos::RCP<Albany::Application>& app_,
    const Teuchos::RCP<Teuchos::ParameterList>& appParams)
    :Albany::ModelEvaluatorT(app_,appParams)
{

#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif

  std::cout << "In HVDecorator app name: " << app->getProblemPL()->get("Name", "") << std::endl;

//Create and store mass and Laplacian operators (in CrsMatrix form). 
  mass_ = createOperator(1.0, 0.0, 0.0); 
  laplace_ = createOperator(0.0, 0.0, 1.0); 
#ifdef WRITE_TO_MATRIX_MARKET
  Tpetra_MatrixMarket_Writer::writeSparseFile("mass.mm", mass_);
  Tpetra_MatrixMarket_Writer::writeSparseFile("laplace.mm", laplace_);
#endif

}
 
//IKT: the following function creates either the mass or Laplacian operator, to be 
//stored as a member function and used in evalModelImpl to perform the update for the auxiliary 
//utilde/htilde variables when integrating the hyperviscosity system in time using 
//an explicit scheme. 
Teuchos::RCP<Tpetra_CrsMatrix> 
Aeras::HVDecorator::createOperator(double alpha, double beta, double omega)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  double curr_time = 0.0; 
  const Teuchos::RCP<Tpetra_Operator> Op =
    Teuchos::nonnull(this->create_W_op()) ?
    ConverterT::getTpetraOperator(this->create_W_op()) :
    Teuchos::null;
  const Teuchos::RCP<Tpetra_CrsMatrix> Op_crs =
    Teuchos::nonnull(Op) ?
    Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(Op, true) :
    Teuchos::null;
  const Teuchos::RCP<const Tpetra_Vector> xT = ConverterT::getConstTpetraVector(this->getNominalValues().get_x());
  const Teuchos::RCP<const Tpetra_Vector> x_dotT =
    Teuchos::nonnull(this->getNominalValues().get_x_dot()) ?
    ConverterT::getConstTpetraVector(this->getNominalValues().get_x_dot()) :
    Teuchos::null;
  //IKT: it's important to make x_dotdotT non-null.  Otherwise 2nd derivative terms defining the laplace operator
  //will not get set in PHAL_GatherSolution_Def.hpp. 
  const Teuchos::RCP<const Tpetra_Vector> x_dotdotT = Teuchos::rcp(new Tpetra_Vector(xT->getMap(), true));
  const Teuchos::RCP<Tpetra_Vector> fT = Teuchos::rcp(new Tpetra_Vector(xT->getMap(), true)); 
  app->computeGlobalJacobianT(alpha, beta, omega, curr_time, x_dotT.get(), x_dotdotT.get(), *xT, 
                               sacado_param_vec, fT.get(), *Op_crs);
  return Op_crs; 
}

//IKT: the following function returns laplace_*mass_^(-1)*laplace_*x_in.  It is to be called 
//in evalModelImpl after the last computeGlobalResidualT call.
//Note that it is more efficient to implement an apply method like is done here, than 
//to form a sparse CrsMatrix laplace_*mass_^(-1)*laplace_ and store it.  
void
Aeras::HVDecorator::applyLinvML(Teuchos::RCP<const Tpetra_Vector> x_in, Teuchos::RCP<Tpetra_Vector> x_out)
const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
#endif
  //initialize vector for inverse of mass diagonal
  Teuchos::RCP<Tpetra_Vector> inv_mass_diag = Teuchos::rcp(new Tpetra_Vector(mass_->getRowMap(), true)); 
  //get mass matrix's diagonal and put it in inv_mass_diag
  mass_->getLocalDiagCopy(*inv_mass_diag); 
  //inv_mass_diag->putScalar(1.0);  
  //take reciprocal
  inv_mass_diag->reciprocal(*inv_mass_diag);
  Teuchos::RCP<Tpetra_Vector> x_temp1 = Teuchos::rcp(new Tpetra_Vector(x_in->getMap())); 
  //x_temp1 = laplace_*x_in
  laplace_->apply(*x_in, *x_temp1, Teuchos::NO_TRANS, 1.0, 0.0); 

  Teuchos::RCP<Tpetra_Vector> x_temp2 = Teuchos::rcp(new Tpetra_Vector(x_in->getMap())); 
  //x_temp2 = inv(M)*x_temp1
  x_temp2->elementWiseMultiply(1.0, *inv_mass_diag, *x_temp1, 0.0);

  //x_out = laplace*x_temp2 = laplace* inv(M) * laplace * x_in
  laplace_->apply(*x_temp2, *x_out, Teuchos::NO_TRANS, 1.0, 0.0);
  

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
void sanitize_nans (const Thyra::ModelEvaluatorBase::Derivative<ST>& v) {
  if ( ! v.isEmpty() && Teuchos::nonnull(v.getMultiVector()))
    ConverterT::getTpetraMultiVector(v.getMultiVector())->putScalar(0.0);
}
} // namespace


// hide the original parental method AMET->evalModelImpl():
void
Aeras::HVDecorator::evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgsT,
    const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgsT) const
{

  std::cout << "DEBUG WHICH HVDecorator: " << __PRETTY_FUNCTION__ << "\n";
	
  Teuchos::TimeMonitor Timer(*timer); //start timer
  //
  // Get the input arguments
  //
  const Teuchos::RCP<const Tpetra_Vector> xT =
    ConverterT::getConstTpetraVector(inArgsT.get_x());

  const Teuchos::RCP<const Tpetra_Vector> x_dotT =
    Teuchos::nonnull(inArgsT.get_x_dot()) ?
    ConverterT::getConstTpetraVector(inArgsT.get_x_dot()) :
    Teuchos::null;

  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  //const Teuchos::RCP<const Tpetra_Vector> x_dotdotT =
  //  Teuchos::nonnull(inArgsT.get_x_dotdot()) ?
  //  ConverterT::getConstTpetraVector(inArgsT.get_x_dotdot()) :
  //  Teuchos::null;
  const Teuchos::RCP<const Tpetra_Vector> x_dotdotT = Teuchos::null;


  const double alpha = (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ? inArgsT.get_alpha() : 0.0;
  // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
  // const double omega = (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ? inArgsT.get_omega() : 0.0;
  const double omega = 0.0;
  const double beta = (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ? inArgsT.get_beta() : 1.0;
  const double curr_time = (Teuchos::nonnull(x_dotT) || Teuchos::nonnull(x_dotdotT)) ? inArgsT.get_t() : 0.0;

  for (int l = 0; l < inArgsT.Np(); ++l) {
    const Teuchos::RCP<const Thyra::VectorBase<ST> > p = inArgsT.get_p(l);
    if (Teuchos::nonnull(p)) {
      const Teuchos::RCP<const Tpetra_Vector> pT = ConverterT::getConstTpetraVector(p);
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
  const Teuchos::RCP<Tpetra_Vector> fT_out =
    Teuchos::nonnull(outArgsT.get_f()) ?
    ConverterT::getTpetraVector(outArgsT.get_f()) :
    Teuchos::null;

  const Teuchos::RCP<Tpetra_Operator> W_op_outT =
    Teuchos::nonnull(outArgsT.get_W_op()) ?
    ConverterT::getTpetraOperator(outArgsT.get_W_op()) :
    Teuchos::null;

#ifdef WRITE_MASS_MATRIX_TO_MM_FILE
  //IK, 4/24/15: adding object to hold mass matrix to be written to matrix market file
  const Teuchos::RCP<Tpetra_Operator> Mass =
    Teuchos::nonnull(outArgsT.get_W_op()) ?
    ConverterT::getTpetraOperator(outArgsT.get_W_op()) :
    Teuchos::null;
  //IK, 4/24/15: needed for writing mass matrix out to matrix market file
  const Teuchos::RCP<Tpetra_Vector> ftmp =
    Teuchos::nonnull(outArgsT.get_f()) ?
    ConverterT::getTpetraVector(outArgsT.get_f()) :
    Teuchos::null;
#endif

  // Cast W to a CrsMatrix, throw an exception if this fails
  const Teuchos::RCP<Tpetra_CrsMatrix> W_op_out_crsT =
    Teuchos::nonnull(W_op_outT) ?
    Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(W_op_outT, true) :
    Teuchos::null;

#ifdef WRITE_MASS_MATRIX_TO_MM_FILE
  //IK, 4/24/15: adding object to hold mass matrix to be written to matrix market file
  const Teuchos::RCP<Tpetra_CrsMatrix> Mass_crs =
    Teuchos::nonnull(Mass) ?
    Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(Mass, true) :
    Teuchos::null;
#endif

  //
  // Compute the functions
  //
  bool f_already_computed = false;

  // W matrix
  if (Teuchos::nonnull(W_op_out_crsT)) {
    app->computeGlobalJacobianT(
        alpha, beta, omega, curr_time, x_dotT.get(), x_dotdotT.get(),  *xT,
        sacado_param_vec, fT_out.get(), *W_op_out_crsT);
    f_already_computed = true;
  }

  // df/dp
  for (int l = 0; l < outArgsT.Np(); ++l) {
    const Teuchos::RCP<Thyra::MultiVectorBase<ST> > dfdp_out =
      outArgsT.get_DfDp(l).getMultiVector();

    const Teuchos::RCP<Tpetra_MultiVector> dfdp_outT =
      Teuchos::nonnull(dfdp_out) ?
      ConverterT::getTpetraMultiVector(dfdp_out) :
      Teuchos::null;

    if (Teuchos::nonnull(dfdp_outT)) {
      const Teuchos::RCP<ParamVec> p_vec = Teuchos::rcpFromRef(sacado_param_vec[l]);

      app->computeGlobalTangentT(
          0.0, 0.0, 0.0, curr_time, false, x_dotT.get(), x_dotdotT.get(), *xT,
          sacado_param_vec, p_vec.get(),
          NULL, NULL, NULL, NULL, fT_out.get(), NULL,
          dfdp_outT.get());

      f_already_computed = true;
    }
  }

  // f
  if (app->is_adjoint) {
    const Thyra::ModelEvaluatorBase::Derivative<ST> f_derivT(
        outArgsT.get_f(),
        Thyra::ModelEvaluatorBase::DERIV_TRANS_MV_BY_ROW);

    const Thyra::ModelEvaluatorBase::Derivative<ST> dummy_derivT;

    const int response_index = 0; // need to add capability for sending this in
    app->evaluateResponseDerivativeT(
        response_index, curr_time, x_dotT.get(), x_dotdotT.get(), *xT,
        sacado_param_vec, NULL,
        NULL, f_derivT, dummy_derivT, dummy_derivT, dummy_derivT);
  } else {
    if (Teuchos::nonnull(fT_out) && !f_already_computed) {
      app->computeGlobalResidualT(
          curr_time, x_dotT.get(), x_dotdotT.get(), *xT,
          sacado_param_vec, *fT_out);
    }
  }

  Teuchos::RCP<Tpetra_Vector> xtildeT = Teuchos::rcp(new Tpetra_Vector(xT->getMap())); 
  //compute xtildeT 
  applyLinvML(xT, xtildeT); 

#ifdef WRITE_TO_MATRIX_MARKET
  //writing to MatrixMarket for debug
  char name[100];  //create string for file name
  sprintf(name, "xT_%i.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, xT);
  sprintf(name, "xtildeT_%i.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, xtildeT);
  mm_counter++; 
#endif  

  //std::cout <<"in HVDec evalModelImpl a, b= " << alpha << "  "<< beta <<std::endl;

  if(Teuchos::nonnull(inArgsT.get_x_dot())){
	  std::cout <<"in the if-statement for the update" <<std::endl;
	  fT_out->update(1.0, *xtildeT, 1.0);
  }

  // Response functions
  for (int j = 0; j < outArgsT.Ng(); ++j) {
    const Teuchos::RCP<Thyra::VectorBase<ST> > g_out = outArgsT.get_g(j);
    Teuchos::RCP<Tpetra_Vector> gT_out =
      Teuchos::nonnull(g_out) ?
      ConverterT::getTpetraVector(g_out) :
      Teuchos::null;

    const Thyra::ModelEvaluatorBase::Derivative<ST> dgdxT_out = outArgsT.get_DgDx(j);
    const Thyra::ModelEvaluatorBase::Derivative<ST> dgdxdotT_out = outArgsT.get_DgDx_dot(j);
    // AGS: x_dotdot time integrators not imlemented in Thyra ME yet
    const Thyra::ModelEvaluatorBase::Derivative<ST> dgdxdotdotT_out;
    sanitize_nans(dgdxT_out);
    sanitize_nans(dgdxdotT_out);
    sanitize_nans(dgdxdotdotT_out);

    // dg/dx, dg/dxdot
    if (!dgdxT_out.isEmpty() || !dgdxdotT_out.isEmpty()) {
      const Thyra::ModelEvaluatorBase::Derivative<ST> dummy_derivT;
      app->evaluateResponseDerivativeT(
          j, curr_time, x_dotT.get(), x_dotdotT.get(), *xT,
          sacado_param_vec, NULL,
          gT_out.get(), dgdxT_out,
          dgdxdotT_out, dgdxdotdotT_out, dummy_derivT);
      // Set gT_out to null to indicate that g_out was evaluated.
      gT_out = Teuchos::null;
    }

    // dg/dp
    for (int l = 0; l < outArgsT.Np(); ++l) {
      const Teuchos::RCP<Thyra::MultiVectorBase<ST> > dgdp_out =
        outArgsT.get_DgDp(j, l).getMultiVector();
      const Teuchos::RCP<Tpetra_MultiVector> dgdpT_out =
        Teuchos::nonnull(dgdp_out) ?
        ConverterT::getTpetraMultiVector(dgdp_out) :
        Teuchos::null;

      if (Teuchos::nonnull(dgdpT_out)) {
        const Teuchos::RCP<ParamVec> p_vec = Teuchos::rcpFromRef(sacado_param_vec[l]);
        app->evaluateResponseTangentT(
            j, alpha, beta, omega, curr_time, false,
            x_dotT.get(), x_dotdotT.get(), *xT,
            sacado_param_vec, p_vec.get(),
            NULL, NULL, NULL, NULL, gT_out.get(), NULL,
            dgdpT_out.get());
        gT_out = Teuchos::null;
      }
    }

    if (Teuchos::nonnull(gT_out)) {
      app->evaluateResponseT(
          j, curr_time, x_dotT.get(), x_dotdotT.get(), *xT,
          sacado_param_vec, *gT_out);
    }
  }
}



