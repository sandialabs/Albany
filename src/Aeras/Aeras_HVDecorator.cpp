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

//Let's keep this for later
  Teuchos::ParameterList &coupled_system_params = appParams->sublist("Coupled System");

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




