//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_BoundaryJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
//#include "Tpetra_LocalMap.h"

#define WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
static int
mm_counter = 0;
#endif // WRITE_TO_MATRIX_MARKET

LCM::Schwarz_BoundaryJacobian::Schwarz_BoundaryJacobian(
    Teuchos::RCP<Teuchos_Comm const> const & comm,
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> > const & ca,
    int const this_app_index,
    int const coupled_app_index) :
        commT_(comm),
        coupled_apps_(ca),
        this_app_index_(this_app_index),
        coupled_app_index_(coupled_app_index),
        b_use_transpose_(false),
        b_initialized_(false),
        n_models_(0)
{
}

LCM::Schwarz_BoundaryJacobian::~Schwarz_BoundaryJacobian()
{
}

// Initialize the operator with everything needed to apply it
void LCM::Schwarz_BoundaryJacobian::initialize()
{
  // FIXME: add parameter list argument, member parameters for
  // specifying boundary conditions.
  // These can be stored in an array of Tpetra_CrsMatrices like the jacobians.
  // Set member variables

#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif
}

// Returns the result of a Tpetra_Operator applied to a
// Tpetra_MultiVector X in Y.
void LCM::Schwarz_BoundaryJacobian::apply(
    const Tpetra_MultiVector& X,
    Tpetra_MultiVector& Y,
    Teuchos::ETransp mode,
    ST alpha,
    ST beta) const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

#ifdef WRITE_TO_MATRIX_MARKET
  // writing to MatrixMarket file for debug
  // initial X where we will set Y = Jac*X
  char name[100];  //create string for file name
  sprintf(name, "X_%i.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, X);
#endif  // WRITE_TO_MATRIX_MARKET

  //FIXME: fill in!

#ifdef WRITE_TO_MATRIX_MARKET
  // writing to MatrixMarket file for debug
  // final solution Y (after all the operations to set Y = Jac*X
  sprintf(name, "Y_%i.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, Y);
  ++mm_counter;
#endif  // WRITE_TO_MATRIX_MARKET
}

