//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_BoundaryJacobian.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//
//
//
Schwarz_BoundaryJacobian::Schwarz_BoundaryJacobian(
    Teuchos::RCP<Teuchos_Comm const> const&                     comm,
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> const& ca,
    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>>              jacs,
    int const                                                   this_app_index,
    int const coupled_app_index)
    : comm_(comm),
      coupled_apps_(ca),
      jacs_(jacs),
      this_app_index_(this_app_index),
      coupled_app_index_(coupled_app_index),
      b_use_transpose_(false),
      b_initialized_(false),
      n_models_(0)
{
  ALBANY_EXPECT(0 <= this_app_index && this_app_index < ca.size());
  ALBANY_EXPECT(0 <= coupled_app_index && coupled_app_index < ca.size());
  domain_map_ = ca[coupled_app_index]->getMapT();
  range_map_  = ca[this_app_index]->getMapT();
}

//
//
//
Schwarz_BoundaryJacobian::~Schwarz_BoundaryJacobian() { return; }

//
// Initialize the operator with everything needed to apply it
//
void
Schwarz_BoundaryJacobian::initialize()
{
  return;
}

//
// Returns explicit matrix representation of operator if available.
//
Teuchos::RCP<Tpetra_CrsMatrix>
Schwarz_BoundaryJacobian::getExplicitOperator() const
{
  auto const max_num_cols = getDomainMap()->getNodeNumElements();

  Teuchos::RCP<Tpetra_CrsMatrix> K = Teuchos::rcp(
      new Tpetra_CrsMatrix(getRangeMap(), getDomainMap(), max_num_cols));

  auto const zero = Teuchos::ScalarTraits<ST>::zero();

  K->setAllToScalar(zero);

  K->fillComplete();

  return K;
}

//
// Returns the result of a Tpetra_Operator applied to a
// Tpetra_MultiVector X in Y.
//
void
Schwarz_BoundaryJacobian::apply(
    Tpetra_MultiVector const& X,
    Tpetra_MultiVector&       Y,
    Teuchos::ETransp          mode,
    ST                        alpha,
    ST                        beta) const
{
  auto const zero = Teuchos::ScalarTraits<ST>::zero();

  Y.putScalar(zero);
}

}  // namespace LCM
