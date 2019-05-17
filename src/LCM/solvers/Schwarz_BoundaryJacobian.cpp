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
    Teuchos::Array<Teuchos::RCP<Thyra_LinearOp>>                jacs,
    int const                                                   this_app_index,
    int const coupled_app_index)
    : comm_(comm),
      coupled_apps_(ca),
      jacs_(jacs),
      this_app_index_(this_app_index),
      coupled_app_index_(coupled_app_index),
      n_models_(0)
{
  ALBANY_EXPECT(0 <= this_app_index && this_app_index < ca.size());
  ALBANY_EXPECT(0 <= coupled_app_index && coupled_app_index < ca.size());
  domain_vs_ = ca[coupled_app_index]->getVectorSpace();
  range_vs_  = ca[this_app_index]->getVectorSpace();
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
Teuchos::RCP<Thyra_LinearOp>
Schwarz_BoundaryJacobian::getExplicitOperator() const
{
  auto const max_num_cols = Albany::getNumLocalElements(this->domain());

  // IKT: there may be problems here in creating jac_factory - will need to
  // check
  Teuchos::RCP<Albany::ThyraCrsMatrixFactory> jac_factory =
      Teuchos::rcp(new Albany::ThyraCrsMatrixFactory(
          this->range(), this->domain(), max_num_cols));

  jac_factory->fillComplete();

  Teuchos::RCP<Thyra_LinearOp> K = jac_factory->createOp();

  return K;
}

//
// Returns the result of a Thyra_Operator applied to a
// Thyra_MultiVector X in Y.
//
void
Schwarz_BoundaryJacobian::applyImpl(
    const Thyra::EOpTransp                 M_trans,
    const Thyra_MultiVector&               X,
    const Teuchos::Ptr<Thyra_MultiVector>& Y,
    const ST                               alpha,
    const ST                               beta) const
{
  auto const zero = Teuchos::ScalarTraits<ST>::zero();

  Y->assign(zero);
}

}  // namespace LCM
