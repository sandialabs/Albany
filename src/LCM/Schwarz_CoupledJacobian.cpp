//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_CoupledJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"
#include "Thyra_DefaultBlockedLinearOp.hpp"

using Teuchos::getFancyOStream;
using Teuchos::rcpFromRef;

//#define WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
static int
mm_counter = 0;
#endif // WRITE_TO_MATRIX_MARKET

//#define OUTPUT_TO_SCREEN

using Thyra::PhysicallyBlockedLinearOpBase;

LCM::Schwarz_CoupledJacobian::Schwarz_CoupledJacobian(
    Teuchos::RCP<Teuchos_Comm const> const & commT)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif
  commT_ = commT;
}

LCM::Schwarz_CoupledJacobian::~Schwarz_CoupledJacobian()
{
}

#define EXPLICIT_OFF_DIAGONAL

// getThyraCoupledJacobian method is similar to getThyraMatrix in panzer
//(Panzer_BlockedTpetraLinearObjFactory_impl.hpp).
Teuchos::RCP<Thyra::LinearOpBase<ST>>
LCM::Schwarz_CoupledJacobian::
getThyraCoupledJacobian(
    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>> jacs,
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> const & ca)
const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  auto const
  block_dim = jacs.size();

#ifdef WRITE_TO_MATRIX_MARKET
  char name[100];  //create string for file name

  for (auto i = 0; i < block_dim; ++i) {
    sprintf(name, "Jac%02d_%04d.mm", i, mm_counter);
    Tpetra_MatrixMarket_Writer::writeSparseFile(name, jacs[i]);
  }
  mm_counter++;
#endif // WRITE_TO_MATRIX_MARKET

  // get the block dimension
  // this operator will be square
  Teuchos::RCP<Thyra::PhysicallyBlockedLinearOpBase<ST>>
  blocked_op = Thyra::defaultBlockedLinearOp<ST>();

  blocked_op->beginBlockFill(block_dim, block_dim);

  // loop over each block
  for (std::size_t i = 0; i < block_dim; i++) {
    for (std::size_t j = 0; j < block_dim; j++) {
      // build (i,j) block matrix and add it to blocked operator
      if (i == j) { // Diagonal blocks
        Teuchos::RCP<Thyra::LinearOpBase<ST>>
        block = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(jacs[i]);
        blocked_op->setNonconstBlock(i, j, block);
      } else { // Off-diagonal blocks

#if defined(EXPLICIT_OFF_DIAGONAL)

        Teuchos::RCP<Schwarz_BoundaryJacobian>
        jac_boundary =
            Teuchos::rcp(
                new LCM::Schwarz_BoundaryJacobian(commT_, ca, jacs, i, j));

        Teuchos::RCP<Tpetra_CrsMatrix>
        exp_jac = jac_boundary->getExplicitOperator();

        Teuchos::RCP<Thyra::LinearOpBase<ST>>
        block = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(exp_jac);

#else

        Teuchos::RCP<Tpetra_Operator>
        jac_boundary =
            Teuchos::rcp(
                new LCM::Schwarz_BoundaryJacobian(commT_, ca, jacs, i, j));

        Teuchos::RCP<Thyra::LinearOpBase<ST>>
        block = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(jac_boundary);

#endif // EXPLICIT_OFF_DIAGONAL

        blocked_op->setNonconstBlock(i, j, block);
      }
    }
  }

  // all done
  blocked_op->endBlockFill();
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = fancyOStream(rcpFromRef(std::cout));
  std::cout << "blocked_op: " << std::endl;
  blocked_op->describe(*out, Teuchos::VERB_HIGH);
#endif
  return blocked_op;
}

