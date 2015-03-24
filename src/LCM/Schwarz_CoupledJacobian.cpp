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

//#include "Tpetra_LocalMap.h"

using Teuchos::getFancyOStream;
using Teuchos::rcpFromRef;


#define WRITE_TO_MATRIX_MARKET

static int c3 = 0; 
static int c4 = 0; 

using Thyra::PhysicallyBlockedLinearOpBase;


LCM::Schwarz_CoupledJacobian::Schwarz_CoupledJacobian(const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  std::cout << __PRETTY_FUNCTION__ << "\n"; 
  commT_ = commT;
}

LCM::Schwarz_CoupledJacobian::~Schwarz_CoupledJacobian()
{
}

//getThyraCoupledJacobian method is similar analogous to getThyraMatrix in panzer 
//(Panzer_BlockedTpetraLinearObjFactory_impl.hpp).
Teuchos::RCP<Thyra::LinearOpBase<ST> > 
LCM::Schwarz_CoupledJacobian::
getThyraCoupledJacobian(Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> >jacs, 
                        Teuchos::Array<Teuchos::RCP<LCM::Schwarz_BoundaryJacobian> > jacs_boundary) const 
{
  std::cout << __PRETTY_FUNCTION__ << "\n"; 
  std::size_t block_dim = jacs.size(); 

#ifdef WRITE_TO_MATRIX_MARKET
  char name[100];  //create string for file name
  sprintf(name, "Jac0_%i.mm", c3);
//write individual model jacobians to matrix market for debug
  Tpetra_MatrixMarket_Writer::writeSparseFile(name, jacs[0]);
  if (block_dim > 1) {
    sprintf(name, "Jac1_%i.mm", c3);
    Tpetra_MatrixMarket_Writer::writeSparseFile(name, jacs[1]);
  }
  c3++; 
#endif
   // get the block dimension
   // this operator will be square
   Teuchos::RCP<Thyra::PhysicallyBlockedLinearOpBase<ST> > blocked_op = Thyra::defaultBlockedLinearOp<ST>();
   blocked_op->beginBlockFill(block_dim,block_dim);

   // loop over each block
   for(std::size_t i=0;i<block_dim;i++) {
     for(std::size_t j=0;j<block_dim;j++) {
        // build (i,j) block matrix and add it to blocked operator
        if (i == j) { //diagonal blocks
          Teuchos::RCP<Thyra::LinearOpBase<ST> > block = Thyra::createLinearOp<ST,LO,GO,KokkosNode>(jacs[i]);
          blocked_op->setNonconstBlock(i,j,block);
        }
        //FIXME: add off-diagonal blocks
     }
   }

   // all done
   blocked_op->endBlockFill();
   Teuchos::RCP<Teuchos::FancyOStream> out = fancyOStream(rcpFromRef(std::cout));
   std::cout << "blocked_op: " << std::endl; 
   blocked_op->describe(*out, Teuchos::VERB_HIGH);
   return blocked_op;
}


