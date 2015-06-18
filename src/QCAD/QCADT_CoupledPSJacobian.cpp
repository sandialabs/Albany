//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCADT_CoupledPSJacobian.hpp"
#include "QCADT_ImplicitPSJacobian.hpp"
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

#define OUTPUT_TO_SCREEN

using Thyra::PhysicallyBlockedLinearOpBase;

QCADT::CoupledPSJacobian::CoupledPSJacobian( int num_models, 
    Teuchos::RCP<Teuchos_Comm const> const & commT)
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif
  num_models_ = num_models; 
  commT_ = commT;
}

QCADT::CoupledPSJacobian::~CoupledPSJacobian()
{
}


// getThyraCoupledJacobian method is similar to getThyraMatrix in panzer
//(Panzer_BlockedTpetraLinearObjFactory_impl.hpp).
Teuchos::RCP<Thyra::LinearOpBase<ST>>
QCADT::CoupledPSJacobian::getThyraCoupledJacobian(int nEigenvals,
                                                  const Teuchos::RCP<const Tpetra_Map>& discretizationMap,
                                           	  const Teuchos::RCP<const Tpetra_Map>& fullPSMap,
                                           	  const Teuchos::RCP<const Teuchos_Comm>& comm,
                                           	  int dim, int valleyDegen, double temp,
                                           	  double lengthUnitInMeters, double energyUnitInElectronVolts,
                                           	  double effMass, double conductionBandOffset, 
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Jac_Poisson, 
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Jac_Schrodinger,
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Mass,
                                                  Teuchos::RCP<Tpetra_Vector> neg_eigenvals, 
                                                  Teuchos::RCP<const Tpetra_MultiVector> eigenvecs) const
{
//FIXME: pass necessary variables for Jacobian blocks
#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif
    
    // Jacobian Matrix is:
    //
    //                   Phi                    Psi[i]                            -Eval[i]
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    // Poisson  |    Jac_poisson       |   M*diag(dn/d{Psi[i](x)})   |        -M*col(dn/dEval[i])           |
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    // Schro[j] |  M*diag(-Psi[j](x))  | delta(i,j)*[ H-Eval[i]*M ]  |        delta(i,j)*M*Psi[i](x)        |    
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    // Norm[j]  |    0                 | -delta(i,j)*(M+M^T)*Psi[i]  |                   0                  |
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //
    //
    //   Where:
    //       n = quantum density function which depends on dimension

  int block_dim = num_models_;  
  
  // this operator will be square
  Teuchos::RCP<Thyra::PhysicallyBlockedLinearOpBase<ST>>blocked_op = Thyra::defaultBlockedLinearOp<ST>();
  blocked_op->beginBlockFill(block_dim, block_dim);

  //populate (0,0) block with Jac_Poisson
  Teuchos::RCP<Thyra::LinearOpBase<ST>> block00 = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(Jac_Poisson);
  blocked_op->setNonconstBlock(0, 0, block00);
  
  //populate remaining blocks
  Teuchos::Array<Teuchos::RCP<ImplicitPSJacobian> > implicitJacs; 
  implicitJacs.resize((2+nEigenvals)*(2+nEigenvals)); 
  for (int i=0; i< 2+nEigenvals; i++) {
    for (int j=0; j<2+nEigenvals; j++) {
        implicitJacs[j+i*(2+nEigenvals)] = Teuchos::rcp(new QCADT::ImplicitPSJacobian(nEigenvals,
                                                  discretizationMap, fullPSMap, comm, dim, valleyDegen, temp,
                                                  lengthUnitInMeters, energyUnitInElectronVolts,
                                                  effMass, conductionBandOffset));
        implicitJacs[j+i*(2+nEigenvals)]->initialize(Jac_Schrodinger, Mass, neg_eigenvals, eigenvecs);
        implicitJacs[j+i*(2+nEigenvals)]->setIndices(i, j); 
    }
  }
  //(0, *) blocks
  for (int j=1; j<2+nEigenvals; j++) {
    Teuchos::RCP<Thyra::LinearOpBase<ST>>
        block = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(implicitJacs[j]);
    blocked_op->setNonconstBlock(0,j, block); 
  }
  //(1:1+nEigenvalues, *) blocks
  for (int i=1; i < 1+nEigenvals; i++) {
    for (int j=0; j<2+nEigenvals; j++) {
      Teuchos::RCP<Thyra::LinearOpBase<ST>>
          block = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(implicitJacs[j+i*(2+nEigenvals)]);
      blocked_op->setNonconstBlock(i,j, block); 
    }
  }
  //(2+nEigenvals, *) blocks
    for (int j=1; j<1+nEigenvals; j++) {
      Teuchos::RCP<Thyra::LinearOpBase<ST>>
          block = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(implicitJacs[j+(1+nEigenvals)*(2+nEigenvals)]);
      blocked_op->setNonconstBlock(1+nEigenvals,j, block); 
    }
  
   //(2+nEigenvals, *) blocks
   for (int j=1; j<1+nEigenvals; j++) {
     Teuchos::RCP<Thyra::LinearOpBase<ST>>    
     block = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(implicitJacs[j+(1+nEigenvals)*(2+nEigenvals)]);
     blocked_op->setNonconstBlock(1+nEigenvals,j, block);
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

