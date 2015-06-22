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

QCADT::CoupledPSJacobian::CoupledPSJacobian(int nEigenvals,
                                            const Teuchos::RCP<const Tpetra_Map>& discretizationMap,
                                            int dim, int valleyDegen, double temp,
                                            double lengthUnitInMeters, double energyUnitInElectronVolts,
                                            double effMass, double conductionBandOffset,
                                            Teuchos::RCP<Tpetra_Vector> neg_eigenvals,
                                            Teuchos::RCP<const Tpetra_MultiVector> eigenvecs, 
                                            Teuchos::RCP<Teuchos_Comm const> const & commT): 
    num_models_(nEigenvals+1), 
    nEigenvals_(nEigenvals),
    commT_(commT), 
    neg_eigenvalues(neg_eigenvals), 
    psiVectors(eigenvecs),
    discMap(discretizationMap),
    numDims(dim),
    valleyDegenFactor(valleyDegen),
    temperature(temp),
    length_unit_in_m(lengthUnitInMeters),
    energy_unit_in_eV(energyUnitInElectronVolts),
    effmass(effMass),
    offset_to_CB(conductionBandOffset) 
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif
   int num_discMap_myEls = discMap->getNodeNumElements();
   // dn_dPsi : vectors of dn/dPsi[i] values
   dn_dPsi = Teuchos::rcp(new Tpetra_MultiVector(*psiVectors)); 
   const Teuchos::ArrayRCP<const ST> neg_eigenvalues_constView = neg_eigenvalues->get1dView(); 
   for (int i=0; i<nEigenvals; i++) {
     Teuchos::RCP<Tpetra_Vector> dn_dPsi_i = dn_dPsi->getVectorNonConst(i); 
     Teuchos::RCP<const Tpetra_Vector> psiVectors_i = psiVectors->getVector(i); 
     dn_dPsi_i->scale( n_prefactor(numDims, valleyDegenFactor, temperature, length_unit_in_m, energy_unit_in_eV, effmass)
                          * 2 * n_weight_factor( -neg_eigenvalues_constView[i], numDims, temperature, energy_unit_in_eV), 
                          *psiVectors_i); 
   }
}

QCADT::CoupledPSJacobian::~CoupledPSJacobian()
{
}


// getThyraCoupledJacobian method is similar to getThyraMatrix in panzer
//(Panzer_BlockedTpetraLinearObjFactory_impl.hpp).
Teuchos::RCP<Thyra::LinearOpBase<ST>>
QCADT::CoupledPSJacobian::getThyraCoupledJacobian(Teuchos::RCP<Tpetra_CrsMatrix> Jac_Poisson, 
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Jac_Schrodinger,
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Mass) const
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

  //populate (Poisson,Poisson) block with Jac_Poisson
  Teuchos::RCP<Thyra::LinearOpBase<ST>> block00 = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(Jac_Poisson);
  blocked_op->setNonconstBlock(0, 0, block00);

  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT; 
  size_t numEntriesT;  
  ST val;
  LO colZero = 0;  
  //create map of 1 column
  Teuchos::RCP<Tpetra_Map> oneColMap = Teuchos::rcp(new Tpetra_Map(1, 0, commT_));
  //create graph 
  Teuchos::RCP<Tpetra_CrsGraph> graph = Teuchos::rcp(new Tpetra_CrsGraph(Mass->getMap(), 1));
  graph->fillComplete();  
 
  //populate (Poisson, Schrodinger) blocks with M*dn_dPsi[j]
  for (int j=1; j<block_dim-1; j++) {
    Teuchos::RCP<Tpetra_CrsMatrix> block01_crs = Teuchos::rcp(new Tpetra_CrsMatrix(graph));
    if (dn_dPsi != Teuchos::null) { //FIXME? Is this logic necessary? 
      //Get (j-1)st entry of dn_dPsi
      Teuchos::RCP<const Tpetra_Vector> dn_dPsi_j = dn_dPsi->getVector(j-1);
      const Teuchos::ArrayRCP<const ST> dn_dPsi_j_constView = dn_dPsi_j->get1dView(); 
      //loop over rows of Mass
      for (LO row = 0; row<Mass->getNodeNumRows(); row++) {
        val = 0.0;
        //get number of entries in tow 
        numEntriesT = Mass->getNumEntriesInLocalRow(row);
        matrixEntriesT.resize(numEntriesT);
        matrixIndicesT.resize(numEntriesT);
        //get copy of row  
        Mass->getLocalRowCopy(row, matrixIndicesT(), matrixEntriesT(), numEntriesT); 
        //loop over colums of mass and calculate Mass*dn_dPsi[j] for each row.
        for (LO col=0; col<numEntriesT; col++) {
          val += matrixEntriesT[col]*dn_dPsi_j_constView[matrixIndicesT[col]]; 
        }
        block01_crs->sumIntoLocalValues(row, Teuchos::arrayView(&colZero,1), Teuchos::arrayView(&val,1)); 
      }
    }
    block01_crs->fillComplete();  
    Teuchos::RCP<Thyra::LinearOpBase<ST>> block01 = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(block01_crs);
    blocked_op->setNonconstBlock(0, j, block01); 
  }
  //populate (Poisson, eigenvalue block) TODO

  //populate (Schrodinger, Poisson) blocks with M*psiVectors[i]

  //FIXME: This is temporary to debug other parts of the code!  Need to populate Jacobian correctly. 
  for (int i=1; i<block_dim; i++) {
    for (int j=1; j<block_dim; j++) { 
        if (i == j) 
          blocked_op->setNonconstBlock(i,j, block00); 
     }
   }
/* 
  //populate remaining blocks
  Teuchos::Array<Teuchos::RCP<ImplicitPSJacobian> > implicitJacs; 
  implicitJacs.resize((2+nEigenvals)*(2+nEigenvals)); 
  for (int i=0; i< 2+nEigenvals; i++) {
    for (int j=0; j<2+nEigenvals; j++) {
        implicitJacs[j+i*(2+nEigenvals)] = Teuchos::rcp(new QCADT::ImplicitPSJacobian(nEigenvals,
                                                  discretizationMap, fullPSMap, commT_, dim, valleyDegen, temp,
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
*/
  // all done
  blocked_op->endBlockFill();
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = fancyOStream(rcpFromRef(std::cout));
  std::cout << "blocked_op: " << std::endl;
  blocked_op->describe(*out, Teuchos::VERB_HIGH);
#endif
  return blocked_op;
}

