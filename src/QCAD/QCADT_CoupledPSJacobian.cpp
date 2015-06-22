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
#include "TpetraExt_MatrixMatrix_def.hpp"

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
   // dn_dEval : vectors of dn/dEval[i]
   dn_dEval = Teuchos::rcp(new Tpetra_MultiVector( discMap, nEigenvals ));
   for (int i=0; i<nEigenvals; i++) {
     double prefactor = n_prefactor(numDims, valleyDegenFactor, temperature, length_unit_in_m, energy_unit_in_eV, effmass);
     double dweight = dn_weight_factor(-neg_eigenvalues_constView[i], numDims, temperature, energy_unit_in_eV);
     //DEBUG
     //double eps = 1e-7;
     //double dweight = (n_weight_factor( -(*neg_eigenvalues)[i] + eps, numDims, temperature, energy_unit_in_eV) - 
     //n_weight_factor( -(*neg_eigenvalues)[i], numDims, temperature, energy_unit_in_eV)) / eps; 
     //const double kbBoltz = 8.617343e-05;
     //std::cout << "DEBUG: dn_dEval["<<i<<"] dweight arg = " <<  (*neg_eigenvalues)[i]/(kbBoltz*temperature) << std::endl;  
     //in [eV]
     //std::cout << "DEBUG: dn_dEval["<<i<<"] factor = " <<  prefactor * dweight << std::endl;  // in [eV]
     //DEBUG
     Teuchos::RCP<Tpetra_Vector> dn_dEval_i = dn_dEval->getVectorNonConst(i); 
     Teuchos::RCP<const Tpetra_Vector> psiVectors_i = psiVectors->getVector(i); 
     for (int k=0; k<num_discMap_myEls; k++) {
        const Teuchos::ArrayRCP<ST> dn_dEval_i_nonconstView = dn_dEval_i->get1dViewNonConst();
        const Teuchos::ArrayRCP<const ST> psiVectors_i_constView = psiVectors_i->get1dView();
        dn_dEval_i_nonconstView[k] =  prefactor * pow( psiVectors_i_constView[k], 2.0 ) * dweight;
    //(*dn_dEval)(i)->Print( std::cout << "DEBUG: dn_dEval["<<i<<"]:" << std::endl );
     }
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
  Teuchos::RCP<Thyra::LinearOpBase<ST>> blockPP = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(Jac_Poisson);
  blocked_op->setNonconstBlock(0, 0, blockPP);

  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT; 
  size_t numEntriesT;  
  ST val;
  //create graph for matrix with nEigenvals columns
  Teuchos::RCP<Tpetra_CrsGraph> graphEvalsCols = Teuchos::rcp(new Tpetra_CrsGraph(Mass->getMap(), nEigenvals_)); 
  graphEvalsCols->fillComplete();  
  //create graph for diagonal 
  Teuchos::RCP<Tpetra_CrsGraph> graphDiag = Teuchos::rcp(new Tpetra_CrsGraph(Mass->getMap(), 1));
  graphDiag->fillComplete(); 
 
  //populate (Poisson, Schrodinger) blocks with M*diag(dn_dPsi[j])
  //fillComplete() Mass matrix -- necessary for Multiply command below
  //Why is Mass not passed in fillCompleted? 
  Mass->fillComplete(); 
  for (int j=1; j<block_dim-1; j++) {
    Teuchos::RCP<Tpetra_CrsMatrix> blockPS_crs = Teuchos::rcp(new Tpetra_CrsMatrix(Mass->getMap(), Mass->getGlobalMaxNumRowEntries()));
    //Get (j-1)st entry of dn_dPsi
    Teuchos::RCP<const Tpetra_Vector> dn_dPsi_j = dn_dPsi->getVector(j-1);
    const Teuchos::ArrayRCP<const ST> dn_dPsi_j_constView = dn_dPsi_j->get1dView(); 
    //Create CrsMatrix whose diagonal is dn_dPsi_j
    //IKT: Is there a Tpetra command to create diagonal CrsMatrix from Vector??
    Teuchos::RCP<Tpetra_CrsMatrix> dn_dPsi_j_crs = Teuchos::rcp(new Tpetra_CrsMatrix(graphDiag));
    for (LO row = 0; row < dn_dPsi_j_crs->getNodeNumRows(); row++){
      val = dn_dPsi_j_constView[row]; 
      dn_dPsi_j_crs->replaceLocalValues(row, Teuchos::arrayView(&row,1), Teuchos::arrayView(&val,1)); 
    }
    dn_dPsi_j_crs->fillComplete(); 
    //blockPS_crs = M*diag(dn_dPsi[j]) 
    Tpetra::MatrixMatrix::Multiply(*Mass, false, *dn_dPsi_j_crs, false, *blockPS_crs, true); 
    Teuchos::RCP<Thyra::LinearOpBase<ST>> blockPS = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(blockPS_crs);
    blocked_op->setNonconstBlock(0, j, blockPS); 
  }
  //FIXME, IKT, 6/22/15: check with Erik what does col mean.
  //populate (Poisson, eigenvalue) block with -M*col(dn/dEval[i]) -- 1 matrix with nEigenvals columns
  Teuchos::RCP<Tpetra_CrsMatrix> blockPE_crs = Teuchos::rcp(new Tpetra_CrsMatrix(graphEvalsCols));
  for (int i=0; i<nEigenvals_; i++) {
    if (dn_dEval != Teuchos::null) {
      //get ith dn_dEval vector
      Teuchos::RCP<const Tpetra_Vector> dn_dEval_i = dn_dEval->getVector(i); 
      const Teuchos::ArrayRCP<const ST> dn_dEval_i_constView = dn_dEval_i->get1dView(); 
      //loop over rows of Mass Matrix
      for (LO row = 0; row<Mass->getNodeNumRows(); row++) {
        val = 0.0;
        //get number of entries in tow 
        numEntriesT = Mass->getNumEntriesInLocalRow(row);
        matrixEntriesT.resize(numEntriesT);
        matrixIndicesT.resize(numEntriesT);
        //get copy of row  
        Mass->getLocalRowCopy(row, matrixIndicesT(), matrixEntriesT(), numEntriesT); 
        //loop over columns of mass and calculate -Mass*dn_dEval[i]
        for (LO col=0; col<numEntriesT; col++) {
          val += -1.0*matrixEntriesT[col]*dn_dEval_i_constView[matrixIndicesT[col]]; 
        }
        //Sum into ith column of blockPE_crs
        blockPE_crs->sumIntoLocalValues(row, Teuchos::arrayView(&i,1), Teuchos::arrayView(&val,1)); 
      }
    }
  }
  blockPE_crs->fillComplete();  
  Teuchos::RCP<Thyra::LinearOpBase<ST>> blockPE = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(blockPE_crs);
  blocked_op->setNonconstBlock(0, block_dim-1, blockPE); 

  //populate (Schrodinger, Poisson) blocks with -M*diag(psiVectors[i]) 
  for (int i=1; i<block_dim-1; i++) {
    Teuchos::RCP<Tpetra_CrsMatrix> blockSP_crs = Teuchos::rcp(new Tpetra_CrsMatrix(Mass->getMap(), Mass->getGlobalMaxNumRowEntries()));
    //Get (i-1)st entry of psiVectors
    Teuchos::RCP<const Tpetra_Vector> psiVectors_i = psiVectors->getVector(i-1); 
    const Teuchos::ArrayRCP<const ST> psiVectors_i_constView = psiVectors_i->get1dView(); 
    //Create CrsMatrix whose diagonal -psiVectors_i
    //IKT: Is there a Tpetra command to create diagonal CrsMatrix from Vector??
    Teuchos::RCP<Tpetra_CrsMatrix> psiVectors_i_crs = Teuchos::rcp(new Tpetra_CrsMatrix(graphDiag));
    for (LO row = 0; row < psiVectors_i_crs->getNodeNumRows(); row++){
      val = -1.0*psiVectors_i_constView[row]; 
      psiVectors_i_crs->replaceLocalValues(row, Teuchos::arrayView(&row,1), Teuchos::arrayView(&val,1)); 
    }
    psiVectors_i_crs->fillComplete(); 
    //blockSP_crs = =M*diag(psiVectors[i]) 
    Tpetra::MatrixMatrix::Multiply(*Mass, false, *psiVectors_i_crs, false, *blockSP_crs, true); 
    Teuchos::RCP<Thyra::LinearOpBase<ST>> blockSP = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(blockSP_crs);
    blocked_op->setNonconstBlock(i, 0, blockSP); 
  }
  //Populate (Schrodinger, Schrodinger) block with delta(i,j)*(H-eval[j]*M)
  //Call fill complete on Jac_Schrodinger
  //Jac_Schrodinger->fillComplete();  
  const Teuchos::ArrayRCP<const ST> neg_eigenvalues_constView = neg_eigenvalues->get1dView();
  for (int i=1; i<block_dim-1; i++) {
    for (int j=1; j<block_dim-1; j++) {
      //Create CrsMatrix to hold SS block
      Teuchos::RCP<Tpetra_CrsMatrix> blockSS_crs = Teuchos::rcp(new Tpetra_CrsMatrix(Jac_Schrodinger->getMap(), 
                                                                    Jac_Schrodinger->getGlobalMaxNumRowEntries()));
      //blockSS_crs = H-eval[j]*M
      Tpetra::MatrixMatrix::Add(*Jac_Schrodinger, false, 1.0, *Mass, false, neg_eigenvalues_constView[j-1], blockSS_crs); 
      blockSS_crs->fillComplete(); 
      Teuchos::RCP<Thyra::LinearOpBase<ST>> blockSS = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(blockSS_crs);
      blocked_op->setNonconstBlock(i, j, blockSS); 
    }
  } 
  //
  //Populate (Schrodinger, eigenvalue) block with delta(i,j)*M*Psi[j] 
  for (int i=1; i<block_dim-1; i++) {
    Teuchos::RCP<Tpetra_CrsMatrix> blockSE_crs = Teuchos::rcp(new Tpetra_CrsMatrix(graphEvalsCols));
    for (int j=0; j<nEigenvals_; j++) {
      if (dn_dEval != Teuchos::null) {
        //get jth psiVectors vector
        Teuchos::RCP<const Tpetra_Vector> psiVectors_j = psiVectors->getVector(j); 
        const Teuchos::ArrayRCP<const ST> psiVectors_j_constView = psiVectors_j->get1dView(); 
        //loop over rows of Mass Matrix
        for (LO row = 0; row<Mass->getNodeNumRows(); row++) {
          val = 0.0;
          //get number of entries in tow 
          numEntriesT = Mass->getNumEntriesInLocalRow(row);
          matrixEntriesT.resize(numEntriesT);
          matrixIndicesT.resize(numEntriesT);
          //get copy of row  
          Mass->getLocalRowCopy(row, matrixIndicesT(), matrixEntriesT(), numEntriesT); 
          //loop over columns of mass and calculate Mass*psiVectors[j]
          for (LO col=0; col<numEntriesT; col++) {
            val += matrixEntriesT[col]*psiVectors_j_constView[matrixIndicesT[col]]; 
          }
         //Sum into jth column of blockPE_crs
         blockSE_crs->sumIntoLocalValues(row, Teuchos::arrayView(&j,1), Teuchos::arrayView(&val,1)); 
        }
      }
    }
    blockSE_crs->fillComplete();  
    Teuchos::RCP<Thyra::LinearOpBase<ST>> blockSE = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(blockSE_crs);
    blocked_op->setNonconstBlock(i, 0, blockSE);
  }
  //Populate (eigenvalue, Schrodinger) block
  //FIXME, IKT, 6/22/15: ask Erik what this block should be! 
  //FIXME: This is temporary to debug other parts of the code!  Need to populate Jacobian correctly. 
   blocked_op->setNonconstBlock(2,1, blockPP); 

  // all done
  blocked_op->endBlockFill();
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = fancyOStream(rcpFromRef(std::cout));
  std::cout << "blocked_op: " << std::endl;
  blocked_op->describe(*out, Teuchos::VERB_HIGH);
#endif
  return blocked_op;
}

