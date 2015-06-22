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
  LO colZero = 0;  
  //create graph for matrix with 1 column 
  Teuchos::RCP<Tpetra_CrsGraph> graph1col = Teuchos::rcp(new Tpetra_CrsGraph(Mass->getMap(), 1));
  graph1col->fillComplete(); 
  //create graph for matrix with nEigenvals columns
  Teuchos::RCP<Tpetra_CrsGraph> graphEvalsCols = Teuchos::rcp(new Tpetra_CrsGraph(Mass->getMap(), nEigenvals_)); 
  graphEvalsCols->fillComplete();  
 
  //populate (Poisson, Schrodinger) blocks with M*dn_dPsi[j] -- nEvals matrices with 1 column
  for (int j=1; j<block_dim-1; j++) {
    Teuchos::RCP<Tpetra_CrsMatrix> blockPS_crs = Teuchos::rcp(new Tpetra_CrsMatrix(graph1col));
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
        blockPS_crs->sumIntoLocalValues(row, Teuchos::arrayView(&colZero,1), Teuchos::arrayView(&val,1)); 
      }
    }
    blockPS_crs->fillComplete();  
    Teuchos::RCP<Thyra::LinearOpBase<ST>> blockPS = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(blockPS_crs);
    blocked_op->setNonconstBlock(0, j, blockPS); 
  }
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

  //populate (Schrodinger, Poisson) blocks with M*psiVectors[i] -- nEvals matrices with 1 column
  for (int i=1; i<block_dim-1; i++) {
    Teuchos::RCP<Tpetra_CrsMatrix> blockSP_crs = Teuchos::rcp(new Tpetra_CrsMatrix(graph1col));
    if (psiVectors != Teuchos::null) { 
      //Get (i-1)st entry of psiVectors
      Teuchos::RCP<const Tpetra_Vector> psiVectors_i = psiVectors->getVector(i-1); 
      const Teuchos::ArrayRCP<const ST> psiVectors_i_constView = psiVectors_i->get1dView(); 
      //loop over rows of Mass
      for (LO row = 0; row<Mass->getNodeNumRows(); row++) {
        val = 0.0; 
        //get number of entries in tow 
        numEntriesT = Mass->getNumEntriesInLocalRow(row);
        matrixEntriesT.resize(numEntriesT);
        matrixIndicesT.resize(numEntriesT);
        //get copy of row  
        Mass->getLocalRowCopy(row, matrixIndicesT(), matrixEntriesT(), numEntriesT); 
        //loop over colums of mass and calculate -Mass*psiVectors[i] for each row.
        for (LO col=0; col<numEntriesT; col++) {
          val += -1*matrixEntriesT[col]*psiVectors_i_constView[matrixIndicesT[col]]; 
        }
        blockSP_crs->sumIntoLocalValues(row, Teuchos::arrayView(&colZero,1), Teuchos::arrayView(&val,1)); 
      }
    }
    blockSP_crs->fillComplete();  
    Teuchos::RCP<Thyra::LinearOpBase<ST>> blockSP = Thyra::createLinearOp<ST, LO, GO, KokkosNode>(blockSP_crs);
    blocked_op->setNonconstBlock(i, 0, blockSP);
  }
  //Populate (Schrodinger, Schrodinger) block TODO 
  //
  //Populate (Schrodinger, eigenvalue) block with delta(i,j)*M*Psi[i] -- nEvals matrices with nEvals columns
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
          //loop over columns of mass and calculate Mass*dn_dEval[j]
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
  //Populate (eigenvalue, Schrodinger) block TODO 


  //FIXME: This is temporary to debug other parts of the code!  Need to populate Jacobian correctly. 
  for (int i=block_dim-1; i<block_dim; i++) {
    for (int j=block_dim-1; j<block_dim; j++) { 
        if (i == j) 
          blocked_op->setNonconstBlock(i,j, blockPP); 
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
  //(1:1+nEigenvals, *) blocks
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

