/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/




#include "AnasaziConfigDefs.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBlockDavidsonSolMgr.hpp"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"

#include "QCAD_GreensFunctionTunneling.hpp"

// Assume EcValues are in units of eV
// Assume effMass is in units of m_0 (electron rest mass)
// Assume ptSpacing is in units of microns (um)
QCAD::GreensFunctionTunnelingSolver::
GreensFunctionTunnelingSolver(const Teuchos::RCP<std::vector<double> >& EcValues_, 
			      double ptSpacing_, double effMass, const Teuchos::RCP<Epetra_Comm>& Comm_,
			      bool bNeumannBC)
{  
  Comm = Comm_;
  EcValues = EcValues_;
  ptSpacing = ptSpacing_;
  mass = effMass;
  bUseNeumannBC = bNeumannBC;
}

QCAD::GreensFunctionTunnelingSolver::
~GreensFunctionTunnelingSolver()
{
}


double QCAD::GreensFunctionTunnelingSolver::
computeCurrent(double Vds, double kbT, double Ecutoff_offset_from_Emax)
{
  const double hbar_1 = 6.58212e-16; // eV * s
  const double hbar_2 = 1.05457e-34; // J * s = kg * m^2 / s
  const double m_0 = 9.11e-31; // kg
  const double um = 1e-6; // m
  const double min_a0 = 1e-4; // um, so == .1nm  HARDCODED MIN PT SPACING

  std::vector<Anasazi::Value<double> > evals;
  Teuchos::RCP<Epetra_MultiVector> evecs;

  //Setup Energy bins:  mu_L == 0, mu_R == -Vds, so set
  // Emin = Ec[0] + mu_L + 20kT
  // Emax = Ec[nPts-1] + mu_R
  // dE = kbT / 10
  int nPts    = EcValues->size();
  double Emax = (*EcValues)[0] + 20*kbT;  // should this just be +20*kbT  (Suzey??)
  double Emin = std::min((*EcValues)[0],(*EcValues)[nPts-1]) - Vds;  // assume one of endpoints is min(EcValues)
  double dE = kbT / 10;
  int nEPts = int((Emax - Emin) / dE) + 1;
  double a0 = ptSpacing;
  bool ret;

  double Ecutoff = Emax + Ecutoff_offset_from_Emax; // maximum eigenvalue needed

  Teuchos::RCP<std::vector<double> > pEc = Teuchos::null;
  Teuchos::RCP<std::vector<double> > pLastEc = Teuchos::null;

  Map = Teuchos::rcp(new Epetra_Map(nPts, 0, *Comm));
  t0  = hbar_1 * hbar_2 /(2*mass*m_0* pow(a0*um,2) ); // gives t0 in units of eV    

  std::cout << "Doing Initial H-mx diagonalization for Vds = " << Vds << std::endl;
  ret = doMatrixDiag(Vds, *EcValues, Ecutoff, evals, evecs);
  pEc = EcValues;
  std::cout << "  Diag w/ a0 = " << a0 << ", nPts = " << nPts << " gives "
	    << "Max Eval = " << evals[evals.size()-1].realpart 
	    << "(need >= "<<Ecutoff<<")" << std::endl;

  while(ret == false && a0 > min_a0) {
    a0 /= 2; nPts *= 2;  
    t0 = hbar_1 * hbar_2 /(2*mass*m_0* pow(a0*um,2) );
    Map = Teuchos::rcp(new Epetra_Map(nPts, 0, *Comm));

    // Interpolate pLastEc onto pEc
    pLastEc = pEc;
    pEc = Teuchos::rcp(new std::vector<double>(nPts));
    for(int i=0; i<nPts; i++) {
      if(i%2 && i/2+1<nPts/2) (*pEc)[i] = ((*pLastEc)[i/2] + (*pLastEc)[i/2+1])/2.0;
      else (*pEc)[i] = (*pLastEc)[i/2];
    }

    // Setup and diagonalize H matrix
    ret = doMatrixDiag(Vds, *pEc, Ecutoff, evals, evecs);
    std::cout << "  Diag w/ a0 = " << a0 << ", nPts = " << nPts << " gives "
	      << "Max Eval = " << evals[evals.size()-1].realpart 
	      << "(need >= "<<Ecutoff<<")" << std::endl;
  }
  std::cout << "Done H-mx diagonalization, now broadcasting results" << std::endl;


  // Since all we need are the eigenvalues at beginning and end (index [0] and [nPts-1] ?)
  //  then broadcast these values to all processors
  int nEvecs = evecs->NumVectors();
  int GIDlist[2];
  GIDlist[0] = 0; // "beginning" index
  GIDlist[1] = nPts-1; // "ending" index
  
  //Get the IDs (rank) of the processors holding beginning and ending index
  std::vector<int> PIDlist(2), LIDlist(2);
  Map->RemoteIDList(2, GIDlist, &PIDlist[0], &LIDlist[0]);

  // Broadcast the beginning and ending elements of each eigenvector to all processors
  std::vector<double> evecBeginEls(nEvecs);
  std::vector<double> evecEndEls(nEvecs);

  if(Comm->MyPID() == PIDlist[0]) { // this proc owns beginning point
    for(int i=0; i<nEvecs; i++)
      evecBeginEls[i] = (*evecs)[i][LIDlist[0]]; // check that this is correct: Epetra_Vector [] operator takes *local* index?
  }
  Comm->Broadcast( &evecBeginEls[0], nEvecs, PIDlist[0] );

  if(Comm->MyPID() == PIDlist[1]) { // this proc owns beginning point
    for(int i=0; i<nEvecs; i++)
      evecEndEls[i] = (*evecs)[i][LIDlist[1]]; // check that this is correct: Epetra_Vector [] operator takes *local* index?
  }
  Comm->Broadcast( &evecEndEls[0], nEvecs, PIDlist[1] );

  // Potential energies, assumed to be constant, in each lead
  double VL = (*pEc)[0];
  double VR = (*pEc)[nPts-1] - Vds;

  std::cout << nEPts << " Energy Pts: " << Emin << " to " << Emax << " eV in steps of " << dE << std::endl;

  // Spread energy points evenly across processors
  Epetra_Map EnergyMap(nEPts, 0, *Comm);

  int NumMyEnergyPts = EnergyMap.NumMyElements();

  std::vector<int> MyGlobalEnergyPts(NumMyEnergyPts);
  EnergyMap.MyGlobalElements(&MyGlobalEnergyPts[0]);

  double I, Iloc = 0, x, y;
  double E, Gamma11, GammaNN, G11, G1N, GNN, T;
  std::complex<double> Sigma11, SigmaNN, G, p11, pNN;
  
  // add up contributions to energy integral from current processor
  for (int i=0; i<NumMyEnergyPts; i++) {
    E = Emin + MyGlobalEnergyPts[i]*dE;

    x = (E-VL)*(t0-(E-VL)/4);
    y = bUseNeumannBC ? 0 : t0;
    if( x >= 0 )
      Sigma11 = std::complex<double>( (E-VL)/2 - y, -sqrt(x) );
    else
      Sigma11 = std::complex<double>( (E-VL)/2 - y + sqrt(-x), 0);

    x = (E-VR)*(t0-(E-VR)/4);
    if( x >= 0 )
      SigmaNN = std::complex<double>( (E-VR)/2 - y, -sqrt(x) );
    else
      SigmaNN = std::complex<double>( (E-VR)/2 - y + sqrt(-x ), 0);

    Gamma11 = 2*Sigma11.imag();
    GammaNN = 2*SigmaNN.imag();

    G11 = G1N = GNN = 0;
    for(int j=0; j<nEvecs; j++) {
      G11 += evecBeginEls[j] * evecBeginEls[j] / (E - evals[j].realpart);
      G1N += evecBeginEls[j] * evecEndEls[j] / (E - evals[j].realpart);
      GNN += evecEndEls[j] * evecEndEls[j] / (E - evals[j].realpart);
    }

    p11 = Sigma11*G11; pNN = SigmaNN*GNN; 
    G = G1N / ((p11-1.0)*(pNN-1.0) - Sigma11*SigmaNN*G1N*G1N);
    T = Gamma11 * GammaNN * std::norm(G);

    //std::cout << "Ept " << i << " Sigma11=" << Sigma11 << " SigmaNN=" << SigmaNN << " Gamma11=" << Gamma11 << " GammaNN=" << GammaNN << " G11=" << G11 << " G1N=" << G1N << " GNN=" << GNN << " G=" << G << " T=" << T << std::endl;

    Iloc += T * f0((E-0)/kbT) * f0((E+Vds)/kbT) * dE;
  }

  std::cout << "Energy Integral contrib from proc " << Comm->MyPID() << " = " << Iloc << std::endl;

  // add contributions from all processors
  Comm->SumAll(&Iloc, &I, 1);

  std::cout << "Total Energy Integral = " << I << " eV" << std::endl;
  
  const double h = 4.13567e-15; // eV * s
  double q = 1.602e-19; // C
  I = 2*q / h * I;  // since I was in units of eV, now I is in units of Amps (C/s)

  return I;
}

double QCAD::GreensFunctionTunnelingSolver::f0(double x) const
{
  return 1.0 / (1 + exp(x));
}

bool QCAD::GreensFunctionTunnelingSolver::
doMatrixDiag(double Vds, std::vector<double>& Ec, double Ecutoff,
	     std::vector<Anasazi::Value<double> >& evals, 
	     Teuchos::RCP<Epetra_MultiVector>& evecs)
{
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;

  bool bPrintResults = false;

  // Get number of local pts for this proc from newly created Map.
  int nPts = Ec.size();
  int nLocalEls = Map->NumMyElements();

  std::vector<int> myGlobalElements(nLocalEls);
  Map->MyGlobalElements(&myGlobalElements[0]);

  // Create an integer vector NumNz that is used to build the Petra Matrix.
  // NumNz[i] is the Number of OFF-DIAGONAL term for the ith H-Mx row
  // on this processor
  std::vector<int> NumNz(nLocalEls);

  /* We are building a matrix of block structure (left = DBC, right = NBC):
  
      | Ec+2t0  -t0                  |          | Ec+t0   -t0                  |
      | -t0    Ec+2t0  -t0           |	        | -t0    Ec+2t0  -t0           |
      |         -t0   ...            |	 OR     |         -t0   ...            |
      |                    ..    -t0 |	        |                    ..    -t0 |
      |                   -t0  Ec+2t0|	        |                   -t0  Ec+t0 |

   where the matrix has nPts rows and nPts columns
  */
  for (int i=0; i<nLocalEls; i++) {
    if (myGlobalElements[i] == 0 || myGlobalElements[i] == nPts-1)
      NumNz[i] = 2;
    else
      NumNz[i] = 3;
  }

  // Create an Epetra_Matrix
  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp( new Epetra_CrsMatrix(Copy, *Map, &NumNz[0]) );

  // Fill Hamiltonian Matrix
  std::vector<double> Values(3), endValues(3);
  std::vector<int> Indices(3);
  int NumEntries;

  Values[0] = -t0; Values[1] = 2*t0; Values[2] = -t0;
  if(bUseNeumannBC) {
    endValues[0] = -t0; endValues[1] =   t0; endValues[2] = -t0;  }
  else {
    endValues[0] = -t0; endValues[1] = 2*t0; endValues[2] = -t0;  }

  
  for (int i=0; i<nLocalEls; i++) {
    double Ulin = -Vds * ((double)myGlobalElements[i]) / nPts;
    Values[1] = 2*t0 + Ec[myGlobalElements[i]] + Ulin;

    if (myGlobalElements[i]==0) {
      Indices[0] = 0;
      Indices[1] = 1;
      NumEntries = 2;
      int info = A->InsertGlobalValues(myGlobalElements[i], NumEntries, &endValues[1], &Indices[0]);
      TEUCHOS_TEST_FOR_EXCEPTION( info != 0, std::runtime_error, "Failure in InsertGlobalValues()" );
    }
    else if (myGlobalElements[i] == nPts-1) {
      Indices[0] = nPts-2;
      Indices[1] = nPts-1;
      NumEntries = 2;
      int info = A->InsertGlobalValues(myGlobalElements[i], NumEntries, &endValues[0], &Indices[0]);
      TEUCHOS_TEST_FOR_EXCEPTION( info != 0, std::runtime_error, "Failure in InsertGlobalValues()" );
    }
    else {
      Indices[0] = myGlobalElements[i]-1;
      Indices[1] = myGlobalElements[i];
      Indices[2] = myGlobalElements[i]+1;
      NumEntries = 3;
      int info = A->InsertGlobalValues(myGlobalElements[i], NumEntries, &Values[0], &Indices[0]);
      TEUCHOS_TEST_FOR_EXCEPTION( info != 0, std::runtime_error, "Failure in InsertGlobalValues()" );
    }
  }

  // Finish up
  int info = A->FillComplete();
  assert( info==0 );
  A->SetTracebackMode(1); // Shutdown Epetra Warning tracebacks

  
  //************************************
  // Call the Block Davidson solver manager
  //***********************************
  //
  //  Variables used for the Block Davidson Method
  //
  std::string  which("SR");
  const int    nev         = nPts/2; //get half the eigenvalues (most we can with Block Davidson?)
  const int    blockSize   = nPts/2;
  const int    numBlocks   = 2;
  const int    maxRestarts = 100;
  const double tol         = 1.0e-8;

  // Create an Epetra_MultiVector for an initial vector to start the solver.
  // Note:  This needs to have the same number of columns as the blocksize.
  Teuchos::RCP<Epetra_MultiVector> ivec = Teuchos::rcp( new Epetra_MultiVector(*Map, blockSize) );
  ivec->Random();

  // Create the eigenproblem.
  Teuchos::RCP<Anasazi::BasicEigenproblem<double, MV, OP> > eigenProblem =
    Teuchos::rcp( new Anasazi::BasicEigenproblem<double, MV, OP>(A, ivec) );

  // Inform the eigenproblem that the operator A is symmetric
  eigenProblem->setHermitian(true);

  // Set the number of eigenvalues requested
  eigenProblem->setNEV( nev );

  // Inform the eigenproblem that you are finishing passing it information
  bool boolret = eigenProblem->setProblem();
  TEUCHOS_TEST_FOR_EXCEPTION( boolret != true, std::runtime_error, 
			      "Anasazi::BasicEigenproblem::setProblem() returned an error.\n");
  
  // Create parameter list to pass into the solver manager
  Teuchos::ParameterList smPL;
  smPL.set( "Which", which );
  smPL.set( "Block Size", blockSize );
  smPL.set( "Num Blocks", numBlocks );
  smPL.set( "Maximum Restarts", maxRestarts );
  smPL.set( "Convergence Tolerance", tol );

  // Create the solver manager
  Anasazi::BlockDavidsonSolMgr<double, MV, OP> solverMan(eigenProblem, smPL);

  // Solve the problem
  Anasazi::ReturnType returnCode = solverMan.solve();

  // Get the eigenvalues and eigenvectors from the eigenproblem
  Anasazi::Eigensolution<double,MV> sol = eigenProblem->getSolution();
  evals = sol.Evals;
  evecs = sol.Evecs;

  // Compute residuals.
  std::vector<double> normR(sol.numVecs);
  if (sol.numVecs > 0) {
    Teuchos::SerialDenseMatrix<int,double> T(sol.numVecs, sol.numVecs);
    Epetra_MultiVector tempAevec( *Map, sol.numVecs );
    T.putScalar(0.0); 
    for (int i=0; i<sol.numVecs; i++) {
      T(i,i) = evals[i].realpart;
    }
    A->Apply( *evecs, tempAevec );
    MVT::MvTimesMatAddMv( -1.0, *evecs, T, 1.0, tempAevec );
    MVT::MvNorm( tempAevec, normR );
  }

  std::ostringstream os;
  os.setf(std::ios_base::right, std::ios_base::adjustfield);
  os<<"Solver manager returned " << (returnCode == Anasazi::Converged ? "converged." : "unconverged.") << std::endl;

  // Print the results
  if(bPrintResults) {
    os<<std::endl;
    os<<"------------------------------------------------------"<<std::endl;
    os<<std::setw(16)<<"Eigenvalue"
      <<std::setw(18)<<"Direct Residual"
      <<std::endl;
    os<<"------------------------------------------------------"<<std::endl;
    for (int i=0; i<sol.numVecs; i++) {
      os<<std::setw(16)<<evals[i].realpart
	<<std::setw(18)<<normR[i]/evals[i].realpart
	<<std::endl;
    }
    os<<"------------------------------------------------------"<<std::endl;
    std::cout << os.str() << std::endl;
  }

  double maxEigenvalue = evals[sol.numVecs-1].realpart;
  return (maxEigenvalue > Ecutoff);
}
    
void QCAD::GreensFunctionTunnelingSolver::
computeCurrentRange(const std::vector<double> Vds, double kbT, 
		    double Ecutoff_offset_from_Emax, std::vector<double>& resultingCurrent)
{
  for(std::size_t i = 0; i < Vds.size(); i++) {
    resultingCurrent[i] = computeCurrent(Vds[i], kbT, Ecutoff_offset_from_Emax);
  }
}
