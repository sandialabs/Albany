//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AnasaziConfigDefs.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBlockDavidsonSolMgr.hpp"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"

#include "QCAD_GreensFunctionTunneling.hpp"
#include <fstream>

// Assume EcValues are in units of eV
// Assume effMass is in units of m_0 (electron rest mass)
// Assume ptSpacing is in units of microns (um)
// Assume pathLen is in unit of microns (um)

QCAD::GreensFunctionTunnelingSolver::
GreensFunctionTunnelingSolver(const Teuchos::RCP<std::vector<double> >& EcValues_,
            const Teuchos::RCP<std::vector<double> >& pathLen_, int nGFPts_,
			      double ptSpacing_, double effMass_, const Teuchos::RCP<const Epetra_Comm>& Comm_,
			      const std::string& outputFilename, bool bNeumannBC_)
{  
  Comm = Comm_;
  ptSpacing = ptSpacing_;
  nGFPts = nGFPts_; 
  effMass = effMass_;
  bNeumannBC = bNeumannBC_;

  // Retrieve Ec and pathLen for spline interpolation
  std::vector<double>  oldEcValues = (*EcValues_);
  std::vector<double>  oldPathLen = (*pathLen_);
  
  // Get size of oldEcValues
  int nOldPts = oldEcValues.size(); 
  
  // Call preSplineInterp() to obtain y2 
  std::vector<double> y2(nOldPts,0.0);
  prepSplineInterp(oldPathLen, oldEcValues, y2, nOldPts);  
  
  // Initialize the pointer, otherwise, seg. fault
  EcValues = Teuchos::rcp( new std::vector<double>(nGFPts) );
  
  // Keep the beginning and ending values the same
  (*EcValues)[0] = oldEcValues[0];
  (*EcValues)[nGFPts-1] = oldEcValues[nOldPts-1];
  
  // Splinely interpolate Ec values to the equally-spaced spatial grid defined by ptSpacing and nGFPts
  for (int i = 1; i < (nGFPts-1); i++)
  {
    double pathLength = double(i) * ptSpacing;
    (*EcValues)[i] = execSplineInterp(oldPathLen, oldEcValues, y2, nOldPts, pathLength);
  }

  // Output the spline interpolated Ec values
  if( outputFilename.length() > 0) 
  {
    std::cout << std::endl << "Append the spline interpolated Ec values to Output Filename..." << std::endl; 
    std::fstream out; 
    out.open(outputFilename.c_str(), std::fstream::out | std::fstream::app);
    out << std::endl << std::endl << "% Splinely interpolated Ec values on the equally-spaced spatial grid" << std::endl; 
    out << "% index  pathLength  Splined-Ec" << std::endl;
    for (int i = 0; i < nGFPts; i++)
    {
      double pathLength = double(i) * ptSpacing;
      out << i << " " << pathLength << " " << (*EcValues)[i] << std::endl; 
    }
    out.close();
  }
  
/* temporarily keep it

  matlabEvals.resize(nGFPts); 
  std::cout << "read Matlab-generated eigenvalues ..." << std::endl; 
  std::string outputFilename = "matlab_evals.dat";
  if( outputFilename.length() > 0) 
  {
    std::fstream out; 
    out.open(outputFilename.c_str(), std::fstream::in);

    for (int i = 0; i < (nGFPts-1); i++)
    { 
	    out >> matlabEvals[i];
	    // std::cout << "i=" << i << ", matlabEvals[i]=" << matlabEvals[i] << std::endl; 
    }
    out.close();
  }   
  std::cout << "finish reading from file ... " << std::endl;  */

}


QCAD::GreensFunctionTunnelingSolver::
~GreensFunctionTunnelingSolver()
{
}


double QCAD::GreensFunctionTunnelingSolver::
computeCurrent(double Vds, double kbT, double Ecutoff_offset_from_Emax, bool bUseAnasazi)
{
  const double hbar_1 = 6.582119e-16; // eV * s
  const double hbar_2 = 1.054572e-34; // J * s = kg * m^2 / s
  const double m_0 = 9.109382e-31;    // kg
  const double um = 1e-6;     // m
  const double min_a0 = 1e-4; // um, so == .1nm  HARDCODED MIN PT SPACING

  std::vector<double> evals;
  std::vector<double> evecBeginEls;
  std::vector<double> evecEndEls;

  // chemical potential energies of the left and right leads
  double muL = 0.; 
  double muR = -Vds;   // Vds can be >= 0 or < 0

  // Setup Energy bins for integration:  muL = 0, muR = -Vds, so set
  // Emin = min(Ec[0], Ec[nPts-1]-Vds)
  // Emax = max(muL, muR) + 20*kbT 
  // dE ~ kbT / 10
  int nPts    = EcValues->size();
  double Emax = std::max(muL, muR) + 20.*kbT;  // fermi dist. ~ exp(-40)=2.06e-9 small enough
  
  // Emin = min(Ec[0], Ec[nPts-1]-Vds), should work for arbitrary 1D Ec profile and Vds >= 0 or <0
  double Emin = std::min((*EcValues)[0], (*EcValues)[nPts-1]+muR); 
  if (Emin > Emax)
  {
    double tmp = Emin; 
    Emin = Emax; 
    Emax = tmp; 
  }

  // set up uniform energy spacing
  double dE = kbT / 10;
  int nEPts = int((Emax - Emin) / dE) + 1;
  dE = (Emax - Emin)/(nEPts-1);  // recalculate dE for given nEPts
  double a0 = ptSpacing;
  bool ret;

  double Ecutoff = Emax + Ecutoff_offset_from_Emax; // maximum eigenvalue needed
  int nEvecs;  // number of converged eigenvectors

  Teuchos::RCP<std::vector<double> > pEc = Teuchos::null;
  Teuchos::RCP<std::vector<double> > pLastEc = Teuchos::null;

  Map = Teuchos::rcp(new Epetra_Map(nPts, 0, *Comm));
  t0  = hbar_1 * hbar_2 /(2*effMass*m_0* pow(a0*um,2) ); // gives t0 in units of eV 
  std::cout << "Emin=" << Emin << ", Emax=" << Emax << ", Ecutoff=" << Ecutoff <<", t0=" << t0 << std::endl;    

  std::cout << "Doing Initial H-mx diagonalization for Vds = " << Vds << std::endl;

  if(bUseAnasazi) {
    Teuchos::RCP<Epetra_MultiVector> evecs;

    ret = doMatrixDiag_Anasazi(Vds, *EcValues, Ecutoff, evals, evecs);
    pEc = EcValues;
    std::cout << "  Diag w/ a0 = " << a0 << ", nPts = " << nPts << " give "
    	    << evals.size() << " eigenvalues, with Max Eval = " << evals[evals.size()-1] << std::endl; 
    	    // << "(need >= "<< Ecutoff << ")" << std::endl;
    
    // The following block is not called when a0 is small enough (<= 0.5 nm)
    while(ret == false && a0 > min_a0) // whether to perform auto-refinement -- add " && false" to disable
    {
      a0 /= 2.; nPts *= 2;  
      t0 = hbar_1 * hbar_2 /(2.*effMass*m_0* pow(a0*um,2.) );
      Map = Teuchos::rcp(new Epetra_Map(nPts, 0, *Comm));
    
      // Interpolate pLastEc onto pEc
      pLastEc = pEc;
      pEc = Teuchos::rcp(new std::vector<double>(nPts));
      for(int i = 0; i < nPts; i++) 
      {
        if(i%2 && i/2+1<nPts/2) (*pEc)[i] = ((*pLastEc)[i/2] + (*pLastEc)[i/2+1])/2.0;
        else (*pEc)[i] = (*pLastEc)[i/2];
      }
    
      // Setup and diagonalize H matrix
      ret = doMatrixDiag_Anasazi(Vds, *pEc, Ecutoff, evals, evecs);
      std::cout << "  Diag w/ a0 = " << a0 << ", nPts = " << nPts << ", t0 = " << t0 << " gives "
    	      << "Max Eval = " << evals[evals.size()-1]
    	      << "(need >= "<< Ecutoff << ")" << std::endl;
    }
    std::cout << "Done H-mx diagonalization, now broadcasting results" << std::endl;
    
    
    // Since all we need are the eigenvalues at beginning and end (index [0] and [nPts-1] ?)
    // then broadcast these values to all processors
    nEvecs = evecs->NumVectors();

    int GIDlist[2];
    GIDlist[0] = 0; // "beginning" index
    GIDlist[1] = nPts-1; // "ending" index
    
    //Get the IDs (rank) of the processors holding beginning and ending index
    std::vector<int> PIDlist(2), LIDlist(2);
    Map->RemoteIDList(2, GIDlist, &PIDlist[0], &LIDlist[0]);
    
    // Broadcast the beginning and ending elements of each eigenvector to all processors
    evecBeginEls.resize(nEvecs);
    evecEndEls.resize(nEvecs);
    
    if(Comm->MyPID() == PIDlist[0]) { // this proc owns beginning point
      for(int i=0; i<nEvecs; i++)
        evecBeginEls[i] = (*evecs)[i][LIDlist[0]]; // check that this is correct: Epetra_Vector [] operator takes *local* index?
    }
    Comm->Broadcast( &evecBeginEls[0], nEvecs, PIDlist[0] );
    
    if(Comm->MyPID() == PIDlist[1]) { // this proc owns ending point
      for(int i=0; i<nEvecs; i++)
        evecEndEls[i] = (*evecs)[i][LIDlist[1]]; // check that this is correct: Epetra_Vector [] operator takes *local* index?
    }
    Comm->Broadcast( &evecEndEls[0], nEvecs, PIDlist[1] );
  }

  else {  // use TQL2 tridiagonal routine.  Not parallel, all procs compute and store all evectors & evals

    std::vector<double> evecs;
    ret = doMatrixDiag_tql2(Vds, *EcValues, Ecutoff, evals, evecs);
    pEc = EcValues;
    std::cout << "  Diag w/ a0 = " << a0 << ", nPts = " << nPts << " give "
    	    << evals.size() << " eigenvalues, with Max Eval = " << evals[evals.size()-1] << std::endl; 
    	    // << "(need >= "<< Ecutoff << ")" << std::endl;
    
    // The following block is not called when a0 is small enough (<= 0.5 nm)
    while(ret == false && a0 > min_a0) // whether to perform auto-refinement -- add " && false" to disable
    {
      a0 /= 2.; nPts *= 2;  
      t0 = hbar_1 * hbar_2 /(2.*effMass*m_0* pow(a0*um,2.) );
    
      // Interpolate pLastEc onto pEc
      pLastEc = pEc;
      pEc = Teuchos::rcp(new std::vector<double>(nPts));
      for(int i = 0; i < nPts; i++) 
      {
        if(i%2 && i/2+1<nPts/2) (*pEc)[i] = ((*pLastEc)[i/2] + (*pLastEc)[i/2+1])/2.0;
        else (*pEc)[i] = (*pLastEc)[i/2];
      }
    
      // Setup and diagonalize H matrix
      ret = doMatrixDiag_tql2(Vds, *pEc, Ecutoff, evals, evecs);
      std::cout << "  Diag w/ a0 = " << a0 << ", nPts = " << nPts << ", t0 = " << t0 << " gives "
    	      << "Max Eval = " << evals[evals.size()-1] 
    	      << "(need >= "<< Ecutoff << ")" << std::endl;
    }
    std::cout << "Done H-mx diagonalization" << std::endl;

    // nEvecs = nPts;
    nEvecs = evals.size();  // include the case where only part of EVs are found, not equal to nPts
      
    evecBeginEls.resize(nEvecs);
    evecEndEls.resize(nEvecs);
    for(int i = 0; i < nEvecs; i++) { // assume evecs are in *columns* of evecs
      evecBeginEls[i] = evecs[i];
      evecEndEls[i] = evecs[nPts*(nPts-1) + i];
    }
  }

  // Potential energies, assumed to be constant, in each lead
  double VL = (*pEc)[0];
  double VR = (*pEc)[nPts-1] - Vds;

  std::cout << nEPts << " Energy Pts: " << Emin << " to " << Emax << " eV in steps of " << dE << std::endl;

  // Spread energy points evenly across processors
  Epetra_Map EnergyMap(nEPts, 0, *Comm);
  
  // Number of energy points on current processor. 
  int NumMyEnergyPts = EnergyMap.NumMyElements();

  // Put list of global elements on this processor into the user-provided array. 
  std::vector<int> MyGlobalEnergyPts(NumMyEnergyPts);
  EnergyMap.MyGlobalElements(&MyGlobalEnergyPts[0]);

  double I, Iloc = 0., x, y;
  double E, Gamma11, GammaNN, G011, G01N, G0NN, Tm;
  std::complex<double> Sigma11, SigmaNN, GR1N, p11, pNN;
  
  // add up contributions to energy integral from current processor
  // eigenvectors are assumed to be real
  for (int i = 0; i < NumMyEnergyPts; i++) {
    E = Emin + MyGlobalEnergyPts[i]*dE;

    x = (E-VL)*(t0-(E-VL)/4.);
    y = bNeumannBC ? 0. : t0;
    if( x >= 0. )
      Sigma11 = std::complex<double>( (E-VL)/2. - y, -sqrt(x) );
    else
      Sigma11 = std::complex<double>( (E-VL)/2. - y + sqrt(-x), 0.);

    x = (E-VR)*(t0-(E-VR)/4.);
    if( x >= 0. )
      SigmaNN = std::complex<double>( (E-VR)/2. - y, -sqrt(x) );
    else
      SigmaNN = std::complex<double>( (E-VR)/2. - y + sqrt(-x ), 0.);

    Gamma11 = -2.*Sigma11.imag();
    GammaNN = -2.*SigmaNN.imag();

    G011 = G01N = G0NN = 0.;
    for(int j = 0; j < nEvecs; j++) {
      G011 += evecBeginEls[j] * evecBeginEls[j] / (E - evals[j]);
      G01N += evecBeginEls[j] * evecEndEls[j] / (E - evals[j]);
      G0NN += evecEndEls[j] * evecEndEls[j] / (E - evals[j]);
    }

    p11 = Sigma11*G011; pNN = SigmaNN*G0NN; 
    GR1N = G01N / ((p11-1.0)*(pNN-1.0) - Sigma11*SigmaNN*G01N*G01N);
    Tm = Gamma11 * GammaNN * std::norm(GR1N);
    //std::cout << "DEBUG: brkdown = " << Gamma11 << " , " << GammaNN << " , " << 
    // std::norm(GR1N) << " , " << E << " , " << VL << " , " << VR << " , " << t0 << " , " << x << std::endl;

    Iloc += Tm * ( f0((E - muL)/kbT) - f0((E - muR)/kbT) ) * dE;
  }

  std::cout << "Energy Integral contrib from proc " << Comm->MyPID() << " = " << Iloc << " (" << NumMyEnergyPts << " energy pts)"<< std::endl;

  // add contributions from all processors
  Comm->SumAll(&Iloc, &I, 1);

  std::cout << "Total Energy Integral = " << I << " eV" << std::endl;
  
  const double h = 4.135668e-15;  // eV * s
  double q = 1.602177e-19;        // C
  
  // single-mode 1D ballistic current in [A]
  I = 2*q / h * I;  // since I was in units of eV, now I is in units of Amps (C/s)

  return I;
}


double QCAD::GreensFunctionTunnelingSolver::f0(double x) const
{
  return 1.0 / (1. + exp(x));
}


bool QCAD::GreensFunctionTunnelingSolver::
doMatrixDiag_tql2(double Vds, std::vector<double>& Ec, double Ecutoff,
		  std::vector<double>& evals, 
		  std::vector<double>& evecs)
{
  bool bPrintResults = false;
  int ierr;
  int nPts = Ec.size();
  std::vector<double> offDiag(nPts);  //off diagonal of symmetric tri-diagonal matrix (last nPts-1 els)

  /* We are building a matrix of block structure (left = DBC, right = NBC):
  
      | Ec+2t0  -t0                  |          | Ec+t0   -t0                  |
      | -t0    Ec+2t0  -t0           |	        | -t0    Ec+2t0  -t0           |
      |         -t0   ...            |	 OR     |         -t0   ...            |
      |                    ..    -t0 |	        |                    ..    -t0 |
      |                   -t0  Ec+2t0|	        |                   -t0  Ec+t0 |

   where the matrix has nPts rows and nPts columns
  */


  // Initialize diagonal of matrix in evals (since these will be the eigenvalues upon exit from tql2)
  //  and the off diagonal elements in the offDiag (tql2 only looks at the last nPts-1 els)
  evals.resize(nPts);
  for (int i = 0; i < nPts; i++) {
    double Ulin = -Vds * ((double)i) / (nPts-1);
    evals[i] = 2*t0 + Ec[i] + Ulin;
    offDiag[i] = -t0;
  }  
  if(bNeumannBC) {
    evals[0] -= t0;
    evals[nPts-1] -= t0;
  }
  
  // initialize evecs to n x n identity mx
  evecs.resize(nPts*nPts);
  for(int i=0; i<nPts; i++) {
    for(int j=0; j<nPts; j++) {
      evecs[nPts*i + j] = (i==j) ? 1.0 : 0.0;  
    } 
  }

  /*DEBUG
  std::cout << "Doing diag with: " << std::endl;
  std::cout << "Diag = "; printVector(evals, 1, evals.size());
  std::cout << "OffDiag = "; printVector(offDiag, 1, offDiag.size());
  std::cout << "Z = " << std::endl;  printVector(evecs,nPts,nPts);
  */
  
  // Diagonalize the matrix
  int nEvecs, max_iter = 1000;
  tql2(nPts, max_iter, evals, offDiag, evecs, ierr);

  /*DEBUG
  std::cout << "Results: (ret = " << ierr << ")" << std::endl;
  std::cout << "Evals = "; printVector(evals, 1, evals.size());
  std::cout << "Evecs = " << std::endl;  printVector(evecs,nPts,nPts);
  */


  // Truncate evals to the number of converged eigenvalues
  //  (note that they aren't necessarily ordered when ierr != 0)
  if(ierr != 0) 
  {
    nEvecs = ierr-1;		// because the ierr-th eigenvalue is not determined
    evals.resize(nEvecs);
    std::cout << "Only " << nEvecs << "eigenvalues out of total " << nPts << " are found !" << std::endl;
    
    // sort evals and evecs (note: evals.size() < nPts)
    sortTql2PartialResults(nPts, evals, evecs);
  }
  else nEvecs = nPts;  // find all evals and evecs

  // Print the results
  if(bPrintResults) {
    std::ostringstream os;
    os.setf(std::ios_base::right, std::ios_base::adjustfield);
    os<<"TQL2 solver returned " << ierr << std::endl;
    os<<std::endl;
    os<<"------------------------------------------------------"<<std::endl;
    os<<std::setw(16)<<"Eigenvalue"<<std::endl;
    os<<"------------------------------------------------------"<<std::endl;
    for (int i=0; i<nEvecs; i++) {
      os<<std::setw(16)<<evals[i]<<std::endl;
    }
    os<<"------------------------------------------------------"<<std::endl;
    std::cout << os.str() << std::endl; 
/*
    std::ostringstream os;
    os.setf(std::ios_base::right, std::ios_base::adjustfield);
    os << "Evals difference between Matlab and tql2 " << std::endl;
    os << std::endl;
    os << "------------------------------------------------------" << std::endl;
    os << "Index" << std::setw(16) << "  Eigenvalue difference" << std::endl;
    os << "------------------------------------------------------" << std::endl;
    
    for (int i = 0; i < nEvecs; i++)
    {
      if (std::abs(matlabEvals[i]-evals[i]) > 1e-3)
        os << i << std::setw(16) << matlabEvals[i]-evals[i] << std::endl;
    }
    os << "------------------------------------------------------" << std::endl;
    std::cout << os.str() << std::endl;  */

  }

  // double maxEigenvalue = evals[nPts-1];
  double maxEigenvalue = evals[nEvecs-1];  // include the (ierr!=0) case
  
  return (maxEigenvalue > Ecutoff);
}


bool QCAD::GreensFunctionTunnelingSolver::
doMatrixDiag_Anasazi(double Vds, std::vector<double>& Ec, double Ecutoff,
	     std::vector<double>& evals, 
	     Teuchos::RCP<Epetra_MultiVector>& evecs)
{
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;

  bool bPrintResults = false;
  std::vector<Anasazi::Value<double> > anasazi_evals;

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

  Values[0] = -t0; Values[1] = 2.*t0; Values[2] = -t0;
  if(bNeumannBC) {
    endValues[0] = -t0; endValues[1] =   t0; endValues[2] = -t0;  }
  else {
    endValues[0] = -t0; endValues[1] = 2.*t0; endValues[2] = -t0;  }

  
  for (int i = 0; i < nLocalEls; i++) {
    double Ulin = -Vds * ((double)myGlobalElements[i]) / (nPts-1);
    Values[1] = 2.*t0 + Ec[myGlobalElements[i]] + Ulin;
    
    if (bNeumannBC)  // NBC
      endValues[1] = t0 + Ec[myGlobalElements[i]] + Ulin;
    else             // DBC
      endValues[1] = 2.*t0 + Ec[myGlobalElements[i]] + Ulin;

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
  assert( info == 0 );
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
  anasazi_evals = sol.Evals;
  evecs = sol.Evecs;
  evals.resize(sol.numVecs);

  // Compute residuals.
  std::vector<double> normR(sol.numVecs);
  if (sol.numVecs > 0) {
    Teuchos::SerialDenseMatrix<int,double> T(sol.numVecs, sol.numVecs);
    Epetra_MultiVector tempAevec( *Map, sol.numVecs );
    T.putScalar(0.0); 
    for (int i=0; i<sol.numVecs; i++) {
      T(i,i) = anasazi_evals[i].realpart;
      evals[i] = anasazi_evals[i].realpart;
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
    for (int i=0; i<sol.numVecs; i++) 
    {
      os<<std::setw(16)<<anasazi_evals[i].realpart
        <<std::setw(18)<<normR[i]/anasazi_evals[i].realpart
        <<std::endl;
    }
    os<<"------------------------------------------------------"<<std::endl;
    std::cout << os.str() << std::endl;  

/*
    std::ostringstream os;
    os.setf(std::ios_base::right, std::ios_base::adjustfield);
    os << "Evals difference between Matlab and Anasazi " << std::endl;
    os << std::endl;
    os << "------------------------------------------------------" << std::endl;
    os << "Index" << std::setw(16) << "Eigenvalue difference" << std::endl;
    os << "------------------------------------------------------" << std::endl;
    
    for (int i = 0; i < sol.numVecs; i++)
    {
      if (std::abs(matlabEvals[i] - anasazi_evals[i].realpart) > 1e-3)
        os << i << std::setw(16) << matlabEvals[i] - anasazi_evals[i].realpart << std::endl;
    }
    os << "------------------------------------------------------" << std::endl; 
    std::cout << os.str() << std::endl;  */

  }

  double maxEigenvalue = anasazi_evals[sol.numVecs-1].realpart;
  return (maxEigenvalue > Ecutoff);
}

    
void QCAD::GreensFunctionTunnelingSolver::
computeCurrentRange(const std::vector<double> Vds, double kbT, 
	double Ecutoff_offset_from_Emax, std::vector<double>& resultingCurrent, bool bUseAnasazi)
{
  for(std::size_t i = 0; i < Vds.size(); i++) {
    resultingCurrent[i] = computeCurrent(Vds[i], kbT, Ecutoff_offset_from_Emax, bUseAnasazi);
  }
}


void QCAD::GreensFunctionTunnelingSolver::prepSplineInterp
  (const std::vector<double>& x, const std::vector<double>& y,  
   std::vector<double>& y2, const int& n)
{
  /* Given arrays x(0:n-1) and y(0:n-1) containing a tabulated function, i.e., yi = f(xi),
  with x0 < x1 < ... < x(n-1), this routine returns an array y2(0:n-1) of length n which 
  contains the second derivatives of the interpolating function at the tabulated
  points xi. Assume zero second derivatives at points 1 and n, that is, natural 
  spline boundary conditions. y2(0:n-1) is used in the execSplineInterp() function.
  */
  
  std::vector<double> u;
  u.resize(n-1);  

  // set the lower BC to be natural (2nd deriv. = 0)
  y2[0] = 0.;
  u[0] = 0.; 
  
  // decomposition loop of the tridiagonal algorithm, y2 and u are used for 
  // temporary storage of the decomposed factors
  double sig = 0.0, p = 0.0; 
  for (int i = 1; i < (n-1); ++i)
  {
    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    p = sig *y2[i-1] + 2.; 
    y2[i] = (sig -1.)/p; 
    u[i] = (6. *( (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]) )
               /(x[i+1]-x[i-1]) - sig*u[i-1]) / p; 
  }
  
  // set the upper BC to be natural; 
  double qn = 0., un = 0.; 
  
  // backsubstition of the tridiagonal algorithm to obtain y2
  y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2]+1.);   
  for (int i = (n-2); i >=0; i--)
  {
    y2[i] = y2[i]*y2[i+1] + u[i];
  }
  
  return; 
} 


double QCAD::GreensFunctionTunnelingSolver::execSplineInterp
  (const std::vector<double>& xa, const std::vector<double>& ya,  
   const std::vector<double>& y2a, const int& n, const double& x)
{
  /* Given the arrays xa(0:n-1) and ya(0:n-1) of length n, which tabulate a function 
  with the xai's in order, and given array y2a(0:n-1), which is the output from 
  prepSplineInterp() above, and given a value of x, this routine returns a 
  cubic-spline interpolated value y. 
  */

  int klo = 0; 
  int khi = n-1;
  
  // bisection searching
  while ( (khi-klo) > 1 )
  {
    int k = (khi+klo)/2;
    if (xa[k] > x) 
      khi = k;
    else
      klo = k; 
  }
  
  // klo and khi now bracket the input value of x
  
  double h = xa[khi] - xa[klo]; 
  if ( h < 1.e-100 ) 
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, 
			 "Bad xa input in execSplineInterp. The xa's must be distinct ! \n");

  // perform spline interpolation
  double a = (xa[khi] - x) / h; 
  double b = (x - xa[klo]) / h; 
  double c = (pow(a, 3.)-a) * pow(h, 2.) / 6.;
  double d = (pow(b, 3.)-b) * pow(h, 2.) / 6.;
  double y = a*ya[klo] + b*ya[khi] + c*y2a[klo] + d*y2a[khi];  
  
  return y; 
} 


//Utility routine for printing a stl vector of doubles.  this
// should probably be moved to a utility source file or replaced.
void printVector(const std::vector<double>& v, int m, int n)
{
  assert((int)v.size() == m*n);
  for(int i=0; i<m; i++) {
    for(int j=0; j<n; j++) {
      std::cout << v[n*i + j] << "  ";
    }
    std::cout << std::endl;
  }
}




// Helper function for tql2
double pythag(double a, double b)
{
  return sqrt(a*a + b*b);
}

// Helper function for tql2:
// returns the value of a with the sign of b
double sign(double a, double b)
{
  return (b >= 0) ? fabs(a) : -fabs(a);
}


// Shamelessly taken from http://www.netlib.org/seispack/tql2.f (converted to C)
int QCAD::GreensFunctionTunnelingSolver::tql2(int n, int max_iter,
					      std::vector<double>& d, 
					      std::vector<double>& e, 
					      std::vector<double>& z,
					      int& ierr)
{
  int i,j,k,l,m,ii,l1,l2;
  double c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2;

  /*
   *     this subroutine is a translation of the algol procedure tql2,
   *     num. math. 11, 293-306(1968) by bowdler, martin, reinsch, and
   *     wilkinson handbook for auto. comp., vol.ii-linear algebra, 227-240(1971).
   *
   *     this subroutine finds the eigenvalues and eigenvectors
   *     of a symmetric tridiagonal matrix by the ql method.
   *     the eigenvectors of a full symmetric matrix can also
   *     be found if  tred2  has been used to reduce this
   *     full matrix to tridiagonal form.
   *
   *     on input
   *
   *        n is the order of the matrix.
   *
   *        d contains the diagonal elements of the input matrix.
   *
   *        e contains the subdiagonal elements of the input matrix
   *		 c          in its last n-1 positions.  e(1) is arbitrary.
   *
   *        z contains the transformation matrix produced in the
   *          reduction by  tred2, if performed.  if the eigenvectors
   *		 c          of the tridiagonal matrix are desired, z must contain
   *          the identity matrix.
   *
   *      on output
   *
   *        d contains the eigenvalues in ascending order.  if an
   *		 c          error exit is made, the eigenvalues are correct but
   *		 c          unordered for indices 1,2,...,ierr-1.
   *
   *        e has been destroyed.
   *
   *        z contains orthonormal eigenvectors of the symmetric
   *		 c          tridiagonal (or full) matrix.  if an error exit is made,
   *          z contains the eigenvectors associated with the stored
   *          eigenvalues.
   *
   *        ierr is set to
   *		 c          zero       for normal return,
   *          j          if the j-th eigenvalue has not been
   *                     determined after max_iter iterations.
   *
   *		 c     calls pythag for  sqrt(a*a + b*b) .
   *
   *		 c     questions and comments should be directed to burton s. garbow,
   *		 c     mathematics and computer science div, argonne national laboratory
   *
   *     this version dated august 1983.
   *
   *     ------------------------------------------------------------------
   */

  ierr = 0;
  if (n == 1) return ierr;

  for(i=1; i<n; i++)
    e[i-1] = e[i];

  f = 0.0;
  tst1 = 0.0;
  e[n-1] = 0.0;

  for(l=0; l<n; l++) {
    j = 0;
    h = fabs(d[l]) + fabs(e[l]);
    if(tst1 < h) tst1 = h;  
    //  .......... look for small sub-diagonal element ..........

    for(m=l; m<n; m++) { 
      tst2 = tst1 + fabs(e[m]);
      if(tst2 == tst1) break;
      // .......... e[n-1] is always zero, so there is no exit
      //              through the bottom of the loop ..........
    }

    if(m != l) {
      do {
        if(j == max_iter) {
	  //     .......... set error -- no convergence to an
	  //                eigenvalue after maximum allowed iterations ..........
	  ierr = l; return ierr;
        }
        
        j = j + 1;
        //   .......... form shift ..........
        l1 = l + 1;
        l2 = l1 + 1;
        g = d[l];
        p = (d[l1] - g) / (2.0 * e[l]);
        r = pythag(p,1.0);
        d[l] = e[l] / (p + sign(r,p));
        d[l1] = e[l] * (p + sign(r,p));
        dl1 = d[l1];
        h = g - d[l];
        if (l2 <= n) {
          for(i=l2; i<n; i++) 
	    d[i] = d[i] - h;
        }
        f = f + h;
        
        //.......... ql transformation ..........
        p = d[m];
        c = 1.0;
        c2 = c;
        el1 = e[l1];
        s = 0.0;
        
        // .......... for i=m-1 step -1 until l do -- ..........
	for(i=m-1; i>=l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = pythag(p,e[i]);
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i+1] = h + s * (c * g + s * d[i]);
        
          // .......... form vector ..........
          for(k=0; k<n; k++) {
	    h = z[n*k + (i+1)];
	    z[n*k + (i+1)] = s * z[n*k + i] + c * h;
	    z[n*k + i] = c * z[n*k + i] - s * h;
	  }
        }
        
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
        tst2 = tst1 + fabs(e[l]);
      }	while(tst2 > tst1);
    }
    
    d[l] = d[l] + f;
  }

  // .......... order eigenvalues and eigenvectors ..........
  for(ii=1; ii<n; ii++) { 
    i = ii - 1;
    k = i;
    p = d[i];

    for(j=ii; j<n; j++) {
      if(d[j] >= p) continue;
      k = j;
      p = d[j];
    }

    if(k == i) continue;
    d[k] = d[i];
    d[i] = p;

    for(j=0; j<n; j++) {
      p = z[n*j + i];
      z[n*j + i] = z[n*j + k];
      z[n*j + k] = p;
    }
  }

  return ierr;
}


// Sort eigenvalues in ascending order and corresponding eigenvectors when tql2() 
// finds only part of evals and evecs, taken from tql2()
void QCAD::GreensFunctionTunnelingSolver::sortTql2PartialResults 
  (int n, std::vector<double>& d, std::vector<double>& z)
{
  int nEVs = d.size(); 
  int ii, i, k, j;
  double p; 
  
  // .......... order eigenvalues and eigenvectors ..........
  for(ii = 1; ii < nEVs; ii++) 
  { 
    i = ii - 1;
    k = i;
    p = d[i];

    for(j = ii; j < nEVs; j++) 
    {
      if(d[j] >= p) continue;
      k = j;
      p = d[j];
    }

    if(k == i) continue;
    d[k] = d[i];
    d[i] = p;

    // rearrange eigenvectors
    for(j = 0; j < n; j++) 
    {
      p = z[n*j + i];
      z[n*j + i] = z[n*j + k];
      z[n*j + k] = p;
    }
  }

  return; 
}
