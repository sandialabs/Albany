//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_CoupledPSJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
#include "Epetra_LocalMap.h"


QCAD::CoupledPSJacobian::CoupledPSJacobian(int nEigenvals, 
					   const Teuchos::RCP<const Epetra_Map>& discretizationMap, 
					   const Teuchos::RCP<const Epetra_Map>& fullPSMap, 
					   const Teuchos::RCP<const Epetra_Comm>& comm,
					   int dim, int valleyDegen, double temp, 
					   double lengthUnitInMeters, double energyUnitInElectronVolts,
					   double effMass, double conductionBandOffset)
{
  discMap = discretizationMap;
  domainMap = rangeMap = fullPSMap;
  myComm = comm;
  bUseTranspose = false;
  bInitialized = false;

  numDims = dim;
  valleyDegenFactor = valleyDegen;
  temperature = temp;
  length_unit_in_m = lengthUnitInMeters;
  energy_unit_in_eV = energyUnitInElectronVolts;
  effmass = effMass;
  offset_to_CB = conductionBandOffset;
}

QCAD::CoupledPSJacobian::~CoupledPSJacobian()
{
}


//! Initialize the operator with everything needed to apply it
void QCAD::CoupledPSJacobian::initialize(const Teuchos::RCP<Epetra_CrsMatrix>& poissonJac, const Teuchos::RCP<Epetra_CrsMatrix>& schrodingerJac, 
					 const Teuchos::RCP<Epetra_CrsMatrix>& massMx,
					 const Teuchos::RCP<Epetra_Vector>& neg_eigenvals, const Teuchos::RCP<const Epetra_MultiVector>& eigenvecs)
{
  // Set member variables
  poissonJacobian = poissonJac;
  schrodingerJacobian = schrodingerJac;
  massMatrix = massMx;
  neg_eigenvalues = neg_eigenvals;
  psiVectors = eigenvecs;

  // Fill vectors that will be needed in Apply, but do so here so 
  //   it is only done once per evalModel call, not each time Apply is called

  int nEigenvalues = neg_eigenvalues->GlobalLength();
  int num_discMap_myEls = discMap->NumMyElements();

  // dn_dPsi : vectors of dn/dPsi[i] values
  dn_dPsi = Teuchos::rcp(new Epetra_MultiVector( *psiVectors ));
  for(int i=0; i<nEigenvalues; i++) {
    (*dn_dPsi)(i)->Scale( n_prefactor(numDims, valleyDegenFactor, temperature, length_unit_in_m, energy_unit_in_eV, effmass) 
			  * 2 * n_weight_factor( -(*neg_eigenvalues)[i], numDims, temperature, energy_unit_in_eV), *((*psiVectors)(i)));
  }

  // dn_dEval : vectors of dn/dEval[i]
  dn_dEval = Teuchos::rcp(new Epetra_MultiVector( *discMap, nEigenvalues ));
  for(int i=0; i<nEigenvalues; i++) {
    double prefactor = n_prefactor(numDims, valleyDegenFactor, temperature, length_unit_in_m, energy_unit_in_eV, effmass);
    double dweight = dn_weight_factor(-(*neg_eigenvalues)[i], numDims, temperature, energy_unit_in_eV);

    //DEBUG
    //double eps = 1e-7;
    //double dweight = (n_weight_factor( -(*neg_eigenvalues)[i] + eps, numDims, temperature, energy_unit_in_eV) - 
    //		      n_weight_factor( -(*neg_eigenvalues)[i], numDims, temperature, energy_unit_in_eV)) / eps; 
    //const double kbBoltz = 8.617343e-05;
    //std::cout << "DEBUG: dn_dEval["<<i<<"] dweight arg = " <<  (*neg_eigenvalues)[i]/(kbBoltz*temperature) << std::endl;  // in [eV]
    //std::cout << "DEBUG: dn_dEval["<<i<<"] factor = " <<  prefactor * dweight << std::endl;  // in [eV]
    //DEBUG

    for(int k=0; k<num_discMap_myEls; k++)
      ( *((*dn_dEval)(i)) )[k] =  prefactor * pow( (*((*psiVectors)(i)))[k], 2.0 ) * dweight;
    //(*dn_dEval)(i)->Print( std::cout << "DEBUG: dn_dEval["<<i<<"]:" << std::endl );
  }

  // mass matrix multiplied by Psi: M*Psi
  M_Psi = Teuchos::rcp(new Epetra_MultiVector( *discMap, nEigenvalues ));
  massMatrix->Multiply(false, *psiVectors, *M_Psi);

  // transpose(mass matrix) multiplied by Psi: MT*Psi
  MT_Psi = Teuchos::rcp(new Epetra_MultiVector( *discMap, nEigenvalues ));
  massMatrix->Multiply(true, *psiVectors, *MT_Psi);

  // Prepare the distributed and local eigenvalue maps
  int my_nEigenvals = domainMap->NumMyElements() - num_discMap_myEls * (1+nEigenvalues);
  dist_evalMap  = Teuchos::rcp(new Epetra_Map(nEigenvalues, my_nEigenvals, 0, *myComm));
  local_evalMap = Teuchos::rcp(new Epetra_LocalMap(nEigenvalues, 0, *myComm));
  eval_importer = Teuchos::rcp(new Epetra_Import(*local_evalMap, *dist_evalMap));
			       
  // Allocate temporary space for local storage of the eigenvalue part of the x vectors passed to Apply()
  x_neg_evals_local = Teuchos::rcp(new Epetra_Vector(*local_evalMap));

}


//! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
int QCAD::CoupledPSJacobian::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{ 
  //int disc_nGlobalElements = discMap->NumGlobalElements();
  int disc_nMyElements = discMap->NumMyElements();
  int nEigenvals = neg_eigenvalues->GlobalLength();

  Teuchos::RCP<Epetra_Vector> x_poisson, y_poisson;
  Teuchos::RCP<Epetra_MultiVector> x_schrodinger, y_schrodinger;
  Teuchos::RCP<Epetra_Vector> x_neg_evals, y_neg_evals;
  double *x_data, *y_data;

  Epetra_Vector tempVec(*discMap); //we could allocate this in initialize instead of on stack...
  Epetra_Vector tempVec2(*discMap); //we could allocate this in initialize instead of on stack...

  for(int i=0; i < X.NumVectors(); i++) {

    // Split X(i) and Y(i) into vector views for Poisson and Schrodinger parts
    if(X(i)->ExtractView(&x_data) != 0 || Y(i)->ExtractView(&y_data) != 0) 
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				 "Error!  QCAD::CoupledPSJacobian -- cannot extract vector views");

    x_poisson = Teuchos::rcp(new Epetra_Vector(::View, *discMap, &x_data[0]));
    x_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *discMap, &x_data[disc_nMyElements], disc_nMyElements, nEigenvals));
    x_neg_evals = Teuchos::rcp(new Epetra_Vector(::View, *dist_evalMap, &x_data[(1+nEigenvals)*disc_nMyElements]));

    y_poisson = Teuchos::rcp(new Epetra_Vector(::View, *discMap, &y_data[0]));
    y_schrodinger = Teuchos::rcp(new Epetra_MultiVector(::View, *discMap, &y_data[disc_nMyElements], disc_nMyElements, nEigenvals));
    y_neg_evals = Teuchos::rcp(new Epetra_Vector(::View, *dist_evalMap, &y_data[(1+nEigenvals)*disc_nMyElements]));

    // Communicate all the x_evals to every processor, since all parts of the mesh need them
    x_neg_evals_local->Import(*x_neg_evals, *eval_importer, Insert);


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
    
    // Scratch:  val = sum_ij x_i * M_ij * x_j
    //         dval/dx_k = sum_j!=k M_kj * x_j + sum_i!=k x_i * M_ik + 2* M_kk * x_k
    //                   = sum_i (M_ki + M_ik) * x_i
    //                   = sum_i (M + M^T)_ki * x_i  == k-th el of (M + M^T)*x
    //   So d(x*M*x)/dx = (M+M^T)*x in matrix form

    // Do multiplication block-wise

    // y_poisson =     
    poissonJacobian->Multiply(false, *x_poisson, *y_poisson); // y_poisson = Jac_poisson * x_poisson
    for(int i=0; i<nEigenvals; i++) {
      tempVec.Multiply(1.0, *((*dn_dPsi)(i)), *((*x_schrodinger)(i)), 0.0);  //tempVec = dn_dPsi[i] @ x_schrodinger[i]   (@ = el-wise product)
      massMatrix->Multiply(false, tempVec, tempVec2);  
      y_poisson->Update(1.0, tempVec2, 1.0);                                 // y_poisson += M * (dn_dPsi[i] @ x_schrodinger[i])

      tempVec.Update( -(*x_neg_evals_local)[i], *((*dn_dEval)(i)), 0.0); // tempVec = -dn_dEval[i] * scalar(x_neg_eval[i])
      massMatrix->Multiply(false, tempVec, tempVec2);
      y_poisson->Update( 1.0, tempVec2, 1.0); // y_poisson += M * (-dn_dEval[i] * scalar(x_neg_eval[i]))

      //OLD: y_poisson->Multiply(1.0, *((*dn_dPsi)(i)), *((*x_schrodinger)(i)), 1.0); // y_poisson += dn_dPsi[i] @ x_schrodinger[i]   (@ = el-wise product)
      //OLD: y_poisson->Update( -(*x_neg_evals_local)[i], *((*dn_dEval)(i)), 1.0); // y_poisson += -dn_dEval[i] * scalar(x_neg_eval[i])
    }

    // y_schrodinger =
    for(int j=0; j<nEigenvals; j++) {

      schrodingerJacobian->Multiply(false, *((*x_schrodinger)(j)), *((*y_schrodinger)(j)) ); // y_schrodinger[j] = H * x_schrodinger[j]
      massMatrix->Multiply(false, *((*x_schrodinger)(j)), tempVec );
      (*y_schrodinger)(j)->Update( (*neg_eigenvalues)[j], tempVec, 1.0); // y_schrodinger[j] += -eval[j] * M * x_schrodinger[j] 

      //tempVec.PutScalar(offset_to_CB);
      //tempVec.Update( -1.0, *((*psiVectors)(j)), 1.0);
      tempVec.Multiply( -1.0, *((*psiVectors)(j)), *x_poisson, 0.0); // tempVec = (-Psi[j]) @ x_poisson
      massMatrix->Multiply(false, tempVec, tempVec2);  
      (*y_schrodinger)(j)->Update( 1.0, tempVec2, 1.0); // y_schrodinger[j] += M * ( (-Psi[j]) @ x_poisson )

      //OLD: (*y_schrodinger)(j)->Multiply( -1.0, *((*psiVectors)(j)), *x_poisson, 1.0); // y_schrodinger[j] += (-Psi[j]) @ x_poisson
      (*y_schrodinger)(j)->Update( (*x_neg_evals_local)[j], *((*M_Psi)(j)), 1.0); // y_schrodinger[j] += M*Psi[j] * scalar(x_neg_eval[j])
    }
      
    // y_neg_evals =
    std::vector<double> y_neg_evals_local(nEigenvals);
    for(int j=0; j<nEigenvals; j++) {
      tempVec.Update(-1.0, *((*M_Psi)(j)), -1.0, *((*MT_Psi)(j)), 0.0); //tempVec = -(M*Psi[j] + MT*Psi[j]) = -(M+MT)*Psi[j]
      tempVec.Dot( *((*x_schrodinger)(j)), &y_neg_evals_local[j] );
    }
      // Fill elements of y_neg_evals that belong to this processor
    int my_nEigenvals = dist_evalMap->NumMyElements();
    std::vector<int> eval_global_elements(my_nEigenvals);
    dist_evalMap->MyGlobalElements(&eval_global_elements[0]);
    for(int i=0; i<my_nEigenvals; i++)
      (*y_neg_evals)[i] = y_neg_evals_local[eval_global_elements[i]];

  }
  return 0;
}

//! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
int QCAD::CoupledPSJacobian::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       std::endl << "QCAD::CoupledPSJacobian::ApplyInverse is not implemented." << std::endl);
    return 0;
}




// *****************************************************************************
// Quantum Density function
//
//  In all dimensions, it has the form:
//  n( x, Eval[*], Psi[*]) = prefactor * sum_k( |Psi[k](x)|^2 * weight_factor(Eval[k]) )
//
//  Thus (assuming Psi's are real):
//     dn/d{Psi[i](x)} = prefactor * 2 * Psi[i] * weight_factor(Eval[i])
//     dn/dEval[i]     = prefactor * Psi[i]^2 * d{weight_factor(Eval[i])}/dEval[i]
//
//  Below we define functions for prefactor, weight_factor, and d{weight_factor(E)}/dE
// *****************************************************************************

// CONSTANTS -- copied from QCAD_PoissonSource.cpp

// Boltzmann constant in [eV/K]
const double kbBoltz = 8.617343e-05;

// vacuum permittivity in [C/(V.cm)]
const double eps0 = 8.854187817e-12*0.01;

// electron elemental charge in [C]
const double eleQ = 1.602176487e-19; 

// vacuum electron mass in [kg]
const double m0 = 9.10938215e-31; 

// reduced planck constant in [J.s]
const double hbar = 1.054571628e-34; 

// pi constant (unitless)
const double pi = 3.141592654; 

// unit conversion factors
const double eVPerJ = 1.0/eleQ; 
const double cm2Perm2 = 1.0e4; 

// maximum allowed exponent in an exponential function (unitless)
const int MAX_EXPONENT = 100.0;


double QCAD::n_prefactor(int numDims, int valleyDegeneracyFactor, double T, double length_unit_in_m, double energy_unit_in_eV, double effmass)
{
  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm] (structure dimension in [um])
  double kbT = kbBoltz*T;  // in [eV]
  double eDenPrefactor;

  // unit factor applied to poisson source rhs1
  double poissonSourcePrefactor = (eleQ*X0*X0)/eps0 * 1.0/energy_unit_in_eV;

  // compute quantum electron density prefactor according to dimensionality
  switch (numDims)
  {
    case 1: // 1D wavefunction (1D confinement)
    {  
      // effmass = in-plane effective mass (y-z plane when the 1D wavefunc. is along x)
      // For Delta2-band (or valley), it is the transverse effective mass (0.19). 
        
      // 2D density of states in [#/(eV.cm^2)] where 2D is the unconfined y-z plane
      // dos2D below includes the spin degeneracy of 2
      double dos2D = effmass*m0/(pi*hbar*hbar*eVPerJ*cm2Perm2); 
        
      // subband-independent prefactor in calculating electron density
      // X0 is used to scale wavefunc. squared from [um^-1] or [nm^-1] to [cm^-1]
      eDenPrefactor = valleyDegeneracyFactor*dos2D*kbT/X0;
      break;
    }
    case 2: // 2D wavefunction (2D confinement)
    {
      // mUnconfined = effective mass in the unconfined direction (z dir. when the 2D wavefunc. is in x-y plane)
      // For Delta2-band and assume SiO2/Si interface parallel to [100] plane, mUnconfined=0.19. 
      double mUnconfined = effmass;
        
      // n1D below is a factor that is part of the line electron density 
      // in the unconfined dir. and includes spin degeneracy and in unit of [cm^-1]
      double n1D = sqrt(2.0*mUnconfined*m0*kbT/(pi*hbar*hbar*eVPerJ*cm2Perm2));
        
      // subband-independent prefactor in calculating electron density
      // X0^2 is used to scale wavefunc. squared from [um^-2] or [nm^-2] to [cm^-2]
      eDenPrefactor = valleyDegeneracyFactor*n1D/pow(X0,2.);
      break;
    }
    case 3: // 3D wavefunction (3D confinement)
    { 
      //degeneracy factor
      int spinDegeneracyFactor = 2;
      double degeneracyFactor = spinDegeneracyFactor * valleyDegeneracyFactor;
        
      // subband-independent prefactor in calculating electron density
      // X0^3 is used to scale wavefunc. squared from [um^-3] or [nm^-3] to [cm^-3]
      eDenPrefactor = degeneracyFactor/pow(X0,3.);
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      eDenPrefactor = 0.0;
      break; 
  }
  return eDenPrefactor * poissonSourcePrefactor;
}


//Note: assumes fermi level Ef == 0, so this is assumed to be true in the quantum region... perhaps pass in as a param later?
double QCAD::n_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV)
{
  double Ef = 0.0; //Fermi level of the quantum region
  double kbT = kbBoltz*T / energy_unit_in_eV;  // in [myV]

  switch (numDims)
  {
    case 1: // 1D wavefunction (1D confinement)
    {
      double tmpArg = (Ef-eigenvalue)/kbT;
      double logFunc; 
      if (tmpArg > MAX_EXPONENT)
	logFunc = tmpArg;  // exp(tmpArg) blows up for large positive tmpArg, leading to bad derivative
      else
	logFunc = log(1.0 + exp(tmpArg));
      return logFunc;
    }
    case 2: // 2D wavefunction (2D confinement)
    {
      double inArg = (Ef-eigenvalue)/kbT;
      return compute_FDIntMinusOneHalf(inArg);
    }
    case 3: // 3D wavefunction (3D confinement)
    { 
      double tmpArg = (eigenvalue-Ef)/kbT;
      double fermiFactor; 
        
      if (tmpArg > MAX_EXPONENT) 
	fermiFactor = exp(-tmpArg);  // use Boltzmann statistics for large positive tmpArg,
      else                           // as the Fermi statistics leads to bad derivative        
	fermiFactor = 1.0/( exp(tmpArg) + 1.0 ); 
      return fermiFactor;
    }      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      break; 
  }
  return 0.0;
}

double QCAD::dn_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV)
{
  double Ef = 0.0; //Fermi level of the quantum region
  double kbT = kbBoltz*T / energy_unit_in_eV;  // in [myV]

  switch (numDims)
  {
    case 1: // 1D wavefunction (1D confinement)
    {
      double tmpArg = (Ef-eigenvalue)/kbT;
      double dlogFunc; 
      if (tmpArg > MAX_EXPONENT)
	dlogFunc = -1.0/kbT;
      else
	dlogFunc = -1.0/(1.0 + exp(tmpArg)) * exp(tmpArg)/kbT;
      return dlogFunc;
    }
    case 2: // 2D wavefunction (2D confinement)
    {
      double inArg = (Ef-eigenvalue)/kbT;
      return compute_dFDIntMinusOneHalf(inArg) * -1.0/kbT;
    }
    case 3: // 3D wavefunction (3D confinement)
    { 
      double tmpArg = (eigenvalue-Ef)/kbT;
      double dfermiFactor; 
        
      if (tmpArg > MAX_EXPONENT) 
	dfermiFactor = exp(-tmpArg) * -1.0/kbT;  // use Boltzmann statistics for large positive tmpArg,
      else                                       // as the Fermi statistics leads to bad derivative        
	dfermiFactor = -1.0/pow( exp(tmpArg) + 1.0, 2) * exp(tmpArg)/kbT ; 
      return dfermiFactor;
    }      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid number of dimensions " << numDims << "!"<< std::endl);
      break; 
  }
  return 0.0;
}


double QCAD::compute_FDIntOneHalf(const double x)
{
   // Use the approximate 1/2 FD integral by D. Bednarczyk and J. Bednarczyk, 
   // "The approximation of the Fermi-Dirac integral F_{1/2}(x),"
   // Physics Letters A, vol.64, no.4, pp.409-410, 1978. The approximation 
   // has error < 4e-3 in the entire x range.  
   
   double fdInt; 
   if (x >= -50.0)
   {
     fdInt = pow(x,4.) + 50. + 33.6*x*(1.-0.68*exp(-0.17*pow((x+1.),2.0)));
     fdInt = pow((exp(-x) + (3./4.*sqrt(pi)) * pow(fdInt, -3./8.)),-1.0);
   }      
   else
     fdInt = exp(x); // for x<-50, the 1/2 FD integral is well approximated by exp(x)
     
   return fdInt;
}

double QCAD::compute_dFDIntOneHalf(const double x)
{
   // Use the approximate 1/2 FD integral by D. Bednarczyk and J. Bednarczyk, 
   // "The approximation of the Fermi-Dirac integral F_{1/2}(x),"
   // Physics Letters A, vol.64, no.4, pp.409-410, 1978. The approximation 
   // has error < 4e-3 in the entire x range.  
   
   double dfdInt, v,dv,u,du; 
   if (x >= -50.0)
   {
     v = pow(x,4.) + 50. + 33.6*x*(1.-0.68*exp(-0.17*pow((x+1.),2.0)));
     dv = 4.*pow(x,3.) + 33.6 * ( x*(-0.68*-0.17*2*(x+1.)*exp(-0.17*pow((x+1.),2.0))) + (1.-0.68*exp(-0.17*pow((x+1.),2.0))) );
     u = exp(-x) + (3./4.*sqrt(pi)) * pow(v, -3./8.);
     du = -exp(-x) + (3./4.*sqrt(pi)) * -3./8. * pow(v, -11./8.) * dv;
     dfdInt = -1.0 * pow( u ,-2.0) * du;
   }      
   else
     dfdInt = exp(x); // for x<-50, the 1/2 FD integral is well approximated by exp(x)
     
   return dfdInt;
}



double QCAD::compute_FDIntMinusOneHalf(const double x)
{
   // Use the approximate -1/2 FD integral by P. Van Halen and D. L. Pulfrey, 
   // "Accurate, short series approximations to Fermi-Dirac integrals of order 
   // -/2, 1/2, 1, 3/2, 2, 5/2, 3, and 7/2" J. Appl. Phys. 57, 5271 (1985) 
   // and its Erratum in J. Appl. Phys. 59, 2264 (1986). The approximation
   // has error < 1e-5 in the entire x range.  
   
   double fdInt; 
   double a1, a2, a3, a4, a5, a6, a7; 
   if (x <= 0.)  // eqn.(4) in the reference
   {
     a1 = 0.999909;  // Table I in Erratum
     a2 = 0.706781;
     a3 = 0.572752;
     a4 = 0.466318;
     a5 = 0.324511;
     a6 = 0.152889;
     a7 = 0.033673;
     fdInt = a1*exp(x)-a2*exp(2.*x)+a3*exp(3.*x)-a4*exp(4.*x)+a5*exp(5.*x)-a6*exp(6.*x)+a7*exp(7.*x);
   }
   else if (x >= 5.)  // eqn.(6) in Erratum
   {
     a1 = 1.12837;  // Table II in Erratum
     a2 = -0.470698;
     a3 = -0.453108;
     a4 = -228.975;
     a5 = 8303.50;
     a6 = -118124;
     a7 = 632895;
     fdInt = sqrt(x)*(a1+ a2/pow(x,2.)+ a3/pow(x,4.)+ a4/pow(x,6.)+ a5/pow(x,8.)+ a6/pow(x,10.)+ a7/pow(x,12.));
   }
   else if ((x > 0.) && (x <= 2.5))  // eqn.(7) in Erratum
   {
     double a8, a9;
     a1 = 0.604856;  // Table III in Erratum
     a2 = 0.380080;
     a3 = 0.059320;
     a4 = -0.014526;
     a5 = -0.004222;
     a6 = 0.001335;
     a7 = 0.000291;
     a8 = -0.000159;
     a9 = 0.000018;
     fdInt = a1+ a2*x+ a3*pow(x,2.)+ a4*pow(x,3.)+ a5*pow(x,4.)+ a6*pow(x,5.)+ a7*pow(x,6.)+ a8*pow(x,7.)+ a9*pow(x,8.);
   }
   else  // 2.5<x<5, eqn.(7) in Erratum
   {
     double a8;
     a1 = 0.638086;  // Table III in Erratum
     a2 = 0.292266;
     a3 = 0.159486;
     a4 = -0.077691;
     a5 = 0.018650;
     a6 = -0.002736;
     a7 = 0.000249;
     a8 = -0.000013;
     fdInt = a1+ a2*x+ a3*pow(x,2.)+ a4*pow(x,3.)+ a5*pow(x,4.)+ a6*pow(x,5.)+ a7*pow(x,6.)+ a8*pow(x,7.);
   }
   
   return fdInt;
}



double QCAD::compute_dFDIntMinusOneHalf(const double x)
{
   // Use the approximate -1/2 FD integral by P. Van Halen and D. L. Pulfrey, 
   // "Accurate, short series approximations to Fermi-Dirac integrals of order 
   // -/2, 1/2, 1, 3/2, 2, 5/2, 3, and 7/2" J. Appl. Phys. 57, 5271 (1985) 
   // and its Erratum in J. Appl. Phys. 59, 2264 (1986). The approximation
   // has error < 1e-5 in the entire x range.  
   
   double dfdInt; 
   double u,v,du,dv;
   double a1, a2, a3, a4, a5, a6, a7; 
   if (x <= 0.)  // eqn.(4) in the reference
   {
     a1 = 0.999909;  // Table I in Erratum
     a2 = 0.706781;
     a3 = 0.572752;
     a4 = 0.466318;
     a5 = 0.324511;
     a6 = 0.152889;
     a7 = 0.033673;
     dfdInt = a1*exp(x)-2*a2*exp(2.*x)+3*a3*exp(3.*x)-4*a4*exp(4.*x)+5*a5*exp(5.*x)-6*a6*exp(6.*x)+7*a7*exp(7.*x);
   }
   else if (x >= 5.)  // eqn.(6) in Erratum
   {
     a1 = 1.12837;  // Table II in Erratum
     a2 = -0.470698;
     a3 = -0.453108;
     a4 = -228.975;
     a5 = 8303.50;
     a6 = -118124;
     a7 = 632895;

     u = sqrt(x);
     du = 0.5/sqrt(x);
     v =  a1 + a2/pow(x,2.)+ a3/pow(x,4.)+ a4/pow(x,6.)+ a5/pow(x,8.)+ a6/pow(x,10.)+ a7/pow(x,12.);
     dv = -2*a2/pow(x,3.) -4*a3/pow(x,5.) -6*a4/pow(x,7.) -8*a5/pow(x,9.) -10*a6/pow(x,11) -12*a7/pow(x,13);
     dfdInt = u*dv + du*v;
   }
   else if ((x > 0.) && (x <= 2.5))  // eqn.(7) in Erratum
   {
     double a8, a9;
     a1 = 0.604856;  // Table III in Erratum
     a2 = 0.380080;
     a3 = 0.059320;
     a4 = -0.014526;
     a5 = -0.004222;
     a6 = 0.001335;
     a7 = 0.000291;
     a8 = -0.000159;
     a9 = 0.000018;
     dfdInt = a2+ 2*a3*x + 3*a4*pow(x,2.)+ 4*a5*pow(x,3.)+ 5*a6*pow(x,4.)+ 6*a7*pow(x,5.)+ 7*a8*pow(x,6.)+ 8*a9*pow(x,7.);
   }
   else  // 2.5<x<5, eqn.(7) in Erratum
   {
     double a8;
     a1 = 0.638086;  // Table III in Erratum
     a2 = 0.292266;
     a3 = 0.159486;
     a4 = -0.077691;
     a5 = 0.018650;
     a6 = -0.002736;
     a7 = 0.000249;
     a8 = -0.000013;
     dfdInt = a2 + 2*a3*x + 3*a4*pow(x,2.)+ 4*a5*pow(x,3.)+ 5*a6*pow(x,4.)+ 6*a7*pow(x,5.)+ 7*a8*pow(x,6.);
   }
   
   return dfdInt;
}
