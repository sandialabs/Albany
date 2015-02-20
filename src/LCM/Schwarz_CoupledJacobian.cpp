//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_CoupledJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
//#include "Tpetra_LocalMap.h"

//Forward Prototypes for utility functions
double n_prefactor(int numDims, int valleyDegeneracyFactor, double T, double length_unit_in_m, double energy_unit_in_eV, double effmass);
double n_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV);
double dn_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV);
double compute_FDIntOneHalf(const double x);
double compute_dFDIntOneHalf(const double x);
double compute_FDIntMinusOneHalf(const double x);
double compute_dFDIntMinusOneHalf(const double x);

LCM::Schwarz_CoupledJacobian::Schwarz_CoupledJacobian(int nEigenvals, 
					   const Teuchos::RCP<const Tpetra_Map>& discretizationMap, 
					   const Teuchos::RCP<const Tpetra_Map>& fullPSMap, 
					   const Teuchos::RCP<const Teuchos_Comm>& comm,
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

LCM::Schwarz_CoupledJacobian::~Schwarz_CoupledJacobian()
{
}


//! Initialize the operator with everything needed to apply it
void LCM::Schwarz_CoupledJacobian::initialize(const Teuchos::RCP<Tpetra_CrsMatrix>& poissonJac, const Teuchos::RCP<Tpetra_CrsMatrix>& schrodingerJac, 
					 const Teuchos::RCP<Tpetra_CrsMatrix>& massMx,
					 const Teuchos::RCP<Tpetra_Vector>& neg_eigenvals, const Teuchos::RCP<const Tpetra_MultiVector>& eigenvecs)
{
  // Set member variables
/*  poissonJacobian = poissonJac;
  schrodingerJacobian = schrodingerJac;
  massMatrix = massMx;
  neg_eigenvalues = neg_eigenvals;
  psiVectors = eigenvecs;

  // Fill vectors that will be needed in Apply, but do so here so 
  //   it is only done once per evalModel call, not each time Apply is called

  int nEigenvalues = neg_eigenvalues->GlobalLength();
  int num_discMap_myEls = discMap->NumMyElements();

  // dn_dPsi : vectors of dn/dPsi[i] values
  dn_dPsi = Teuchos::rcp(new Tpetra_MultiVector( *psiVectors ));
  for(int i=0; i<nEigenvalues; i++) {
    (*dn_dPsi)(i)->Scale( n_prefactor(numDims, valleyDegenFactor, temperature, length_unit_in_m, energy_unit_in_eV, effmass) 
			  * 2 * n_weight_factor( -(*neg_eigenvalues)[i], numDims, temperature, energy_unit_in_eV), *((*psiVectors)(i)));
  }

  // dn_dEval : vectors of dn/dEval[i]
  dn_dEval = Teuchos::rcp(new Tpetra_MultiVector( *discMap, nEigenvalues ));
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
  M_Psi = Teuchos::rcp(new Tpetra_MultiVector( *discMap, nEigenvalues ));
  massMatrix->Multiply(false, *psiVectors, *M_Psi);

  // transpose(mass matrix) multiplied by Psi: MT*Psi
  MT_Psi = Teuchos::rcp(new Tpetra_MultiVector( *discMap, nEigenvalues ));
  massMatrix->Multiply(true, *psiVectors, *MT_Psi);

  // Prepare the distributed and local eigenvalue maps
  int my_nEigenvals = domainMap->NumMyElements() - num_discMap_myEls * (1+nEigenvalues);
  dist_evalMap  = Teuchos::rcp(new Tpetra_Map(nEigenvalues, my_nEigenvals, 0, *myComm));
  local_evalMap = Teuchos::rcp(new Tpetra_LocalMap(nEigenvalues, 0, *myComm));
  eval_importer = Teuchos::rcp(new Tpetra_Import(*local_evalMap, *dist_evalMap));
			       
  // Allocate temporary space for local storage of the eigenvalue part of the x vectors passed to Apply()
  x_neg_evals_local = Teuchos::rcp(new Tpetra_Vector(*local_evalMap));
*/
}


//! Returns the result of a Tpetra_Operator applied to a Tpetra_MultiVector X in Y.
int LCM::Schwarz_CoupledJacobian::apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y) const
{ 
  std::cout << "In LCM::Schwarz_CoupledJacobian::Apply! \n" << std::endl; 
  //int disc_nGlobalElements = discMap->NumGlobalElements();
  /*int disc_nMyElements = discMap->NumMyElements();
  int nEigenvals = neg_eigenvalues->GlobalLength();

  Teuchos::RCP<Tpetra_Vector> x_poisson, y_poisson;
  Teuchos::RCP<Tpetra_MultiVector> x_schrodinger, y_schrodinger;
  Teuchos::RCP<Tpetra_Vector> x_neg_evals, y_neg_evals;
  double *x_data, *y_data;

  Tpetra_Vector tempVec(*discMap); //we could allocate this in initialize instead of on stack...
  Tpetra_Vector tempVec2(*discMap); //we could allocate this in initialize instead of on stack...

  for(int i=0; i < X.NumVectors(); i++) {

    // Split X(i) and Y(i) into vector views for Poisson and Schrodinger parts
    if(X(i)->ExtractView(&x_data) != 0 || Y(i)->ExtractView(&y_data) != 0) 
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				 "Error!  LCM::Schwarz_CoupledJacobian -- cannot extract vector views");

    x_poisson = Teuchos::rcp(new Tpetra_Vector(::View, *discMap, &x_data[0]));
    x_schrodinger = Teuchos::rcp(new Tpetra_MultiVector(::View, *discMap, &x_data[disc_nMyElements], disc_nMyElements, nEigenvals));
    x_neg_evals = Teuchos::rcp(new Tpetra_Vector(::View, *dist_evalMap, &x_data[(1+nEigenvals)*disc_nMyElements]));

    y_poisson = Teuchos::rcp(new Tpetra_Vector(::View, *discMap, &y_data[0]));
    y_schrodinger = Teuchos::rcp(new Tpetra_MultiVector(::View, *discMap, &y_data[disc_nMyElements], disc_nMyElements, nEigenvals));
    y_neg_evals = Teuchos::rcp(new Tpetra_Vector(::View, *dist_evalMap, &y_data[(1+nEigenvals)*disc_nMyElements]));

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

  } */
  return 0;
}


