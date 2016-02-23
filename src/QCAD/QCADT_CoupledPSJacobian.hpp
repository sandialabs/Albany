//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCADT_COUPLEDPSJACOBIAN_H
#define QCADT_COUPLEDPSJACOBIAN_H

#include <iostream>
#include "Teuchos_Comm.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"

#include "Albany_DataTypes.hpp"

#include "Teuchos_RCP.hpp"

#include "Thyra_BlockedLinearOpBase.hpp"
#include "Thyra_PhysicallyBlockedLinearOpBase.hpp"

//Forward Prototypes for utility functions
namespace QCADT {
  double n_prefactor(int numDims, int valleyDegeneracyFactor, double T, 
                     double length_unit_in_m, double energy_unit_in_eV, double effmass);
  double n_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV);
  double dn_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV);
  double compute_FDIntOneHalf(const double x);
  double compute_dFDIntOneHalf(const double x);
  double compute_FDIntMinusOneHalf(const double x);
  double compute_dFDIntMinusOneHalf(const double x);
}

namespace QCADT {

/** 
 *  \brief A class that evaluates the Jacobian of a
 *  QCAD coupled Poisson-Schrodinger problem
 */

class CoupledPSJacobian {
public:
  CoupledPSJacobian(int nEigenvals,
                    const Teuchos::RCP<const Tpetra_Map>& discretizationMap,
                    int dim, int valleyDegen, double temp,
                    double lengthUnitInMeters, double energyUnitInElectronVolts,
                    double effMass, double conductionBandOffset, 
                    Teuchos::RCP<Tpetra_Vector> neg_eigenvals,
                    Teuchos::RCP<const Tpetra_MultiVector> eigenvecs,
                    Teuchos::RCP<Teuchos_Comm const> const & commT);

  ~CoupledPSJacobian();

  Teuchos::RCP<Thyra::LinearOpBase<ST>> getThyraCoupledJacobian(Teuchos::RCP<Tpetra_CrsMatrix> Jac_Poisson,
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Jac_Schrodinger,
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Mass) const; 

private:

  Teuchos::RCP<Teuchos_Comm const> commT_;
  int num_models_; 
  int nEigenvals_; 
  Teuchos::RCP<Tpetra_MultiVector> dn_dPsi, dn_dEval;
  Teuchos::RCP<Tpetra_Vector> neg_eigenvalues;
  Teuchos::RCP<const Tpetra_MultiVector> psiVectors;
  Teuchos::RCP<const Tpetra_Map> discMap;

  // Values for computing the quantum density
  int numDims;
  int valleyDegenFactor;
  double temperature;
  double length_unit_in_m;
  double energy_unit_in_eV;
  double effmass;
  double offset_to_CB;

};

}
#endif  
