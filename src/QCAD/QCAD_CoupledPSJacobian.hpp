//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_COUPLEDPSJACOBIAN_H
#define QCAD_COUPLEDPSJACOBIAN_H

#include <iostream>
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_Operator.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"


#include "Teuchos_RCP.hpp"

//Forward Prototypes for utility functions
namespace QCAD {
  double n_prefactor(int numDims, int valleyDegeneracyFactor, 
                     double T, double length_unit_in_m, double energy_unit_in_eV, double effmass);
  double n_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV);
  double dn_weight_factor(double eigenvalue, int numDims, double T, double energy_unit_in_eV);
  double compute_FDIntOneHalf(const double x);
  double compute_dFDIntOneHalf(const double x);
  double compute_FDIntMinusOneHalf(const double x);
  double compute_dFDIntMinusOneHalf(const double x);
}

namespace QCAD {

/** 
 *  \brief An Epetra operator that evaluates the Jacobian of a QCAD coupled Poisson-Schrodinger problem
 */

  class CoupledPSJacobian : public Epetra_Operator {
  public:
    CoupledPSJacobian(int nEigenvals, 
		      const Teuchos::RCP<const Epetra_Map>& discMap, 
		      const Teuchos::RCP<const Epetra_Map>& fullPSMap,
		      const Teuchos::RCP<const Epetra_Comm>& comm,
		      int dim, int valleyDegen, double temp,
		      double lengthUnitInMeters, double energyUnitInElectronVolts,
		      double effMass, double conductionBandOffset);
    ~CoupledPSJacobian();

    //! Initialize the operator with everything needed to apply it
    void initialize(const Teuchos::RCP<Epetra_CrsMatrix>& poissonJac, const Teuchos::RCP<Epetra_CrsMatrix>& schrodingerJac, 
		    const Teuchos::RCP<Epetra_CrsMatrix>& massMatrix,
		    const Teuchos::RCP<Epetra_Vector>& eigenvals, const Teuchos::RCP<const Epetra_MultiVector>& eigenvecs);

    //! If set true, transpose of this operator will be applied.
    virtual int SetUseTranspose(bool UseTranspose) { bUseTranspose = UseTranspose; return 0; }; //Note: could return -1 if transpose isn't supported

    //! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
    virtual int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
    virtual int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Returns the infinity norm of the global matrix.
    virtual double NormInf() const { return 0.0; }

    //! Returns a character string describing the operator
    virtual const char * Label() const { return "Coupled Poisson-Schrodinger Jacobian"; }

    //! Returns the current UseTranspose setting.
    virtual bool UseTranspose() const { return bUseTranspose; }

    //! Returns true if this object can provide an approximate Inf-norm, false otherwise.
    virtual bool HasNormInf() const { return false; }

    //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
    virtual const Epetra_Comm & Comm() const { return *myComm; }

    //! Returns the Epetra_Map object associated with the domain of this operator.
    virtual const Epetra_Map & OperatorDomainMap() const { return *domainMap; }

    //! Returns the Epetra_Map object associated with the range of this operator.
    virtual const Epetra_Map & OperatorRangeMap() const { return *rangeMap; }
    
  private:

    Teuchos::RCP<const Epetra_Map> discMap;
    Teuchos::RCP<const Epetra_Map> dist_evalMap, local_evalMap;
    Teuchos::RCP<const Epetra_Map> domainMap, rangeMap;
    Teuchos::RCP<const Epetra_Comm> myComm;
    Teuchos::RCP<const Epetra_Import> eval_importer;
    bool bUseTranspose;
    bool bInitialized;

    Teuchos::RCP<Epetra_CrsMatrix> poissonJacobian, schrodingerJacobian;
    Teuchos::RCP<Epetra_CrsMatrix> massMatrix;
    Teuchos::RCP<Epetra_Vector> neg_eigenvalues;
    Teuchos::RCP<const Epetra_MultiVector> psiVectors;

    // Intermediate quantities precomputed in initialize() to speed up Apply()
    Teuchos::RCP<Epetra_MultiVector> dn_dPsi, dn_dEval;
    Teuchos::RCP<Epetra_MultiVector> M_Psi, MT_Psi;
    Teuchos::RCP<Epetra_Vector> x_neg_evals_local;
    
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
