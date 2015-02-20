//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LCM_SCHWARZ_JACOBIAN_H
#define LCM_SCHWARZ_JACOBIAN_H

#include <iostream>
#include "Teuchos_Comm.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"

#include "Albany_DataTypes.hpp"

#include "Teuchos_RCP.hpp"

namespace LCM {

/** 
 *  \brief An Epetra operator that evaluates the Jacobian of a LCM coupled Poisson-Schrodinger problem
 */

  class Schwarz_CoupledJacobian : public Tpetra_Operator {
  public:
    Schwarz_CoupledJacobian(int nEigenvals, 
		      const Teuchos::RCP<const Tpetra_Map>& discMap, 
		      const Teuchos::RCP<const Tpetra_Map>& fullPSMap,
		      const Teuchos::RCP<const Teuchos_Comm>& comm,
		      int dim, int valleyDegen, double temp,
		      double lengthUnitInMeters, double energyUnitInElectronVolts,
		      double effMass, double conductionBandOffset);
    ~Schwarz_CoupledJacobian();

    //! Initialize the operator with everything needed to apply it
    void initialize(const Teuchos::RCP<Tpetra_CrsMatrix>& poissonJac, const Teuchos::RCP<Tpetra_CrsMatrix>& schrodingerJac, 
		    const Teuchos::RCP<Tpetra_CrsMatrix>& massMatrix,
		    const Teuchos::RCP<Tpetra_Vector>& eigenvals, const Teuchos::RCP<const Tpetra_MultiVector>& eigenvecs);


    //! Returns the result of a Tpetra_Operator applied to a Tpetra_MultiVector X in Y.
    virtual int apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y) const;

    //! Returns the current UseTranspose setting.
    virtual bool hasTransposeApply() const { return bUseTranspose; }

    //! Returns the Tpetra_Map object associated with the domain of this operator.
    virtual Teuchos::RCP<const Tpetra_Map> getDomainMap() const { return domainMap; }

    //! Returns the Tpetra_Map object associated with the range of this operator.
    virtual Teuchos::RCP<const Tpetra_Map> getRangeMap() const { return rangeMap; }
    
  private:

    Teuchos::RCP<const Tpetra_Map> discMap;
    Teuchos::RCP<const Tpetra_Map> dist_evalMap, local_evalMap;
    Teuchos::RCP<const Tpetra_Map> domainMap, rangeMap;
    Teuchos::RCP<const Teuchos_Comm> myComm;
    Teuchos::RCP<const Tpetra_Import> eval_importer;
    bool bUseTranspose;
    bool bInitialized;

    Teuchos::RCP<Tpetra_CrsMatrix> poissonJacobian, schrodingerJacobian;
    Teuchos::RCP<Tpetra_CrsMatrix> massMatrix;
    Teuchos::RCP<Tpetra_Vector> neg_eigenvalues;
    Teuchos::RCP<const Tpetra_MultiVector> psiVectors;

    // Intermediate quantities precomputed in initialize() to speed up Apply()
    Teuchos::RCP<Tpetra_MultiVector> dn_dPsi, dn_dEval;
    Teuchos::RCP<Tpetra_MultiVector> M_Psi, MT_Psi;
    Teuchos::RCP<Tpetra_Vector> x_neg_evals_local;
    
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
