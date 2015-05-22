//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCADT_IMPLICITPSJACOBIAN_H
#define QCADT_IMPLICITPSJACOBIAN_H

#include <iostream>
#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Vector.hpp"

#include "Teuchos_RCP.hpp"
#include "Albany_DataTypes.hpp"

namespace QCADT {

/** 
 *  \brief A Tpetra operator that evaluates the Jacobian of a QCAD coupled Poisson-Schrodinger problem
 */

  class ImplicitPSJacobian : public Tpetra_Operator {
  public:
    ImplicitPSJacobian(int nEigenvals, 
		      const Teuchos::RCP<const Tpetra_Map>& discMap, 
		      const Teuchos::RCP<const Tpetra_Map>& fullPSMap,
		      const Teuchos::RCP<const Teuchos_Comm>& comm,
		      int dim, int valleyDegen, double temp,
		      double lengthUnitInMeters, double energyUnitInElectronVolts,
		      double effMass, double conductionBandOffset);
    ~ImplicitPSJacobian();

    //! Initialize the operator with everything needed to apply it
    void initialize(const Teuchos::RCP<Tpetra_CrsMatrix>& schrodingerJac, 
		    const Teuchos::RCP<Tpetra_CrsMatrix>& massMatrix,
		    const Teuchos::RCP<Tpetra_Vector>& eigenvals, const Teuchos::RCP<const Tpetra_MultiVector>& eigenvecs);

    //! If set true, transpose of this operator will be applied.
    virtual bool hasTransposeApply() const {return bUseTranspose;}

    //! Returns the result of a Tpetra_Operator applied to a Tpetra_MultiVector X in Y.
    virtual void  apply(
      Tpetra_MultiVector const & X,
      Tpetra_MultiVector & Y,
      Teuchos::ETransp mode = Teuchos::NO_TRANS,
      ST alpha = Teuchos::ScalarTraits<ST>::one(),
      ST beta = Teuchos::ScalarTraits<ST>::zero()) const;

    //! Returns the infinity norm of the global matrix.
    virtual double NormInf() const { return 0.0; }

    //! Returns a character string describing the operator
    virtual const char * Label() const { return "Coupled Poisson-Schrodinger Jacobian"; }

    //! Returns true if this object can provide an approximate Inf-norm, false otherwise.
    virtual bool HasNormInf() const { return false; }

    //! Returns the Tpetra_Map object associated with the domain of this operator.
    virtual Teuchos::RCP<Tpetra_Map const> getDomainMap() const {return domainMap;}
  
    /// Returns the Tpetra_Map object associated with the range of this operator.
    virtual Teuchos::RCP<Tpetra_Map const> getRangeMap() const {return rangeMap;}


    
  private:

    Teuchos::RCP<const Tpetra_Map> discMap;
    Teuchos::RCP<const Tpetra_Map> dist_evalMap, local_evalMap;
    Teuchos::RCP<const Tpetra_Map> domainMap, rangeMap;
    Teuchos::RCP<const Teuchos_Comm> myComm;
    Teuchos::RCP<const Tpetra_Import> eval_importer;
    bool bUseTranspose;

    Teuchos::RCP<Tpetra_CrsMatrix> schrodingerJacobian;
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
