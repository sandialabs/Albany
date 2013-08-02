//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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

#include "Teuchos_RCP.hpp"

namespace QCAD {

/** 
 *  \brief An Epetra operator that evaluates the Jacobian of a QCAD coupled Poisson-Schrodinger problem
 */

  class CoupledPSJacobian : public Epetra_Operator {
  public:
    CoupledPSJacobian(int nEigenvals, 
		      const Teuchos::RCP<const Epetra_Map>& discMap, 
		      const Teuchos::RCP<const Epetra_Map>& fullPSMap,
		      const Teuchos::RCP<const Epetra_Comm>& comm);
    ~CoupledPSJacobian();

    //! Initialize the operator with everything needed to apply it
    void initialize(const Teuchos::RCP<Epetra_CrsMatrix>& poissonJac, const Teuchos::RCP<Epetra_CrsMatrix>& schrodingerJac, 
		    const Teuchos::RCP<Epetra_CrsMatrix>& massMatrix,
		    const Teuchos::RCP<Epetra_Vector>& eigenvals, const Teuchos::RCP<Epetra_MultiVector>& eigenvecs);

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
    Teuchos::RCP<const Epetra_Map> domainMap, rangeMap;
    Teuchos::RCP<const Epetra_Comm> myComm;
    bool bUseTranspose;
    bool bInitialized;

  Teuchos::RCP<Epetra_CrsMatrix> poissonJacobian, schrodingerJacobian;
  Teuchos::RCP<Epetra_CrsMatrix> overlapMatrix;
  Teuchos::RCP<Epetra_Vector> eigenvalues;
  Teuchos::RCP<Epetra_MultiVector> eigenvectors;
  };

}
#endif
