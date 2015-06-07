//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
#include "QCADT_ImplicitPSJacobian.hpp"

namespace QCADT {

/** 
 *  \brief A class that evaluates the Jacobian of a
 *  QCAD coupled Poisson-Schrodinger problem
 */

class CoupledPSJacobian {
public:
  CoupledPSJacobian(int num_models, Teuchos::RCP<Teuchos_Comm const> const & commT);

  ~CoupledPSJacobian();

  Teuchos::RCP<Thyra::LinearOpBase<ST>> getThyraCoupledJacobian(Teuchos::RCP<Tpetra_CrsMatrix> Jac_Poisson = Teuchos::null,
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Jac_Schrodinger = Teuchos::null,
                                                  Teuchos::RCP<Tpetra_CrsMatrix> Mass = Teuchos::null,
                                                  Teuchos::RCP<Tpetra_Vector> neg_eigenvals = Teuchos::null, 
                                                  Teuchos::RCP<const Tpetra_MultiVector> eigenvecs = Teuchos::null) const; 

private:

  Teuchos::RCP<Teuchos_Comm const> commT_;
  int num_models_; 
};

}
#endif  
