//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_COUPLEDPSJACOBIANT_H
#define QCAD_COUPLEDPSJACOBIANT_H

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
#include "QCAD_ImplicitPSJacobianT.hpp"

namespace QCAD {

/** 
 *  \brief A class that evaluates the Jacobian of a
 *  QCAD coupled Poisson-Schrodinger problem
 */

class CoupledPSJacobianT {
public:
  CoupledPSJacobianT(Teuchos::RCP<Teuchos_Comm const> const & commT);

  ~CoupledPSJacobianT();

  Teuchos::RCP<Thyra::LinearOpBase<ST>> getThyraCoupledJacobian(Teuchos::RCP<Tpetra_CrsMatrix> Jac_Poisson) const;

private:

  Teuchos::RCP<Teuchos_Comm const>
  commT_;
};

}
#endif  
