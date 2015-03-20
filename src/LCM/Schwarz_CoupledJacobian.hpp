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

#include "Thyra_BlockedLinearOpBase.hpp"
#include <Thyra_PhysicallyBlockedLinearOpBase.hpp>

namespace LCM {

/** 
 *  \brief A class that evaluates the Jacobian of a LCM coupled Schwarz Multiscale problem
 */

  class Schwarz_CoupledJacobian { 
  public:
    Schwarz_CoupledJacobian(const Teuchos::RCP<const Teuchos_Comm>& comm);

    ~Schwarz_CoupledJacobian();
     
    Teuchos::RCP<Thyra::LinearOpBase<ST> > getThyraCoupledJacobian(Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> >jacs) const; 

  private:

    Teuchos::RCP<const Teuchos_Comm> commT_;
  
    Teuchos::RCP<Thyra::PhysicallyBlockedLinearOpBase<ST> > blockedOp_; 


  };

}
#endif
