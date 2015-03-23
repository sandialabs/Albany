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
 *  \brief A Tpetra operator that evaluates the Jacobian of a LCM coupled Schwarz Multiscale problem
 */

  class Schwarz_CoupledJacobian : public Tpetra_Operator {
  public:
    Schwarz_CoupledJacobian(Teuchos::Array<Teuchos::RCP<const Tpetra_Map> > disc_maps, 
		      Teuchos::RCP<const Tpetra_Map> coupled_disc_map,
		      const Teuchos::RCP<const Teuchos_Comm>& comm);

    ~Schwarz_CoupledJacobian();

    //! Initialize the operator with everything needed to apply it
    void initialize(Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> > jacs);


    //! Returns the result of a Tpetra_Operator applied to a Tpetra_MultiVector X in Y.
    virtual void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y,
                      Teuchos::ETransp mode = Teuchos::NO_TRANS,
                      ST alpha = Teuchos::ScalarTraits<ST>::one(),
                      ST beta = Teuchos::ScalarTraits<ST>::zero()) const;

    //! Returns the current UseTranspose setting.
    virtual bool hasTransposeApply() const { return b_use_transpose_; }

    //! Returns the Tpetra_Map object associated with the domain of this operator.
    virtual Teuchos::RCP<const Tpetra_Map> getDomainMap() const { return domain_map_; }

    //! Returns the Tpetra_Map object associated with the range of this operator.
    virtual Teuchos::RCP<const Tpetra_Map> getRangeMap() const { return range_map_; }
    
    
  private:

    Teuchos::Array<Teuchos::RCP<const Tpetra_Map> > disc_maps_;
    Teuchos::RCP<const Tpetra_Map> domain_map_, range_map_;
    Teuchos::RCP<const Teuchos_Comm> commT_;
    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> > jacs_;
    bool b_use_transpose_;
    bool b_initialized_;
    int n_models_; 


  };

}
#endif
