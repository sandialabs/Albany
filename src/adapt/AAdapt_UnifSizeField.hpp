//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_UNIFSIZEFIELD_HPP
#define AADAPT_UNIFSIZEFIELD_HPP

#include "AlbPUMI_FMDBDiscretization.hpp"
#include "Epetra_Vector.h"
#include <ma.h>
#include "Albany_StateManager.hpp"

namespace AAdapt {

class UnifSizeField : public ma::IsotropicFunction {

  public:
    UnifSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc);

    ~UnifSizeField();

    double getValue(ma::Entity* v);

    void setParams(
		   double element_size, double err_bound,
		   const std::string state_var_name);

    void computeError();


  private:

    Teuchos::RCP<const Epetra_Comm> comm;

    double elem_size;

};

}

#endif

