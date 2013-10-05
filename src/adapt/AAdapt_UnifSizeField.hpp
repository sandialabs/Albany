//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_UNIFSIZEFIELD_HPP
#define AADAPT_UNIFSIZEFIELD_HPP

#include "AlbPUMI_FMDBDiscretization.hpp"
#include "Epetra_Vector.h"
#include "AdaptTypes.h"
#include "MeshAdapt.h"

namespace AAdapt {

class UnifSizeField {

  public:
    UnifSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc);
    ~UnifSizeField();

    int computeSizeField(pPart part, pSField field);

    void setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, 
		   double element_size, double err_bound);

    void computeError();


  private:

    Teuchos::RCP<const Epetra_Comm> comm;
    const Epetra_Vector* solution;
    const Epetra_Vector* ovlp_solution;

    double elem_size;

    double rel_err;
};

}

#endif

