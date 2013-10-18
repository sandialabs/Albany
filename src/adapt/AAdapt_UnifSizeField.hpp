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
#include "Albany_StateManager.hpp"

namespace AAdapt {

class UnifSizeField {

  public:
    UnifSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc,
		  Albany::StateManager& state_manager);

    ~UnifSizeField();

    int computeSizeField(pPart part, pSField field);

    void setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, 
		   double element_size, double err_bound,
		   const std::string state_var_name);

    void computeError();


  private:

    Albany::StateManager& state_mgr;
    Teuchos::RCP<const Epetra_Comm> comm;
    const Epetra_Vector* solution;
    const Epetra_Vector* ovlp_solution;

    double elem_size;

};

}

#endif

