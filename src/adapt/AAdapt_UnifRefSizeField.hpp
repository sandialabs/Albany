//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_UNIFREFSIZEFIELD_HPP
#define AADAPT_UNIFREFSIZEFIELD_HPP

#include "AlbPUMI_AbstractPUMIDiscretization.hpp"
#include "Epetra_Vector.h"
#include "AdaptTypes.h"
#include "MeshAdapt.h"
#include "Albany_StateManager.hpp"

namespace AAdapt {

class UnifRefSizeField {

  public:
    UnifRefSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc);

    ~UnifRefSizeField();

    int computeSizeField(pPart part, pSField field);

    void setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, 
		   double element_size, double err_bound,
		   const std::string state_var_name);
  
    void computeError();


  private:

    int getCurrentSize(pPart part, double& globMinSize, double& globMaxSize, double& globAvgSize);

    Teuchos::RCP<const Epetra_Comm> comm;
    AlbPUMI::AbstractPUMIDiscretization* disc;
    const Epetra_Vector* solution;
    const Epetra_Vector* ovlp_solution;

    double elem_size;

};

}

#endif

