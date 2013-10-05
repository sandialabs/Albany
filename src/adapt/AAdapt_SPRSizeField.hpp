//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_SPRSIZEFIELD_HPP
#define AADAPT_SPRSIZEFIELD_HPP

#include "AlbPUMI_AbstractPUMIDiscretization.hpp"
#include "Epetra_Vector.h"
#include "AdaptTypes.h"
#include "MeshAdapt.h"
#include "apf.h"

namespace AAdapt {

class SPRSizeField {

  public:
    SPRSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc);
    ~SPRSizeField();

    void setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, double element_size, double relative_error);
    void computeError();

    int computeSizeField(pPart part, pSField field);

  private:
  
    Teuchos::RCP<AlbPUMI::FMDBMeshStruct> mesh_struct;
    pMeshMdl mesh;

    Teuchos::RCP<const Epetra_Comm> comm;
    const Epetra_Vector* solution;
    const Epetra_Vector* ovlp_solution;

    double elem_size;
    double rel_err;
  
    void getFieldFromTag(apf::Field* f, pMeshMdl mesh_pumi, const char* tag_name);
    void getTagFromField(apf::Field* f, pMeshMdl mesh_pumi, const char* tag_name);
};

}

#endif

