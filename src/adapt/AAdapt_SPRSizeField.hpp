//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_SPRSIZEFIELD_HPP
#define AADAPT_SPRSIZEFIELD_HPP

#include "AlbPUMI_FMDBDiscretization.hpp"
#include "Epetra_Vector.h"
#include "AdaptTypes.h"
#include "MeshAdapt.h"
#include "Albany_StateManager.hpp"
#include "apf.h"

namespace AAdapt {

class SPRSizeField {

  public:
    SPRSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc);
  
    ~SPRSizeField();

    int computeSizeField(pPart part, pSField field);

    void setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, 
		   double element_size, double err_bound,
		   const std::string state_var_name);

    void computeError();


  private:

    pMeshMdl mesh;
    Albany::StateArrayVec& esa;
    Albany::WsLIDList& elemGIDws;

    Teuchos::RCP<const Epetra_Comm> comm;
    const Epetra_Vector* solution;
    const Epetra_Vector* ovlp_solution;

    std::string sv_name;
    double rel_err;

    void getFieldFromTag(apf::Field* f, pMeshMdl mesh, const char* tag_name);
    void getTagFromField(apf::Field* f, pMeshMdl mesh, const char* tag_name);
    void getFieldFromStateVariable(apf::Field* eps, pMeshMdl mesh);
    void computeErrorFromRecoveredGradients();
    void computeErrorFromStateVariable();

};

}

#endif

