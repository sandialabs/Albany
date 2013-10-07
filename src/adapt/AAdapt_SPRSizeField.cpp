//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_SPRSizeField.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "Epetra_Import.h"
#include "PWLinearSField.h"

AAdapt::SPRSizeField::SPRSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc,
			   Albany::StateManager& state_manager) :
  comm(disc->getComm()), 
  state_mgr(state_manager) {
}

AAdapt::SPRSizeField::
~SPRSizeField() {
}

void
AAdapt::SPRSizeField::computeError() {
}


void
AAdapt::SPRSizeField::setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, 
			    double element_size, double err_bound,
			    const std::string state_var_name) {

  solution = sol;
  ovlp_solution = ovlp_sol;
  elem_size = element_size;

}

int AAdapt::SPRSizeField::computeSizeField(pPart part, pSField field) {

  pMeshEnt vtx;
  double h[3], dirs[3][3], xyz[3];

  std::cout << elem_size << std::endl;

  pPartEntIter vtx_iter;
  FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, vtx_iter);

  while(FMDB_PartEntIter_GetNext(vtx_iter, vtx) == SCUtil_SUCCESS) {
    h[0] = elem_size;
    h[1] = elem_size;
    h[2] = elem_size;

    dirs[0][0] = 1.0;
    dirs[0][1] = 0.;
    dirs[0][2] = 0.;
    dirs[1][0] = 0.;
    dirs[1][1] = 1.0;
    dirs[1][2] = 0.;
    dirs[2][0] = 0.;
    dirs[2][1] = 0.;
    dirs[2][2] = 1.0;

    ((PWLsfield*)field)->setSize(vtx, dirs, h);
  }

  FMDB_PartEntIter_Del(vtx_iter);

  double beta[] = {1.5, 1.5, 1.5};
  ((PWLsfield*)field)->anisoSmooth(beta);

  return 1;
}

