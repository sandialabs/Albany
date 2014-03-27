//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_SPRSizeField.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "Epetra_Import.h"

#include "pumi.h"
#include "pumi_mesh.h"
#include "apfPUMI.h"
#include "apfSPR.h"

AAdapt::SPRSizeField::SPRSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc) :
  comm(disc->getComm()),
  mesh(disc->getFMDBMeshStruct()->apfMesh),
  esa(disc->getStateArrays().elemStateArrays),
  elemGIDws(disc->getElemGIDws()) {
}

AAdapt::SPRSizeField::
~SPRSizeField() {
}

void
AAdapt::SPRSizeField::computeError() {

  if ( sv_name.length() > 0 ) 
    computeErrorFromStateVariable(); 
  else 
    computeErrorFromRecoveredGradients();

}


void
AAdapt::SPRSizeField::setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, 
			    double element_size, double err_bound,
			    const std::string state_var_name) {

  solution = sol;
  ovlp_solution = ovlp_sol;
  sv_name = state_var_name;
  rel_err = err_bound;

}

double AAdapt::SPRSizeField::getValue(ma::Entity* v) {
  return apf::getScalar(field,v,0);
}

void
AAdapt::SPRSizeField::getFieldFromStateVariable(apf::Field* eps) {
  apf::MeshIterator* it = mesh->begin(3);
  apf::MeshEntity* e;
  while ((e = mesh->iterate(it))) {
    int elemID = FMDB_Ent_ID(reinterpret_cast<pMeshEnt>(e));
    int ws = elemGIDws[elemID].ws;
    int lid = elemGIDws[elemID].LID;
    apf::Matrix3x3 value;
    for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {
        value[i][j] = esa[ws][sv_name](lid,0,i,j);
      }
    }
    apf::setMatrix(eps,e,0,value);
  }
}

void
AAdapt::SPRSizeField::computeErrorFromRecoveredGradients() {
  
  apf::Field* f = mesh->findField("solution");
  apf::Field* solution_gradient = apf::getGradIPField(f,"solution_gradient",1);
  field = apf::getSPRSizeField(solution_gradient,rel_err);
  apf::destroyField(solution_gradient);

}


void
AAdapt::SPRSizeField::computeErrorFromStateVariable() {

  apf::Field* eps = apf::createIPField(mesh,"eps",apf::MATRIX,1);
  getFieldFromStateVariable(eps);
  field = apf::getSPRSizeField(eps,rel_err);
  apf::destroyField(eps);

}
