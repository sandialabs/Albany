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
  std::vector<int> dims;
  esa[0][sv_name].dimensions(dims);
  num_qp = dims[1];
  cub_degree = getCubatureDegree(num_qp);

}

double AAdapt::SPRSizeField::getValue(ma::Entity* v) {
  return apf::getScalar(field,v,0);
}

int AAdapt::SPRSizeField::getCubatureDegree(int num_qp) {
  switch(num_qp) {
    case 1:
      return 1;
    case 4:
      return 2;
    case 5:
      return 3;
    default:
      fprintf(stderr,"Invalid cubature degree");
  }
}

void
AAdapt::SPRSizeField::getFieldFromStateVariable(apf::Field* eps) {
  apf::Numbering* en = mesh->findNumbering("apf_element_numbering");
  apf::MeshIterator* it = mesh->begin(mesh->getDimension());
  apf::MeshEntity* e;
  /* DAI: this does not account for more than one quadrature point per element */
  while ((e = mesh->iterate(it))) {
    int elemID = apf::getNumber(en,e,0,0);
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

  apf::Field* eps = apf::createIPField(mesh,"eps",apf::MATRIX,cub_degree);
  getFieldFromStateVariable(eps);
  field = apf::getSPRSizeField(eps,rel_err);
  apf::destroyField(eps);

}
