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
  global_numbering(disc->getAPFGlobalNumbering()),
  esa(disc->getStateArrays().elemStateArrays),
  elemGIDws(disc->getElemGIDws()),
  cub_degree(disc->getFMDBMeshStruct()->cubatureDegree),
  pumi_disc(disc) {
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
AAdapt::SPRSizeField::setParams(double element_size, double err_bound,
			    const std::string state_var_name) {

  sv_name = state_var_name;
  rel_err = err_bound;
  std::vector<int> dims;
  esa[0][sv_name].dimensions(dims);
  num_qp = dims[1];

}

double AAdapt::SPRSizeField::getValue(ma::Entity* v) {
  return apf::getScalar(field,v,0);
}

void
AAdapt::SPRSizeField::getFieldFromStateVariable(apf::Field* eps) {
  global_numbering = pumi_disc->getAPFGlobalNumbering();
  apf::MeshIterator* it = mesh->begin(mesh->getDimension());
  apf::MeshEntity* e;
  while ((e = mesh->iterate(it))) {
    long elemID = apf::getNumber(global_numbering,apf::Node(e,0));
    int ws = elemGIDws[elemID].ws;
    int lid = elemGIDws[elemID].LID;
    for (int qp=0; qp < num_qp; qp++) {
      apf::Matrix3x3 value;
      for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
          value[i][j] = esa[ws][sv_name](lid,qp,i,j);
        }
      }
      apf::setMatrix(eps,e,qp,value);
    }
  }
  mesh->end(it);
}

void
AAdapt::SPRSizeField::computeErrorFromRecoveredGradients() {
  
  apf::Field* f = mesh->findField("solution");
  apf::Field* solution_gradient = apf::getGradIPField(f,"solution_gradient",cub_degree);
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
