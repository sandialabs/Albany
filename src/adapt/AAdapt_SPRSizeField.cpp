//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: no Epetra if ALBANY_EPETRA_EXE turned off except Epetra_Comm.

#include "AAdapt_SPRSizeField.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"

#include <spr.h>
#include <apfShape.h>

AAdapt::SPRSizeField::SPRSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc) :
  comm(disc->getComm()),
  mesh(disc->getFMDBMeshStruct()->getMesh()),
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
AAdapt::SPRSizeField::copyInputFields()
{
  apf::FieldShape* fs = apf::getVoronoiShape(mesh->getDimension(), cub_degree);
  apf::Field* eps = apf::createField(mesh, "eps", apf::MATRIX, fs);
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

void AAdapt::SPRSizeField::freeSizeField()
{
  apf::destroyField(mesh->findField("size"));
}

void AAdapt::SPRSizeField::freeInputFields()
{
  apf::destroyField(mesh->findField("eps"));
}

void
AAdapt::SPRSizeField::computeErrorFromRecoveredGradients() {
  
  apf::Field* f = mesh->findField("solution");
  apf::Field* sol_grad = spr::getGradIPField(f,"sol_grad",cub_degree);
  field = spr::getSPRSizeField(sol_grad,rel_err);
  apf::destroyField(sol_grad);

}

void
AAdapt::SPRSizeField::computeErrorFromStateVariable() {

  apf::Field* eps = mesh->findField("eps");
  field = spr::getSPRSizeField(eps,rel_err);

}
