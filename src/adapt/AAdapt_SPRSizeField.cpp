//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_SPRSizeField.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "Epetra_Import.h"
#include "PWLinearSField.h"

#include "pumi.h"
#include "pumi_mesh.h"
#include "apfPUMI.h"
#include "apfSPR.h"

AAdapt::SPRSizeField::SPRSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc) :
  comm(disc->getComm()),
  mesh(disc->getFMDBMeshStruct()->getMesh()),
  sa(disc->getStateArrays()),
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

int AAdapt::SPRSizeField::computeSizeField(pPart part, pSField field) {

  pMeshEnt vtx;
  double h[3], dirs[3][3], xyz[3];
  
  pPartEntIter vtx_iter;
  FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, vtx_iter);
  
  pTag size_tag;
  FMDB_Mesh_FindTag(mesh, "size", size_tag);
  
  while(FMDB_PartEntIter_GetNext(vtx_iter, vtx) == SCUtil_SUCCESS) {
    
    double size;
    FMDB_Ent_GetDblTag(mesh,vtx,size_tag,&size);
    
    h[0] = size;
    h[1] = size;
    h[2] = size;

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

  double beta[] = {1.25, 1.25, 1.25};
  ((PWLsfield*)field)->anisoSmooth(beta);

  return 1;

}

void 
AAdapt::SPRSizeField::getFieldFromTag(apf::Field* f, pMeshMdl mesh, const char* tag_name) {
  pTag tag;
  int rc = PUMI_Mesh_FindTag(mesh,tag_name,tag);
  assert(rc == PUMI_SUCCESS);
  pPart part;
  PUMI_Mesh_GetPart(mesh,0,part);
  for (mPart::iterall it = part->beginall(0);
      it != part->endall(0); ++it) {
    pMeshEnt vertex = *it;
    apf::Vector3 v;
    double* unnecessary = &(v[0]);
    int really_unnecessary;
    rc = PUMI_MeshEnt_GetDblArrTag(mesh,vertex,tag,
        &unnecessary,&really_unnecessary);
    assert(rc == PUMI_SUCCESS);
    setVector(f,apf::castEntity(vertex),0,v);
  }
}

void
AAdapt::SPRSizeField::getTagFromField(apf::Field* f, pMeshMdl mesh, const char* tag_name) {
  pTag tag;
  PUMI_Mesh_CreateTag(mesh,tag_name,PUMI_DBL,1,tag);
  pPart part;
  PUMI_Mesh_GetPart(mesh,0,part);
  for (mPart::iterall it = part->beginall(0);
      it != part->endall(0); ++it) {
    pMeshEnt vertex = *it;
    double v = getScalar(f,apf::castEntity(vertex),0);
    PUMI_MeshEnt_SetDblTag(mesh,vertex,tag,v);
  }
}

void
AAdapt::SPRSizeField::getFieldFromStateVariable(apf::Field* eps, pMeshMdl mesh) {

  pPart part;
  FMDB_Mesh_GetPart(mesh, 0, part);
  pMeshEnt elem;
  pPartEntIter elem_iter;
  FMDB_PartEntIter_Init(part, FMDB_REGION, FMDB_ALLTOPO, elem_iter);
  while(FMDB_PartEntIter_GetNext(elem_iter, elem) == SCUtil_SUCCESS) {
    int ws = elemGIDws[FMDB_Ent_ID(elem)].ws;
    int lid = elemGIDws[FMDB_Ent_ID(elem)].LID;
    apf::Matrix3x3 value;
    for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {
	value[i][j] = sa[ws][sv_name](lid,0,i,j);
      }
    }
    setMatrix(eps,apf::castEntity(elem),0,value);
  }
}

void
AAdapt::SPRSizeField::computeErrorFromRecoveredGradients() {
  
  apf::Mesh* apf_mesh = apf::createMesh(mesh);
  apf::Field* f = apf::createLagrangeField(apf_mesh,"f",apf::VECTOR,1);
  getFieldFromTag(f,mesh,"solution");
  apf::Field* solution_gradient = apf::getVectorGradIPField(f,"solution_gradient",1);
  apf::destroyField(f);
  apf::Field* sizef = apf::getSPRSizeField(solution_gradient,rel_err);
  apf::destroyField(solution_gradient);
  getTagFromField(sizef,mesh,"size");
  apf::destroyField(sizef);
  apf::writeVtkFiles("out-after-spr",apf_mesh);
  apf::destroyMesh(apf_mesh);
  
}


void
AAdapt::SPRSizeField::computeErrorFromStateVariable() {

  apf::Mesh* apf_mesh = apf::createMesh(mesh);
  apf::Field* eps = apf::createIPField(apf_mesh,"eps",apf::MATRIX,1);
  getFieldFromStateVariable(eps,mesh);
  apf::Field* sizef = apf::getSPRSizeField(eps,rel_err);
  apf::destroyField(eps);
  getTagFromField(sizef,mesh,"size");
  apf::destroyField(sizef);
  apf::writeVtkFiles("out-after-spr",apf_mesh);
  apf::destroyMesh(apf_mesh);
    
}
