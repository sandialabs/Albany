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
  comm(disc->getComm()) {

  mesh_struct = disc->getFMDBMeshStruct();
  mesh = mesh_struct->getMesh();

}


AAdapt::SPRSizeField::
~SPRSizeField() {
}

void
AAdapt::SPRSizeField::setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, double element_size, double relative_error) {

  solution = sol;
  ovlp_solution = ovlp_sol;
  elem_size = element_size;
  rel_err = relative_error;

}

void
AAdapt::SPRSizeField::computeError() {

  apf::Mesh* apf_mesh = apf::createMesh(mesh);
 
#if 0
 apf::writeVtkFiles("out-before-spr.vtk",apf_mesh);
#endif

  apf::Field* f = apf::createLagrangeField(apf_mesh,"f",apf::VECTOR,1);
  
  getFieldFromTag(f,mesh,"solution");
  apf::Field* solution_gradient = apf::getVectorGradField(f,"solution_gradient");
  apf::destroyField(f);
  
  apf::Field* sizef = apf::getSPRSizeField(solution_gradient,rel_err);
  apf::destroyField(solution_gradient);
  
  getTagFromField(sizef,mesh,"size");
  apf::destroyField(sizef);

  apf::writeVtkFiles("out-after-spr",apf_mesh);

  apf::destroyMesh(apf_mesh);

}

int
AAdapt::SPRSizeField::computeSizeField(pPart part, pSField field) {

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
AAdapt::SPRSizeField::getFieldFromTag(apf::Field* f, pMeshMdl mesh_pumi, const char* tag_name)
{
  pTag tag;
  int rc = PUMI_Mesh_FindTag(mesh_pumi,tag_name,tag);
  assert(rc == PUMI_SUCCESS);
  pPart part;
  PUMI_Mesh_GetPart(mesh_pumi,0,part);
  for (mPart::iterall it = part->beginall(0);
      it != part->endall(0); ++it)
  {
    pMeshEnt vertex = *it;
    apf::Vector3 v;
    double* unnecessary = &(v[0]);
    int really_unnecessary;
    rc = PUMI_MeshEnt_GetDblArrTag(mesh_pumi,vertex,tag,
        &unnecessary,&really_unnecessary);
    assert(rc == PUMI_SUCCESS);
    setVector(f,apf::castEntity(vertex),0,v);
  }
}

void
AAdapt::SPRSizeField::getTagFromField(apf::Field* f, pMeshMdl mesh_pumi, const char* tag_name)
{
  pTag tag;
  PUMI_Mesh_CreateTag(mesh_pumi,tag_name,PUMI_DBL,1,tag);
  pPart part;
  PUMI_Mesh_GetPart(mesh_pumi,0,part);
  for (mPart::iterall it = part->beginall(0);
      it != part->endall(0); ++it)
  {
    pMeshEnt vertex = *it;
    double v = getScalar(f,apf::castEntity(vertex),0);
    PUMI_MeshEnt_SetDblTag(mesh_pumi,vertex,tag,v);
  }
}
