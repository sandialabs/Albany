//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MeshAdapt.hpp"

#include "AdaptUtil.h"
#include "PWLinearSField.h"

#include "Teuchos_TimeMonitor.hpp"

Albany::MeshAdapt::
MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_) :
    Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
    remeshFileIndex(1)
{

    disc = StateMgr.getDiscretization();

    fmdb_discretization = static_cast<Albany::FMDBDiscretization *>(disc.get());

    fmdbMeshStruct = fmdb_discretization->getFMDBMeshStruct();

    mesh = fmdbMeshStruct->getMesh();

    this->sizeFieldFunc = &Albany::MeshAdapt::setSizeField;

}

Albany::MeshAdapt::
~MeshAdapt()
{
}

bool
Albany::MeshAdapt::queryAdaptationCriteria(){

  int remesh_iter = params->get<int>("Remesh Step Number");

   if(iter == remesh_iter)
     return true;

  return false; 

}

bool
Albany::MeshAdapt::adaptMesh(){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::MeshAdapt method        " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  pPart pmesh;
  FMDB_Mesh_GetPart(mesh, 0, pmesh);

//  pSField sfield = new PWLinearSField(mesh);
  pSField sfield = new PWLsfield(pmesh);


  int num_iteration = 1;

  meshAdapt rdr(pmesh, sfield, 0, 1);  // snapping off; do refinement only
  rdr.run(num_iteration, 1, this->sizeFieldFunc);

  return true;

}

//! Transfer solution between meshes.
void
Albany::MeshAdapt::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::MeshAdapt::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");

/*
  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
*/

  validPL->set<int>("Remesh Step Number", 1, "Iteration step at which to remesh the problem");

  return validPL;
}

int 
Albany::MeshAdapt::setSizeField(pMesh mesh, pSField pSizeField, void *vp){

  double L = 10.0;
  double R = 0.8;

/*
  pMeshEnt node;
  double size, xyz[3];
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    FMDB_Vtx_GetCoord (node, &xyz);

    double circle= fabs(xyz[0] * xyz[0] +xyz[1] * xyz[1] + xyz[2] * xyz[2] - R*R);
    size = .1 * fabs(1. - exp (-circle*L)) + 1.e-3;
    pSizeField->setSize(node, size);
  }
  FMDB_PartEntIter_Del (node_it);
*/
  return 1;
}

#if 0
// *********************************
//  isotropic mesh size specification - This code has not been run for long since it has not been updated since it's made.
// Just please get an idea of how to set  isotropic mesh size field from this example.
// *********************************

int main(int argc, char* argv[])
{
  pMeshMdl mesh;
  ...
  pSField field=new metricField(mesh);
  meshAdapt rdr(mesh,field,0,1);  // snapping off; do refinement only
  rdr.run(num_iteration,1, setSizeField);
  ..
}

/* size field for cube.msh */
int setSizeField(pMesh mesh, pSField pSizeField)
{
  double L = 10.0;
  double R = 0.8;

  pMeshEnt node;
  double size, xyz[3];
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    FMDB_Vtx_GetCoord (node, &xyz);

    double circle= fabs(xyz[0] * xyz[0] +xyz[1] * xyz[1] + xyz[2] * xyz[2] - R*R);
    size = .1 * fabs(1. - exp (-circle*L)) + 1.e-3;
    pSizeField->setSize(node, size);
  }
  FMDB_PartEntIter_Del (node_it);
  return 1;
}

// *********************************
// anisotropic mesh size fields
// *********************************
int main(int argc, char* argv[])
{
  pMeshMdl mesh_instance;
  ...
  meshAdapt *rdr = new meshAdapt(mesh_instance, Analytical, 1);
  rdr->run(num_iteration,1, setSizeField);
  ...
}

int setSizeField(pMesh mesh, pSField field)
{
  double R0=1.; //.62;
  double L=3.;
  double center[]={1.0, 0.0, 0.0};
  double tol=0.01;
  double h[3], dirs[3][3], xyz[3], R, norm;
  R0=R0*R0;

  pVertex node;

  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    FMDB_Vtx_GetCoord (node, &xyz);
    R=dotProd(xyz,xyz);

    h[0] = .125 * fabs(1. - exp (-fabs(R-R0)*L)) + 0.00125;
    h[1] = .125;
    h[2] = .124;

    for(int i=0; i<3; i++)
      h[i] *= nSplit;

    norm=sqrt(R);
    if( norm>tol )
      {
        dirs[0][0]=xyz[0]/norm;
        dirs[0][1]=xyz[1]/norm;
        dirs[0][2]=xyz[2]/norm;
        if( xyz[0]*xyz[0] + xyz[1]*xyz[1] > tol*tol ) {
          dirs[1][0]=-1.0*xyz[1]/norm;
          dirs[1][1]=xyz[0]/norm;
          dirs[1][2]=0;
        } else {
          dirs[1][0]=-1.0*xyz[2]/norm;
          dirs[1][1]=0;
          dirs[1][2]=xyz[0]/norm;
        }
        crossProd(dirs[0],dirs[1],dirs[2]);
      }
    else
      {
        dirs[0][0]=1.0;
        dirs[0][1]=0.0;
        dirs[0][2]=0;
        dirs[1][0]=0.0;
        dirs[1][1]=1.0;
        dirs[1][2]=0;
        dirs[2][0]=0;
        dirs[1][2]=0;
        dirs[2][0]=0;
        dirs[2][1]=0;
        dirs[2][2]=1.0;
      }

    ((PWLsfield *)field)->setSize((pEntity)vt,dirs,h);
  }
  FMDB_PartEntIter_Del (node_it);

  double beta[]={2.5,2.5,2.5};
  ((PWLsfield *)field)->anisoSmooth(beta);
  return 1;
}
#endif
