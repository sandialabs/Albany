//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MeshAdapt.hpp"

//#include "AdaptUtil.h"
#include "Albany_SizeField.hpp"

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

//    this->sizeFieldFunc = &Albany::MeshAdapt::setSizeField;
//    this->sizeFieldFunc = &(this->setSizeField);

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

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
    std::endl << "Error in Adaptation: calling Albany::MeshAdapt adaptMesh() without passing solution vector." << std::endl);

}

int 
setSizeField(pMesh mesh, pSField pSizeField, void *vp){

  Albany::SizeField *aSF = static_cast<Albany::SizeField*>(pSizeField);

  return aSF->computeSizeField();

}


bool
//Albany::MeshAdapt::adaptMesh(const Epetra_Vector& Solution, const Teuchos::RCP<Epetra_Import>& importer){
Albany::MeshAdapt::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::MeshAdapt method        " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

/*
The api to use displaced mesh for mesh adaptation will be
PUMI_Mesh_UseDisp (pMeshMdl mesh, pPag displacement_tag);

For our case, it will be PUMI_Mesh_UseDisp (*fmdbMeshStruct->getMesh(), *fmdbMeshStruct->solution_field);

In mesh adaptation, original coord+displacement will be used.
After mesh adaptation, the new displacement value will be available through solution_field. and new vertex coordinates will be new reference value.
*/

  pPart pmesh;
  FMDB_Mesh_GetPart(mesh, 0, pmesh);

//  pSField sfield = new PWLinearSField(mesh);
//  pSField sfield = new PWLsfield(pmesh);
  pSField sfield = new SizeField(pmesh, fmdb_discretization, sol, ovlp_sol);

  int num_iteration = 1;

  meshAdapt rdr(pmesh, sfield, 0, 1);  // snapping off; do refinement only
//  rdr.run(num_iteration, 1, this->sizeFieldFunc);
  rdr.run(num_iteration, 1, setSizeField);

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


