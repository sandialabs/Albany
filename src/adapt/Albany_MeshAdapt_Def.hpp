//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MeshAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"

template<class SizeField>
Teuchos::RCP<SizeField> Albany::MeshAdapt<SizeField>::szField = Teuchos::null;

template<class SizeField>
Albany::MeshAdapt<SizeField>::
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

    szField = Teuchos::rcp(new SizeField(fmdb_discretization));

}

template<class SizeField>
Albany::MeshAdapt<SizeField>::
~MeshAdapt()
{
}

template<class SizeField>
bool
Albany::MeshAdapt<SizeField>::queryAdaptationCriteria(){

  int remesh_iter = params->get<int>("Remesh Step Number");

   if(iter == remesh_iter)
     return true;

  return false; 

}

template<class SizeField>
bool
Albany::MeshAdapt<SizeField>::adaptMesh(){

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
    std::endl << "Error in Adaptation: calling Albany::MeshAdapt adaptMesh() without passing solution vector." << std::endl);

}

template<class SizeField>
int 
Albany::MeshAdapt<SizeField>::setSizeField(pPart part, pSField pSizeField, void *vp){

  return szField->computeSizeField(part, pSizeField);

}

/*
int sizefield(pPart part, pSField field, void *)
{
  uniform (part, field);
//  uniformRefSzFld(part, field, NULL);
  return 1;
}
*/

template<class SizeField>
bool
//Albany::MeshAdapt::adaptMesh(const Epetra_Vector& Solution, const Teuchos::RCP<Epetra_Import>& importer){
Albany::MeshAdapt<SizeField>::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::MeshAdapt method        " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  // display # entities before adaptation
  FMDB_Mesh_DspNumEnt (mesh);
  szField->setParams(&sol, &ovlp_sol,
              params->get<double>("Target Element Size", 0.1));

/*
The api to use displaced mesh for mesh adaptation will be
PUMI_Mesh_UseDisp (pMeshMdl mesh, pPag displacement_tag);

For our case, it will be PUMI_Mesh_UseDisp (*fmdbMeshStruct->getMesh(), *fmdbMeshStruct->solution_field);

In mesh adaptation, original coord+displacement will be used.
After mesh adaptation, the new displacement value will be available through solution_field. and new vertex coordinates will be new reference value.
*/


  int num_iteration = params->get<int>("Max Number of Mesh Adapt Iterations", 1);

  // Do basic uniform refinement
  /** Type of the size field: 
      - Application - the size field will be provided by the application (default).
      - TagDriven - tag driven size field. 
      - Analytical - analytical size field.  */
  /** Type of model: 
      - 0 - no model (not snap), 1 - mesh model (always snap), 2 - solid model (always snap)
  */
  meshAdapt *rdr = new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 2 );

  /** void meshAdapt::run(int niter,    // specify the maximum number of iterations 
		    int flag,           // indicate if a size field function call is available
		    adaptSFunc sizefd)  // the size field function call  */
//  rdr->run (num_iteration, 1, sizefield);
  rdr->run (num_iteration, 1, this->setSizeField);

  // dump the adapted mesh for visualization
  FMDB_Mesh_WriteToFile (mesh, "adapted_mesh_out.vtk",  (SCUTIL_CommSize()>1?1:0));
  // display # entities after adaptation
  FMDB_Mesh_DspNumEnt (mesh);
  // check the validity of adapted mesh
  int isValid=0;
  FMDB_Mesh_Verify(mesh, &isValid);

  delete rdr;

#if 0

  pPart pmesh;
  FMDB_Mesh_GetPart(mesh, 0, pmesh);

//  pSField sfield = new PWLinearSField(mesh);
//  pSField sfield = new PWLsfield(pmesh);
//  pSField sfield = new SizeField(pmesh, fmdb_discretization, sol, ovlp_sol);

  int num_iteration = 1;

  // Does nothing I think
  meshAdapt rdr(pmesh, 0, 0, 1);  // snapping off; do refinement only
  rdr.run(num_iteration, 0, 0);
/*
  // Do basic uniform
  meshAdapt rdr(pmesh, sfield, 0, 1);  // snapping off; do refinement only
  rdr.run(num_iteration, 1, uniform);
*/
//  meshAdapt rdr(pmesh, sfield, 0, 1);  // snapping off; do refinement only
//  rdr.run(num_iteration, 1, setSizeField);

#endif
  return true;
}

//! Transfer solution between meshes.
template<class SizeField>
void
Albany::MeshAdapt<SizeField>::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){

// Just copy across for now!

    newSolution = oldSolution;


}

template<class SizeField>
Teuchos::RCP<const Teuchos::ParameterList>
Albany::MeshAdapt<SizeField>::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");

  validPL->set<int>("Remesh Step Number", 1, "Iteration step at which to remesh the problem");
  validPL->set<int>("Max Number of Mesh Adapt Iterations", 1, "Number of iterations to limit meshadapt to");
  validPL->set<double>("Target Element Size", 0.1, "Seek this element size when isotropically adapting");

  return validPL;
}


