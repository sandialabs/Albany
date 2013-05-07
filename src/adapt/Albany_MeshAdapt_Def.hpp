//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MeshAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "PUMI.h"

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

    num_iterations = params->get<int>("Max Number of Mesh Adapt Iterations", 1);

    // Do basic uniform refinement
    /** Type of the size field: 
        - Application - the size field will be provided by the application (default).
        - TagDriven - tag driven size field. 
        - Analytical - analytical size field.  */
    /** Type of model: 
        - 0 - no model (not snap), 1 - mesh model (always snap), 2 - solid model (always snap)
    */


    rdr = Teuchos::rcp(new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 2 ));

}

template<class SizeField>
Albany::MeshAdapt<SizeField>::
~MeshAdapt()
{
// Not needed with RCP
//  delete rdr;
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

template<class SizeField>
bool
//Albany::MeshAdapt::adaptMesh(const Epetra_Vector& Solution, const Teuchos::RCP<Epetra_Import>& importer){
Albany::MeshAdapt<SizeField>::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol){

  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *out << "Adapting mesh using Albany::MeshAdapt method        " << std::endl;
  *out << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  // display # entities before adaptation

  FMDB_Mesh_DspNumEnt (mesh);


//  PUMI_Mesh_SetDisp(mesh, fmdbMeshStruct->solution_field_tag);  


  szField->setParams(&sol, &ovlp_sol,
              params->get<double>("Target Element Size", 0.1));


  /** void meshAdapt::run(int niter,    // specify the maximum number of iterations 
		    int flag,           // indicate if a size field function call is available
		    adaptSFunc sizefd)  // the size field function call  */

  rdr->run (num_iterations, 1, this->setSizeField);

//  PUMI_Mesh_DelDisp(mesh, fmdbMeshStruct->solution_field_tag);

  // dump the adapted mesh for visualization

  FMDB_Mesh_WriteToFile (mesh, "adapted_mesh_out.vtk",  (SCUTIL_CommSize()>1?1:0));

  // display # entities after adaptation

  FMDB_Mesh_DspNumEnt (mesh);

/* mesh verification overwrites mesh entity id so commented out temporarily
   FMDB will be updated to use different id for validity check 

  int isValid=0;
  FMDB_Mesh_Verify(mesh, &isValid);
*/

  // Reinitialize global and local ids in FMDB

  PUMI_Exodus_Init (mesh); // generate global/local id 

  // Throw away all the Albany data structures and re-build them from the mesh

  fmdb_discretization->updateMesh(fmdbMeshStruct, comm);

  return true;

}

//! Transfer solution between meshes.
template<class SizeField>
void
Albany::MeshAdapt<SizeField>::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){

// Just copy across for now!

std::cout << "WARNING: solution transfer not implemented yet!!!" << std::endl;


std::cout << "Albany_MeshAdapt<> will now throw an exception from line #156" << std::endl;

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


