//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"

template<class SizeField>
Teuchos::RCP<SizeField> AAdapt::MeshAdapt<SizeField>::szField = Teuchos::null;

template<class SizeField>
AAdapt::MeshAdapt<SizeField>::
MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
          const Teuchos::RCP<ParamLib>& paramLib_,
          Albany::StateManager& StateMgr_,
          const Teuchos::RCP<const Epetra_Comm>& comm_) :
  AAdapt::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
  remeshFileIndex(1) {

  disc = StateMgr_.getDiscretization();

  pumi_discretization = Teuchos::rcp_dynamic_cast<AlbPUMI::AbstractPUMIDiscretization>(disc);

  fmdbMeshStruct = pumi_discretization->getFMDBMeshStruct();

  mesh = fmdbMeshStruct->getMesh();

  szField = Teuchos::rcp(new SizeField(pumi_discretization, state_mgr_));

  num_iterations = params_->get<int>("Max Number of Mesh Adapt Iterations", 1);

  adaptation_method = params_->get<std::string>("Method");

  // Do basic uniform refinement
  /** Type of the size field:
      - Application - the size field will be provided by the application (default).
      - TagDriven - tag driven size field.
      - Analytical - analytical size field.  */
  /** Type of model:
      - 0 - no model (not snap), 1 - mesh model (always snap), 2 - solid model (always snap)
  */


  //    rdr = Teuchos::rcp(new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 2 ));
  rdr = Teuchos::rcp(new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 0));

}

template<class SizeField>
AAdapt::MeshAdapt<SizeField>::
~MeshAdapt() {
  // Not needed with RCP
  //  delete rdr;
}

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::queryAdaptationCriteria() {

  int remesh_iter = adapt_params_->get<int>("Remesh Step Number");

  if(iter == remesh_iter)
    return true;

  return false;

}

template<class SizeField>
int
AAdapt::MeshAdapt<SizeField>::setSizeField(pPart part, pSField pSizeField, void* vp) {

  return szField->computeSizeField(part, pSizeField);

}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::printElementData() {

  Albany::StateArrays& sa = disc->getStateArrays();
  int numWorksets = sa.size();
  Teuchos::RCP<Albany::StateInfoStruct> stateInfo = state_mgr_.getStateInfoStruct();

  for(unsigned int i = 0; i < stateInfo->size(); i++) {

    const std::string stateName = (*stateInfo)[i]->name;
    const std::string init_type = (*stateInfo)[i]->initType;
    std::vector<int> dims;
    sa[0][stateName].dimensions(dims);
    int size = dims.size();

    std::cout << "Meshadapt: have element field \"" << stateName << "\" of type \"" << init_type << "\"" << std::endl;

    if(init_type == "scalar") {


      switch(size) {

        case 1:
          std::cout << "sa[ws][stateName](0)" << std::endl;
          break;

        case 2:
          std::cout << "sa[ws][stateName](cell, qp)" << std::endl;
          break;

        case 3:
          std::cout << "sa[ws][stateName](cell, qp, i)" << std::endl;
          break;

        case 4:
          std::cout << "sa[ws][stateName](cell, qp, i, j)" << std::endl;
          break;

      }
    }

    else if(init_type == "identity") {
      std::cout << "Have an identity matrix: " << "sa[ws][stateName](cell, qp, i, j)" << std::endl;
    }
  }
}

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol) {

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using AAdapt::MeshAdapt method        " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  //  printElementData();

  // display # entities before adaptation

  FMDB_Mesh_DspSize(mesh);

#if 0
  // write out the mesh and solution before adapting
  pumi_discretization->debugMeshWrite(sol, "unmodified_mesh_out.vtk");
#endif

  // replace nodes' coordinates with displaced coordinates
  PUMI_Mesh_SetDisp(mesh, fmdbMeshStruct->solution_field_tag);

  szField->setParams(&sol, &ovlp_sol,
                     adapt_params_->get<double>("Target Element Size", 0.1),
		     adapt_params_->get<double>("Error Bound", 0.01),
		     adapt_params_->get<std::string>("State Variable", ""));

  szField->computeError();

  /** void meshAdapt::run(int niter,    // specify the maximum number of iterations
        int flag,           // indicate if a size field function call is available
        adaptSFunc sizefd)  // the size field function call  */

  rdr->run(num_iterations, 1, this->setSizeField);

  // replace nodes' displaced coordinates with coordinates
  PUMI_Mesh_DelDisp(mesh, fmdbMeshStruct->solution_field_tag);

  // display # entities after adaptation
  FMDB_Mesh_DspSize(mesh);

  // Reinitialize global and local ids in FMDB
  PUMI_Exodus_Init(mesh);  // generate global/local id

  // Throw away all the Albany data structures and re-build them from the mesh
  pumi_discretization->updateMesh();

  // dump the adapted mesh for visualization
  Teuchos::RCP<Epetra_Vector> new_sol = disc->getSolutionField();
  new_sol->Print(std::cout);

  //  pumi_discretization->debugMeshWrite(sol, "adapted_mesh_out.vtk");
  pumi_discretization->debugMeshWrite(*new_sol, "adapted_mesh_out.vtk");

  return true;

}


//! Transfer solution between meshes.
template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution) {

  // Just copy across for now!

  //std::cout << "WARNING: solution transfer not implemented yet!!!" << std::endl;


  //std::cout << "AAdapt<> will now throw an exception from line #156" << std::endl;

  //    newSolution = oldSolution;

  // Should now pick up solution from AAdapt::FMDBDiscretization::getSolutionField()


}

template<class SizeField>
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::MeshAdapt<SizeField>::getValidAdapterParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");

  validPL->set<int>("Remesh Step Number", 1, "Iteration step at which to remesh the problem");
  validPL->set<int>("Max Number of Mesh Adapt Iterations", 1, "Number of iterations to limit meshadapt to");
  validPL->set<double>("Target Element Size", 0.1, "Seek this element size when isotropically adapting");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<std::string>("State Variable", "", "Error is estimated using this state variable at integration points. Must be a 3x3 tensor. If no state variable is specified during error-estimation based adaptivity, then the gradient of solution field will be recovered and used");
  
  return validPL;
}


