//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <ma.h>

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

  Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct =
      pumi_discretization->getFMDBMeshStruct();

  mesh = fmdbMeshStruct->apfMesh;
  pumiMesh = fmdbMeshStruct->getMesh();

  szField = Teuchos::rcp(new SizeField(pumi_discretization));

  num_iterations = params_->get<int>("Max Number of Mesh Adapt Iterations", 1);

  // Save the initial output file name
  base_exo_filename = fmdbMeshStruct->outputFileName;

  adaptation_method = params_->get<std::string>("Method");

  if ( adaptation_method.compare(0,15,"RPI SPR Size") == 0 )
    checkValidStateVariable(params_->get<std::string>("State Variable",""));

}

template<class SizeField>
AAdapt::MeshAdapt<SizeField>::
~MeshAdapt() {
}

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::queryAdaptationCriteria() {

  if(adapt_params_->get<std::string>("Remesh Strategy", "None").compare("Continuous") == 0){

    if(iter > 1)

      return true;

    else

      return false;

  }


  Teuchos::Array<int> remesh_iter = adapt_params_->get<Teuchos::Array<int> >("Remesh Step Number");

  for(int i = 0; i < remesh_iter.size(); i++)

    if(iter == remesh_iter[i])

      return true;

  return false;

}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::printElementData() {

  Albany::StateArrays& sa = disc->getStateArrays();
  Albany::StateArrayVec& esa = sa.elemStateArrays;
  int numElemWorksets = esa.size();
  Teuchos::RCP<Albany::StateInfoStruct> stateInfo = state_mgr_.getStateInfoStruct();

  for(unsigned int i = 0; i < stateInfo->size(); i++) {

    const std::string stateName = (*stateInfo)[i]->name;
    const std::string init_type = (*stateInfo)[i]->initType;
    std::vector<int> dims;
    esa[0][stateName].dimensions(dims);
    int size = dims.size();

    std::cout << "Meshadapt: have element field \"" << stateName << "\" of type \"" << init_type << "\"" << std::endl;

    if(init_type == "scalar") {


      switch(size) {

        case 1:
          std::cout << "esa[ws][stateName](0)" << std::endl;
          break;

        case 2:
          std::cout << "esa[ws][stateName](cell, qp)" << std::endl;
          break;

        case 3:
          std::cout << "esa[ws][stateName](cell, qp, i)" << std::endl;
          break;

        case 4:
          std::cout << "esa[ws][stateName](cell, qp, i, j)" << std::endl;
          break;

      }
    }

    else if(init_type == "identity") {
      std::cout << "Have an identity matrix: " << "esa[ws][stateName](cell, qp, i, j)" << std::endl;
    }
  }
}

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol) {

  if(epetra_comm_->MyPID() == 0){
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "Adapting mesh using AAdapt::MeshAdapt method        " << std::endl;
    std::cout << "Iteration: " << iter                                  << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  }

 // Create a remeshed output file naming convention by adding the remesh_file_index_ ahead of the period
  std::size_t found = base_exo_filename.find("exo");
  if(found != std::string::npos){
    std::ostringstream ss;
    std::string str = base_exo_filename;
    ss << "_" << remeshFileIndex << ".";
    str.replace(str.find('.'), 1, ss.str());

    *output_stream_ << "Remeshing: renaming exodus output file to - " << str << std::endl;

    // Open the new exodus file for results
    pumi_discretization->reNameExodusOutput(str);

    remeshFileIndex++;

  }

  // display # entities before adaptation

  FMDB_Mesh_DspSize(pumiMesh);

  // attach qp data to mesh if solution transfer is turned on
  bool shouldTransferIPData = adapt_params_->get<bool>("Transfer IP Data",false);
  if (shouldTransferIPData)
    pumi_discretization->attachQPData();

  std::string adaptVector = adapt_params_->get<std::string>("Adaptation Displacement Vector","");
  apf::Field* solutionField;
  if (adaptVector.length() != 0)
    solutionField = mesh->findField(adaptVector.c_str());
  else
    solutionField = mesh->findField("solution");

  // replace nodes' coordinates with displaced coordinates
  if ( ! PCU_Comm_Self())
    fprintf(stderr,"assuming deformation problem: displacing coordinates\n");
  apf::displaceMesh(mesh,solutionField);

  szField->setParams(adapt_params_->get<double>("Target Element Size", 0.1),
		     adapt_params_->get<double>("Error Bound", 0.01),
		     adapt_params_->get<std::string>("State Variable", ""));

  szField->computeError();

  ma::Input* input = ma::configure(mesh,&(*szField));
      // Teuchos::RCP to regular pointer ^
  input->maximumIterations = num_iterations;
  //do not snap on deformation problems even if the model supports it
  input->shouldSnap = false;

  bool loadBalancing = adapt_params_->get<bool>("Load Balancing",true);
  double lbMaxImbalance = adapt_params_->get<double>("Maximum LB Imbalance",1.30);
  if (loadBalancing) {
    input->shouldRunPreZoltan = true;
    input->shouldRunMidDiffusion = true;
    input->shouldRunPostDiffusion = true;
    input->maximumImbalance = lbMaxImbalance;
  }

  ma::adapt(input);

  if ( adaptation_method.compare(0,15,"RPI SPR Size") == 0 ) {
    apf::destroyField(mesh->findField("size"));
  }
  
  // replace nodes' displaced coordinates with coordinates
  apf::displaceMesh(mesh,solutionField,-1.0);

  // display # entities after adaptation
  FMDB_Mesh_DspSize(pumiMesh);

  // Throw away all the Albany data structures and re-build them from the mesh
  // Note that the solution transfer for the QP fields happens in this call
  pumi_discretization->updateMesh(shouldTransferIPData);

  // detach QP fields from the apf mesh
  if (shouldTransferIPData)
    pumi_discretization->detachQPData();

  return true;

}


//! Transfer solution between meshes.
template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution) {
}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::checkValidStateVariable(const std::string name) {

  if (name.length() > 0) {

    // does state variable exist?
    std::string stateName;
    
    Albany::StateArrays& sa = disc->getStateArrays();
    Albany::StateArrayVec& esa = sa.elemStateArrays;
    Teuchos::RCP<Albany::StateInfoStruct> stateInfo = state_mgr_.getStateInfoStruct();
    bool exists = false;
    for(unsigned int i = 0; i < stateInfo->size(); i++) {
      stateName = (*stateInfo)[i]->name;
      if ( name.compare(0,100,stateName) == 0 ){
        exists = true;
        break;
      }
    }
    if (!exists)
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
          "Error!    Invalid State Variable Parameter!");
    
    // is state variable a 3x3 tensor?
    
    std::vector<int> dims;
    esa[0][name].dimensions(dims);
    int size = dims.size();
    if (size != 4)
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      "Error! Invalid State Variable Parameter \"" << name << "\" looking for \"" << stateName << "\"" << std::endl);
  }
}

template<class SizeField>
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::MeshAdapt<SizeField>::getValidAdapterParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");

  Teuchos::Array<int> defaultArgs;

  validPL->set<Teuchos::Array<int> >("Remesh Step Number", defaultArgs, "Iteration step at which to remesh the problem");
  validPL->set<std::string>("Remesh Strategy", "", "Strategy to use when remeshing: Continuous - remesh every step.");
  validPL->set<int>("Max Number of Mesh Adapt Iterations", 1, "Number of iterations to limit meshadapt to");
  validPL->set<double>("Target Element Size", 0.1, "Seek this element size when isotropically adapting");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<std::string>("State Variable", "", "SPR operates on this variable");
  validPL->set<bool>("Load Balancing", true, "Turn on predictive load balancing");
  validPL->set<double>("Maximum LB Imbalance", 1.3, "Set maximum imbalance tolerance for predictive laod balancing");
  validPL->set<std::string>("Adaptation Displacement Vector", "", "Name of APF displacement field");
  validPL->set<bool>("Transfer IP Data", false, "Turn on solution transfer of integration point data");

  return validPL;
}


