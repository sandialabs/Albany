//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <ma.h>
#include <PCU.h>
#include <parma.h>

template<class SizeField>
AAdapt::MeshAdapt<SizeField>::
MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
          const Albany::StateManager& StateMgr_):
  remeshFileIndex(1) {

  disc = StateMgr_.getDiscretization();

  pumi_discretization = Teuchos::rcp_dynamic_cast<AlbPUMI::AbstractPUMIDiscretization>(disc);

  Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct =
      pumi_discretization->getFMDBMeshStruct();

  mesh = fmdbMeshStruct->getMesh();

  szField = Teuchos::rcp(new SizeField(pumi_discretization));

  num_iterations = params_->get<int>("Max Number of Mesh Adapt Iterations", 1);

  // Save the initial output file name
  base_exo_filename = fmdbMeshStruct->outputFileName;

  adaptation_method = params_->get<std::string>("Method");

  if ( adaptation_method.compare(0,15,"RPI SPR Size") == 0 )
    checkValidStateVariable(StateMgr_,params_->get<std::string>("State Variable",""));

}

template<class SizeField>
AAdapt::MeshAdapt<SizeField>::
~MeshAdapt() {
}

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::queryAdaptationCriteria(
    const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
    int iter)
{
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
AAdapt::MeshAdapt<SizeField>::beforeAdapt(
    const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
    Teuchos::RCP<Teuchos::FancyOStream>& output_stream_)
{

  if(PCU_Comm_Self() == 0){
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "Adapting mesh using AAdapt::MeshAdapt method        " << std::endl;
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


  // attach qp data to mesh if solution transfer is turned on
  bool shouldTransferIPData = adapt_params_->get<bool>("Transfer IP Data",false);
  if (shouldTransferIPData)
    pumi_discretization->attachQPData();

  std::string adaptVector = adapt_params_->get<std::string>("Adaptation Displacement Vector","");

  szField->setParams(adapt_params_->get<double>("Target Element Size", 0.1),
		     adapt_params_->get<double>("Error Bound", 0.01),
		     adapt_params_->get<std::string>("State Variable", ""));

  szField->copyInputFields();
}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::adaptInPartition(
    const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{
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
    input->shouldRunMidParma = true;
    input->shouldRunPostParma = true;
    input->maximumImbalance = lbMaxImbalance;
  }

  ma::adapt(input);

  szField->freeSizeField();
}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::afterAdapt(
    const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{
  mesh->verify();

  bool shouldTransferIPData = adapt_params_->get<bool>("Transfer IP Data",false);
  szField->freeInputFields();
  // Throw away all the Albany data structures and re-build them from the mesh
  // Note that the solution transfer for the QP fields happens in this call
  pumi_discretization->updateMesh(shouldTransferIPData);

  // detach QP fields from the apf mesh
  if (shouldTransferIPData)
    pumi_discretization->detachQPData();
}

template <class T>
struct AdaptCallbackOf : public Parma_GroupCode
{
  T* adapter;
  const Teuchos::RCP<Teuchos::ParameterList>* adapt_params;
  void run(int group) {
    adapter->adaptInPartition(*adapt_params);
  }
};

void adaptShrunken(apf::Mesh2* m, double minPartDensity);

template<class SizeField>
bool
AAdapt::MeshAdapt<SizeField>::adaptMesh(
    const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
    Teuchos::RCP<Teuchos::FancyOStream>& output_stream_)
{
  beforeAdapt(adapt_params_, output_stream_);

  AdaptCallbackOf<AAdapt::MeshAdapt<SizeField> > callback;
  callback.adapter = this;
  callback.adapt_params = &adapt_params_;
  double minPartDensity = adapt_params_->get<double>("Minimum Part Density", 1000);
  adaptShrunken(mesh, minPartDensity, callback);
  afterAdapt(adapt_params_);

  return true;

}

template<class SizeField>
void
AAdapt::MeshAdapt<SizeField>::checkValidStateVariable(
    const Albany::StateManager& state_mgr_,
    const std::string name)
{
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
AAdapt::MeshAdapt<SizeField>::getValidAdapterParameters(
    Teuchos::RCP<Teuchos::ParameterList>& validPL) const
{

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
  validPL->set<double>("Minimum Part Density", 1000, "Minimum elements per part: triggers partition shrinking");
  return validPL;
}

template<class SizeField>
AAdapt::MeshAdaptT<SizeField>::
MeshAdaptT(const Teuchos::RCP<Teuchos::ParameterList>& params_,
           const Teuchos::RCP<ParamLib>& paramLib_,
           const Albany::StateManager& StateMgr_,
           const Teuchos::RCP<const Teuchos_Comm>& commT_):
  AbstractAdapterT(params_,paramLib_,StateMgr_,commT_),
  meshAdapt(params_,StateMgr_)
{
}

template<class SizeField>
bool
AAdapt::MeshAdaptT<SizeField>::queryAdaptationCriteria(int iteration)
{
  return meshAdapt.queryAdaptationCriteria(this->adapt_params_,iteration);
}

template<class SizeField>
bool
AAdapt::MeshAdaptT<SizeField>::adaptMesh(
        const Teuchos::RCP<const Tpetra_Vector>& solution,
        const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution)
{
  return meshAdapt.adaptMesh(
      this->adapt_params_,this->output_stream_);
}

template<class SizeField>
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::MeshAdaptT<SizeField>::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");
  return meshAdapt.getValidAdapterParameters(validPL);
}

