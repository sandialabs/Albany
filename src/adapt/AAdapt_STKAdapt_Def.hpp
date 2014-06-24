//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_STKAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"

#include <stk_adapt/IEdgeBasedAdapterPredicate.hpp>
#include <stk_adapt/IElementBasedAdapterPredicate.hpp>
#include <stk_adapt/PredicateBasedElementAdapter.hpp>
#include <stk_adapt/PredicateBasedEdgeAdapter.hpp>


template<class SizeField>
AAdapt::STKAdapt<SizeField>::
STKAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
         const Teuchos::RCP<ParamLib>& paramLib_,
         Albany::StateManager& StateMgr_,
         const Teuchos::RCP<const Epetra_Comm>& comm_) :
  AAdapt::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
  remeshFileIndex(1) {

  disc = StateMgr_.getDiscretization();

  stk_discretization = static_cast<Albany::STKDiscretization*>(disc.get());

  genericMeshStruct = Teuchos::rcp_dynamic_cast<Albany::GenericSTKMeshStruct>(stk_discretization->getSTKMeshStruct());

  eMesh = genericMeshStruct->getPerceptMesh();
  TEUCHOS_TEST_FOR_EXCEPT(eMesh.is_null());

  refinerPattern = genericMeshStruct->getRefinerPattern();
  TEUCHOS_TEST_FOR_EXCEPT(refinerPattern.is_null());

  num_iterations = adapt_params_->get<int>("Max Number of STK Adapt Iterations", 1);

  // Save the initial output file name
  base_exo_filename = stk_discretization->getSTKMeshStruct()->exoOutFile;


}

template<class SizeField>
AAdapt::STKAdapt<SizeField>::
~STKAdapt() {
}

template<class SizeField>
bool
AAdapt::STKAdapt<SizeField>::queryAdaptationCriteria() {

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
AAdapt::STKAdapt<SizeField>::printElementData() {

  Albany::StateArrays& sa = disc->getStateArrays();
  Albany::StateArrayVec& esa = sa.elemStateArrays;
  int numElemWorksets = esa.size();
  Teuchos::RCP<Albany::StateInfoStruct> stateInfo = state_mgr_.getStateInfoStruct();

  std::cout << "Num Worksets = " << numElemWorksets << std::endl;

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
          std::cout << "Size = " << dims[0] << std::endl;
          break;

        case 2:
          std::cout << "esa[ws][stateName](cell, qp)" << std::endl;
          std::cout << "Size = " << dims[0] << " , " << dims[1] << std::endl;
          break;

        case 3:
          std::cout << "esa[ws][stateName](cell, qp, i)" << std::endl;
          std::cout << "Size = " << dims[0] << " , " << dims[1] << " , " << dims[2] << std::endl;
          break;

        case 4:
          std::cout << "esa[ws][stateName](cell, qp, i, j)" << std::endl;
          std::cout << "Size = " << dims[0] << " , " << dims[1] << " , " << dims[2] << " , " << dims[3] << std::endl;
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
AAdapt::STKAdapt<SizeField>::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol) {

  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *output_stream_ << "Adapting mesh using AAdapt::STKAdapt method        " << std::endl;
  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  Albany::AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field 
      = genericMeshStruct->getFieldContainer()->getProcRankField();
  Albany::AbstractSTKFieldContainer::IntScalarFieldType* refine_field 
      = genericMeshStruct->getFieldContainer()->getRefineField();

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by adding the remesh_file_index_ ahead of the period
  std::ostringstream ss;
  std::string str = base_exo_filename;
  ss << "_" << remeshFileIndex << ".";
  str.replace(str.find('.'), 1, ss.str());

  *output_stream_ << "Remeshing: renaming output file to - " << str << std::endl;

  // Open the new exodus file for results
  stk_discretization->reNameExodusOutput(str);

  remeshFileIndex++;

//  printElementData();

  SizeField set_ref_field(*eMesh);
  eMesh->elementOpLoop(set_ref_field, refine_field);

  //    SetUnrefineField set_unref_field(*eMesh);
  //eMesh.elementOpLoop(set_ref_field, refine_field);

//  eMesh->save_as("local_tet_N_5_ElementBased_0_.e");

  stk_classic::adapt::ElementRefinePredicate erp(0, refine_field, 0.0);

  stk_classic::adapt::PredicateBasedElementAdapter<stk_classic::adapt::ElementRefinePredicate>
  breaker(erp, *eMesh, *refinerPattern, proc_rank_field);

  breaker.setRemoveOldElements(false);
  breaker.setAlwaysInitializeNodeRegistry(false);

  for(int ipass = 0; ipass < 3; ipass++) {

    eMesh->elementOpLoop(set_ref_field, refine_field);

#if 0
    std::vector<stk_classic::mesh::Entity*> elems;
    const std::vector<stk_classic::mesh::Bucket*>& buckets = eMesh->get_bulk_data()->buckets(eMesh->element_rank());

    for(std::vector<stk_classic::mesh::Bucket*>::const_iterator k = buckets.begin() ; k != buckets.end() ; ++k) {
      stk_classic::mesh::Bucket& bucket = **k ;

      const unsigned num_elements_in_bucket = bucket.size();

      for(unsigned i_element = 0; i_element < num_elements_in_bucket; i_element++) {
        stk_classic::mesh::Entity& element = bucket[i_element];
        double* f_data = stk_classic::percept::PerceptMesh::field_data_entity(refine_field, element);

        std::cout << "Element: " << element.identifier() << "Refine field: " << f_data[0] << std::endl;
      }
    }

#endif


    //     std::cout << "P[" << eMesh->get_rank() << "] ipass= " << ipass << std::endl;
    breaker.doBreak();
    //     std::cout << "P[" << eMesh->get_rank() << "] done... ipass= " << ipass << std::endl;
    //     eMesh->save_as("local_tet_N_5_ElementBased_1_ipass_"+Teuchos::toString(ipass)+"_.e");
  }

  breaker.deleteParentElements();
//  eMesh->save_as("local_tet_N_5_ElementBased_1_.e");

  // Throw away all the Albany data structures and re-build them from the mesh

  if(adapt_params_->get<bool>("Rebalance", false))

    genericMeshStruct->rebalanceAdaptedMesh(adapt_params_, epetra_comm_);
    
  stk_discretization->updateMesh();
//  printElementData();

  return true;

}

//! Transfer solution between meshes.
template<class SizeField>
void
AAdapt::STKAdapt<SizeField>::
solutionTransfer(const Epetra_Vector& oldSolution,
                 Epetra_Vector& newSolution) {
#if 0

  // Just copy across for now!

  std::cout << "WARNING: solution transfer not implemented yet!!!" << std::endl;


  std::cout << "AAdapt<> will now throw an exception from line #156" << std::endl;

  newSolution = oldSolution;

#endif

}

template<class SizeField>
Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::STKAdapt<SizeField>::getValidAdapterParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidSTKAdaptParams");

  Teuchos::Array<int> defaultArgs;

  validPL->set<Teuchos::Array<int> >("Remesh Step Number", defaultArgs, "Iteration step at which to remesh the problem");
  validPL->set<std::string>("Remesh Strategy", "", "Strategy to use when remeshing: Continuous - remesh every step.");
  validPL->set<int>("Max Number of STK Adapt Iterations", 1, "Number of iterations to limit stk_adapt to");
  validPL->set<std::string>("Refiner Pattern", "", "Element pattern to use for refinement");
  validPL->set<double>("Target Element Size", 0.1, "Seek this element size when isotropically adapting");
  validPL->set<bool>("Rebalance", "1", "Rebalance mesh after each refinement operation");

  return validPL;
}


