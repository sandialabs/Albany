//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STKAdapt.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"

#include <stk_adapt/IEdgeBasedAdapterPredicate.hpp>
#include <stk_adapt/IElementBasedAdapterPredicate.hpp>
#include <stk_adapt/PredicateBasedElementAdapter.hpp>
#include <stk_adapt/PredicateBasedEdgeAdapter.hpp>


template<class SizeField>
Albany::STKAdapt<SizeField>::
STKAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_) :
    Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
    remeshFileIndex(1)
{

    disc = StateMgr_.getDiscretization();

    stk_discretization = static_cast<Albany::STKDiscretization *>(disc.get());

    genericMeshStruct = Teuchos::rcp_dynamic_cast<Albany::GenericSTKMeshStruct>(stk_discretization->getSTKMeshStruct());

    eMesh = genericMeshStruct->getPerceptMesh();
    TEUCHOS_TEST_FOR_EXCEPT(eMesh.is_null());

    refinerPattern = genericMeshStruct->getRefinerPattern();
    TEUCHOS_TEST_FOR_EXCEPT(refinerPattern.is_null());

    num_iterations = adapt_params_->get<int>("Max Number of STK Adapt Iterations", 1);

}

template<class SizeField>
Albany::STKAdapt<SizeField>::
~STKAdapt()
{
}

template<class SizeField>
bool
Albany::STKAdapt<SizeField>::queryAdaptationCriteria(){

  int remesh_iter = adapt_params_->get<int>("Remesh Step Number");

   if(iter == remesh_iter)
     return true;

  return false;

}

template<class SizeField>
bool
Albany::STKAdapt<SizeField>::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol){

  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  *output_stream_ << "Adapting mesh using Albany::STKAdapt method        " << std::endl;
  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = genericMeshStruct->getFieldContainer()->getProcRankField();
  AbstractSTKFieldContainer::IntScalarFieldType* refine_field = genericMeshStruct->getFieldContainer()->getRefineField();

  SizeField set_ref_field(*eMesh);
  eMesh->elementOpLoop(set_ref_field, refine_field);

//    SetUnrefineField set_unref_field(*eMesh);
    //eMesh.elementOpLoop(set_ref_field, refine_field);

  eMesh->save_as( "local_tet_N_5_ElementBased_0_.e");

  stk::adapt::ElementRefinePredicate erp(0, refine_field, 0.0);

  stk::adapt::PredicateBasedElementAdapter<stk::adapt::ElementRefinePredicate>
     breaker(erp, *eMesh, *refinerPattern, proc_rank_field);

  breaker.setRemoveOldElements(false);
  breaker.setAlwaysInitializeNodeRegistry(false);

  for (int ipass=0; ipass < 3; ipass++) {

     eMesh->elementOpLoop(set_ref_field, refine_field);

     std::cout << "P[" << eMesh->get_rank() << "] ipass= " << ipass << std::endl;
     breaker.doBreak();
     std::cout << "P[" << eMesh->get_rank() << "] done... ipass= " << ipass << std::endl;
     eMesh->save_as("local_tet_N_5_ElementBased_1_ipass_"+Teuchos::toString(ipass)+"_.e");
   }

   breaker.deleteParentElements();
   eMesh->save_as("local_tet_N_5_ElementBased_1_.e");

   return true;

}

//! Transfer solution between meshes.
template<class SizeField>
void
Albany::STKAdapt<SizeField>::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){
#if 0

// Just copy across for now!

std::cout << "WARNING: solution transfer not implemented yet!!!" << std::endl;


std::cout << "Albany_MeshAdapt<> will now throw an exception from line #156" << std::endl;

    newSolution = oldSolution;

#endif

}

template<class SizeField>
Teuchos::RCP<const Teuchos::ParameterList>
Albany::STKAdapt<SizeField>::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidSTKAdaptParams");

  validPL->set<int>("Remesh Step Number", 1, "Iteration step at which to remesh the problem");
  validPL->set<int>("Max Number of STK Adapt Iterations", 1, "Number of iterations to limit stk_adapt to");
  validPL->set<string>("Refiner Pattern", "", "Element pattern to use for refinement");
  validPL->set<double>("Target Element Size", 0.1, "Seek this element size when isotropically adapting");

  return validPL;
}


