//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AdaptiveSolutionManager.hpp"
#include "Albany_AdaptationFactory.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"

// FIXME needed to implement cast below.
#ifdef ALBANY_SCOREC
#include "Albany_MeshAdapt.hpp"
#endif


using Teuchos::rcp;
using Teuchos::RCP;

Albany::AdaptiveSolutionManager::AdaptiveSolutionManager(
           const Teuchos::RCP<Teuchos::ParameterList>& appParams,
           const Teuchos::RCP<Albany::AbstractDiscretization> &disc_) :
    disc(disc_),
    out(Teuchos::VerboseObjectBase::getDefaultOStream()),
    Piro::Epetra::AdaptiveSolutionManager(appParams, 
       disc_->getMap(), disc_->getOverlapMap(), disc_->getOverlapJacobianGraph())
{
    setInitialSolution(disc->getSolutionField());
}

Albany::AdaptiveSolutionManager::~AdaptiveSolutionManager(){

}

void 
Albany::AdaptiveSolutionManager::buildAdaptiveProblem(const Teuchos::RCP<ParamLib>& paramLib,
                   Albany::StateManager& stateMgr,
		               const Teuchos::RCP<const Epetra_Comm>& comm){

    RCP<Albany::AdaptationFactory> adaptationFactory
       = rcp(new Albany::AdaptationFactory(adaptParams, paramLib, stateMgr, comm));

    adaptManager = adaptationFactory->create();

    *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
         << " Mesh adapter has been initialized:\n" 
         << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
         << endl;

}

bool
Albany::AdaptiveSolutionManager::
adaptProblem(){

#ifdef ALBANY_SCOREC

  if(adaptParams->get<std::string>("Method") == "RPI Mesh Adapt"){

    Albany::MeshAdapt* rpiAdaptManager = dynamic_cast<Albany::MeshAdapt*>(adaptManager.get());

    const Epetra_Vector& oldSolution 
      = dynamic_cast<const NOX::Epetra::Vector&>(grp->getX()).getEpetraVector();

    const Epetra_Vector& oldOvlpSolution = *getOverlapSolution(oldSolution);

//    if(rpiAdaptManager->adaptMesh(oldSolution, importer)){ // RPI meshAdapt needs current solution vector to calculate size field
    if(rpiAdaptManager->adaptMesh(oldSolution, oldOvlpSolution)){ // RPI meshAdapt needs current solution vector to calculate size field
  
      resizeMeshDataArrays(disc->getMap(), 
         disc->getOverlapMap(), disc->getOverlapJacobianGraph());
  
      setInitialSolution(disc->getSolutionField());
  
      // Note: the current solution on the old mesh is projected onto this new mesh inside the stepper,
      // at LOCA_Epetra_AdaptiveStepper.C line 515. This line calls 
      // Albany::AdaptiveSolutionManager::projectCurrentSolution()
      // if we return true.
  
      *out << "Mesh adaptation was successfully performed!" << endl;
  
      return true;
  
    }
  }

  else
#endif
    if(adaptManager->adaptMesh()){ // resize problem if the mesh adapts
  
      resizeMeshDataArrays(disc->getMap(), 
         disc->getOverlapMap(), disc->getOverlapJacobianGraph());
  
      setInitialSolution(disc->getSolutionField());
  
      // Note: the current solution on the old mesh is projected onto this new mesh inside the stepper,
      // at LOCA_Epetra_AdaptiveStepper.C line 515. This line calls 
      // Albany::AdaptiveSolutionManager::projectCurrentSolution()
      // if we return true.
  
      *out << "Mesh adaptation was successfully performed!" << endl;
  
      return true;
  
    }

  return false;

}

void
Albany::AdaptiveSolutionManager::
projectCurrentSolution()
{

  const Epetra_Vector& oldSolution 
     = dynamic_cast<const NOX::Epetra::Vector&>(grp->getX()).getEpetraVector();

  adaptManager->solutionTransfer(oldSolution, currentSolution->getEpetraVector());

}

