//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AdaptiveSolutionManager.hpp"
#include "Albany_AdaptationFactory.hpp"
#include "Albany_InitialCondition.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"


using Teuchos::rcp;
using Teuchos::RCP;

Albany::AdaptiveSolutionManager::AdaptiveSolutionManager(
           const Teuchos::RCP<Teuchos::ParameterList>& appParams,
           const Teuchos::RCP<Albany::AbstractDiscretization>& disc_,
           const Teuchos::RCP<const Epetra_Vector>& initial_guess) :
    disc(disc_),
    out(Teuchos::VerboseObjectBase::getDefaultOStream()),
    Piro::Epetra::AdaptiveSolutionManager(appParams, 
       disc_->getMap(), disc_->getOverlapMap(), disc_->getOverlapJacobianGraph())
{
    setInitialSolution(disc->getSolutionField());

 // Load connectivity map and coordinates 
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > > wsElNodeEqID = disc->getWsElNodeEqID();
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords = disc->getCoords();
  Teuchos::ArrayRCP<std::string> wsEBNames = disc->getWsEBNames();
  int numDim = disc->getNumDim();
  int neq = disc->getNumEq();

  RCP<Teuchos::ParameterList> problemParams = 
    Teuchos::sublist(appParams, "Problem", true);

  if (initial_guess != Teuchos::null) *initial_x = *initial_guess;
  else {
    overlapped_x->Import(*initial_x, *importer, Insert);
    Albany::InitialConditions(overlapped_x, wsElNodeEqID, wsEBNames, coords, neq, numDim,
                              problemParams->sublist("Initial Condition"),
                              disc->hasRestartSolution());
    Albany::InitialConditions(overlapped_xdot,  wsElNodeEqID, wsEBNames, coords, neq, numDim,
                              problemParams->sublist("Initial Condition Dot"));
    initial_x->Export(*overlapped_x, *exporter, Insert);
    initial_xdot->Export(*overlapped_xdot, *exporter, Insert);
  }

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

  if(adaptManager->adaptMesh()){ // resize problem if the mesh adapts

    resizeMeshDataArrays(disc->getMap(), 
       disc->getOverlapMap(), disc->getOverlapJacobianGraph());

    // Build a new solution vector, transfer the last solution to it, and re-initialize the vectors and groups in the solver.
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

void
Albany::AdaptiveSolutionManager::
scatterX(const Epetra_Vector& x, const Epetra_Vector* xdot){

  // Scatter x and xdot to the overlapped distribution
  overlapped_x->Import(x, *importer, Insert);
  if (xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);

}

