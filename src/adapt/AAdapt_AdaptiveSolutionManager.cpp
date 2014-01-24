//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_AdaptiveSolutionManager.hpp"
#include "AAdapt_AdaptationFactory.hpp"
#include "AAdapt_InitialCondition.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "AAdapt_ThyraAdaptiveModelEvaluator.hpp"

using Teuchos::rcp;
using Teuchos::RCP;

AAdapt::AdaptiveSolutionManager::AdaptiveSolutionManager(
  const Teuchos::RCP<Teuchos::ParameterList>& appParams,
  const Teuchos::RCP<Albany::AbstractDiscretization>& disc_,
  const Teuchos::RCP<const Epetra_Vector>& initial_guess) :
  disc(disc_),
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  thyra_model_factory(new AAdapt::AdaptiveModelFactory(appParams)),
  solutionObserver(new AAdapt::SolutionObserver()),
  Piro::Epetra::AdaptiveSolutionManager(appParams,
                                        disc_->getMap(), disc_->getOverlapMap(), disc_->getOverlapJacobianGraph()) {
  setInitialSolution(disc->getSolutionField());

  // Load connectivity map and coordinates
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();

  int numDim = disc->getNumDim();
  int neq = disc->getNumEq();

  RCP<Teuchos::ParameterList> problemParams =
    Teuchos::sublist(appParams, "Problem", true);

  if(initial_guess != Teuchos::null) *initial_x = *initial_guess;

  else {
    overlapped_x->Import(*initial_x, *importer, Insert);
    AAdapt::InitialConditions(overlapped_x, wsElNodeEqID, wsEBNames, coords, neq, numDim,
                              problemParams->sublist("Initial Condition"),
                              disc->hasRestartSolution());
    AAdapt::InitialConditions(overlapped_xdot,  wsElNodeEqID, wsEBNames, coords, neq, numDim,
                              problemParams->sublist("Initial Condition Dot"));
    initial_x->Export(*overlapped_x, *exporter, Insert);
    initial_xdot->Export(*overlapped_xdot, *exporter, Insert);
  }

}

AAdapt::AdaptiveSolutionManager::~AdaptiveSolutionManager() {

  thyra_model_factory = Teuchos::null;
#ifdef ALBANY_DEBUG
  *out << "Calling destructor for Albany_AdaptiveSolutionManager" << std::endl;
#endif
}

void
AAdapt::AdaptiveSolutionManager::buildAdaptiveProblem(const Teuchos::RCP<ParamLib>& paramLib,
    Albany::StateManager& stateMgr,
    const Teuchos::RCP<const Epetra_Comm>& comm) {

  RCP<AAdapt::AdaptationFactory> adaptationFactory
    = rcp(new AAdapt::AdaptationFactory(adaptParams, paramLib, stateMgr, comm));

  adaptManager = adaptationFactory->createAdapter();

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << " Mesh adapter has been initialized:\n"
       << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
       << std::endl;

}

bool
AAdapt::AdaptiveSolutionManager::
adaptProblem() {

  const Epetra_Vector& oldSolution
    = dynamic_cast<const NOX::Epetra::Vector&>(grp->getX()).getEpetraVector();

  const Epetra_Vector& oldOvlpSolution = *getOverlapSolution(oldSolution);

  // resize problem if the mesh adapts
  if(adaptManager->adaptMesh(oldSolution, oldOvlpSolution)) {

    resizeMeshDataArrays(disc->getMap(),
                         disc->getOverlapMap(), disc->getOverlapJacobianGraph());

    // getSolutionField() below returns the new solution vector with the fields transferred to it
    setInitialSolution(disc->getSolutionField());

    // Get the Thrya solver ME and resize the solution array
    Teuchos::RCP<ThyraAdaptiveModelEvaluator> thyra_model 
        = Teuchos::rcp_dynamic_cast<ThyraAdaptiveModelEvaluator>(thyra_model_factory->getThyraModel());

    // Get the total number of responses. Note that we assume here that the last one is the final solution vector
    const int num_g = thyra_model->Ng();

    // Resize the solution vector. getMap() returns the new, larger map
    const Teuchos::RCP<Thyra::VectorBase<double> > g_j 
        = thyra_model->resize_g_space(num_g-1, disc->getMap());
    solutionObserver->set_g_vector(num_g-1, g_j);

    // Original design:
    // Note: the current solution on the old mesh is projected onto this new mesh inside the stepper,
    // at LOCA_Epetra_AdaptiveStepper.C line 515. This line calls
    // AAdapt::AdaptiveSolutionManager::projectCurrentSolution()
    // if we return true.
    // Note that solution transfer now occurs above, and the projectCurrentSolution() is now a no-op

    *out << "Mesh adaptation was successfully performed!" << std::endl;

    return true;

  }

  return false;

}

Teuchos::RCP<AAdapt::AdaptiveModelFactory>
AAdapt::AdaptiveSolutionManager::
modelFactory() const {

  return thyra_model_factory;

}


void
AAdapt::AdaptiveSolutionManager::
projectCurrentSolution() {
#if 1 // turn on to print debug info from MeshAdapt
  // Not currently needed

  const Epetra_Vector& oldSolution
    = dynamic_cast<const NOX::Epetra::Vector&>(grp->getX()).getEpetraVector();

  adaptManager->solutionTransfer(oldSolution, currentSolution->getEpetraVector());
#endif

}

void
AAdapt::AdaptiveSolutionManager::
scatterX(const Epetra_Vector& x, const Epetra_Vector* xdot, const Epetra_Vector* xdotdot) {

  // Scatter x and xdot to the overlapped distribution
  overlapped_x->Import(x, *importer, Insert);

  if(xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);
  if(xdotdot != NULL) overlapped_xdotdot->Import(*xdotdot, *importer, Insert);

}

