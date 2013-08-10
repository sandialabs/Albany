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
  Piro::Epetra::AdaptiveSolutionManager(appParams,
                                        disc_->getMap(), disc_->getOverlapMap(), disc_->getOverlapJacobianGraph()) {
  setInitialSolution(disc->getSolutionField());

  // Load connectivity map and coordinates
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > > wsElNodeEqID = disc->getWsElNodeEqID();
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords = disc->getCoords();
  Teuchos::ArrayRCP<std::string> wsEBNames = disc->getWsEBNames();
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

    setInitialSolution(disc->getSolutionField());

    // Get the Thrya solver ME and resize the solution array
//    Teuchos::RCP<Thyra::ModelEvaluator<double> > thyra_model = thyra_model_factory->getThyraModel();
    Teuchos::RCP<ThyraAdaptiveModelEvaluator> thyra_model 
        = Teuchos::rcp_dynamic_cast<ThyraAdaptiveModelEvaluator>(thyra_model_factory->getThyraModel());

    // Get the total number of responses. Note that we assume here that the last one is the final solution vector
    const int num_g = thyra_model->Ng();

    // Resize the solution vector. getMap() returns the new, larger map
    thyra_model->resize_g_space(num_g-1, disc->getMap());

/*
    EpetraExt::ModelEvaluator::OutArgs outArgs = thyra_model->getEpetraModel()->createOutArgs();
    outArgs.set_g(num_g-1, evector);

  g_map_.resize(outArgs.Ng()); g_space_.resize(outArgs.Ng());
  g_map_is_local_.resize(outArgs.Ng(),false);
  for( int j = 0; j < implicit_cast<int>(g_space_.size()); ++j ) {
    RCP<const Epetra_Map>
      g_map_j = ( g_map_[j] = epetraModel_->get_g_map(j) );
    g_map_is_local_[j] = !g_map_j->DistributedGlobal();
    g_space_[j] = create_VectorSpace( g_map_j );
*/

    // Note: the current solution on the old mesh is projected onto this new mesh inside the stepper,
    // at LOCA_Epetra_AdaptiveStepper.C line 515. This line calls
    // AAdapt::AdaptiveSolutionManager::projectCurrentSolution()
    // if we return true.

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
#if 0
  // Not currently needed

  const Epetra_Vector& oldSolution
    = dynamic_cast<const NOX::Epetra::Vector&>(grp->getX()).getEpetraVector();

  adaptManager->solutionTransfer(oldSolution, currentSolution->getEpetraVector());
#endif

}

void
AAdapt::AdaptiveSolutionManager::
scatterX(const Epetra_Vector& x, const Epetra_Vector* xdot) {

  // Scatter x and xdot to the overlapped distribution
  overlapped_x->Import(x, *importer, Insert);

  if(xdot != NULL) overlapped_xdot->Import(*xdot, *importer, Insert);

}

