//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_AdaptiveSolutionManagerT.hpp"
#if defined(HAVE_STK)
#include "AAdapt_CopyRemeshT.hpp"
#if defined(ALBANY_LCM)
#include "AAdapt_TopologyModificationT.hpp"
#endif
#if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)
//#include "AAdapt_RandomFracture.hpp"
#endif
#if defined(ALBANY_LCM) && defined(ALBANY_STK_PERCEPT)
#include "AAdapt_STKAdaptT.hpp"
#endif
#endif
#ifdef ALBANY_SCOREC
#include "AAdapt_MeshAdapt.hpp"
#endif
#include "AAdapt_RC_Manager.hpp"

#include "Thyra_ModelEvaluatorDelegatorBase.hpp"

#include "Albany_ModelEvaluatorT.hpp"

AAdapt::AdaptiveSolutionManagerT::AdaptiveSolutionManagerT(
    const Teuchos::RCP<Teuchos::ParameterList>& appParams,
    const Teuchos::RCP<const Tpetra_Vector>& initial_guessT,
    const Teuchos::RCP<ParamLib>& param_lib,
    const Albany::StateManager& stateMgr,
    const Teuchos::RCP<rc::Manager>& rc_mgr,
    const Teuchos::RCP<const Teuchos_Comm>& commT) :

    out(Teuchos::VerboseObjectBase::getDefaultOStream()),
    appParams_(appParams),
    disc_(stateMgr.getDiscretization()),
    paramLib_(param_lib),
    stateMgr_(stateMgr),
    commT_(commT)
{

  // Create problem PL
  Teuchos::RCP<Teuchos::ParameterList> problemParams =
      Teuchos::sublist(appParams_, "Problem", true);

  // Note that piroParams_ is a member of LOCA_Thyra_AdaptiveSolutionManager
  piroParams_ = Teuchos::sublist(appParams_, "Piro", true);

  if (problemParams->isSublist("Adaptation")) { // If the user has specified adaptation on input, grab the sublist
    // Note that piroParams_ and adaptiveMesh_ are members of LOCA_Thyra_AdaptiveSolutionManager
    adaptParams_ = Teuchos::sublist(problemParams, "Adaptation", true);
    if (Teuchos::nonnull(rc_mgr)) {
      // This call stack: LOCA::*Stepper -> printSolution -> ObserverImpl ->
      // updateStates means _new states get copied to _old any time
      // printSolution is called. It is incorrect to call updateStates in the
      // relaxation step. Without modifying LOCA::AdaptiveStepper, which I don't
      // want to do right now, I can't fine-tune ObserverImpl. So disallow:
      adaptParams_->set<bool>("Print Relaxation Solution", false);
      // As a side effect, postProcessContinuationStep is also not called, so
      // RFs won't be evaluated.
    }
    adaptiveMesh_ = true;
    buildAdapter(rc_mgr);
  }

  const Teuchos::RCP<const Tpetra_Map> mapT = disc_->getMapT();
  const Teuchos::RCP<const Tpetra_Map> overlapMapT = disc_->getOverlapMapT();
  const Teuchos::RCP<const Tpetra_CrsGraph> overlapJacGraphT = disc_
      ->getOverlapJacobianGraphT();

  resizeMeshDataArrays(mapT, overlapMapT, overlapJacGraphT);

  {
    Teuchos::ArrayRCP<
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > > wsElNodeEqID =
        disc_->getWsElNodeEqID();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords =
        disc_->getCoords();
    Teuchos::ArrayRCP<std::string> wsEBNames = disc_->getWsEBNames();
    const int numDim = disc_->getNumDim();
    const int neq = disc_->getNumEq();

    Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(
        appParams_,
        "Problem",
        true);
    if (Teuchos::nonnull(initial_guessT)) {
      initial_xT = Teuchos::rcp(new Tpetra_Vector(*initial_guessT));
    } else {
      overlapped_xT->doImport(*initial_xT, *importerT, Tpetra::INSERT);

      AAdapt::InitialConditionsT(
          overlapped_xT, wsElNodeEqID, wsEBNames, coords, neq, numDim,
          problemParams->sublist("Initial Condition"),
          disc_->hasRestartSolution());
      AAdapt::InitialConditionsT(
          overlapped_xdotT, wsElNodeEqID, wsEBNames, coords, neq, numDim,
          problemParams->sublist("Initial Condition Dot"));

      initial_xT->doExport(*overlapped_xT, *exporterT, Tpetra::INSERT);
      initial_xdotT->doExport(*overlapped_xdotT, *exporterT, Tpetra::INSERT);
    }
  }
}

void AAdapt::AdaptiveSolutionManagerT::
buildAdapter(const Teuchos::RCP<rc::Manager>& rc_mgr)
{

  std::string& method = adaptParams_->get("Method", "");

#if defined(HAVE_STK)
  if (method == "Copy Remesh") {
    adapter_ = Teuchos::rcp(new AAdapt::CopyRemeshT(adaptParams_,
        paramLib_,
        stateMgr_,
        commT_));
  } else

# if defined(ALBANY_LCM)
  if (method == "Topmod") {
    adapter_ = Teuchos::rcp(new AAdapt::TopologyModT(adaptParams_,
        paramLib_,
        stateMgr_,
        commT_));
  } else
# endif
#endif

#if 0
# if defined(ALBANY_LCM) && defined(LCM_SPECULATIVE)
  if (method == "Random") {
    strategy = rcp(new AAdapt::RandomFracture(adaptParams_,
            param_lib_,
            state_mgr_,
            epetra_comm_));
  } else
# endif
#endif
#ifdef ALBANY_SCOREC
  // RCP needs to be non-owned because otherwise there is an RCP circle.
  if (method == "RPI Unif Size" || method == "RPI UnifRef Size" ||
      method == "RPI SPR Size") {
    adapter_ = Teuchos::rcp(
      new AAdapt::MeshAdaptT(adaptParams_, paramLib_, stateMgr_, rc_mgr,
                             commT_));
  } else
#endif
#if defined(ALBANY_LCM) && defined(ALBANY_STK_PERCEPT)
  if (method == "Unif Size") {
    adapter_ = Teuchos::rcp(new AAdapt::STKAdaptT<AAdapt::STKUnifRefineField>(adaptParams_,
            paramLib_,
            stateMgr_,
            commT_));
  } else
#endif

  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,
        Teuchos::Exceptions::InvalidParameter,
        std::endl <<
        "Error! Unknown adaptivity method requested:"
        << method <<
        " !" << std::endl
        << "Supplied parameter list is " <<
        std::endl << *adaptParams_);
  }

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
      << " Mesh adapter has been initialized:\n"
      << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
      << std::endl;

}

bool
AAdapt::AdaptiveSolutionManagerT::
adaptProblem()
{

  const Teuchos::RCP<const Tpetra_Vector> oldSolution =
      ConverterT::getConstTpetraVector(model_->getNominalValues().get_x());

  Teuchos::RCP<Tpetra_Vector> oldOvlpSolution = getOverlapSolutionT(
      *oldSolution);

  // resize problem if the mesh adapts
  if (adapter_->adaptMesh(oldSolution, oldOvlpSolution)) {

    resizeMeshDataArrays(disc_->getMapT(),
        disc_->getOverlapMapT(), disc_->getOverlapJacobianGraphT());

    Teuchos::RCP<Thyra::ModelEvaluatorDelegatorBase<ST> > base =
        Teuchos::rcp_dynamic_cast<Thyra::ModelEvaluatorDelegatorBase<ST> >(
            model_);

    // If dynamic cast fails
    TEUCHOS_TEST_FOR_EXCEPTION(
        base == Teuchos::null,
        std::logic_error,
        std::endl <<
        "Error! : Cast to Thyra::ModelEvaluatorDelegatorBase failed!" << std::endl);

    Teuchos::RCP<Albany::ModelEvaluatorT> me =
        Teuchos::rcp_dynamic_cast<Albany::ModelEvaluatorT>(
            base->getNonconstUnderlyingModel());

    // If dynamic cast fails
    TEUCHOS_TEST_FOR_EXCEPTION(me == Teuchos::null,
        std::logic_error,
        std::endl <<
        "Error! : Cast to Albany::ModelEvaluatorT failed!" << std::endl);

    // Allocate storage in the model evaluator
    me->allocateVectors();

    // Build the solution group down in LOCA_Thyra_AdaptiveSolutionManager.C
    this->buildSolutionGroup();

    // getSolutionField() below returns the new solution vector with the fields transferred to it
    initial_xT = disc_->getSolutionFieldT();

    *out << "Mesh adaptation was successfully performed!" << std::endl;

    return true;

  }

  *out << "Mesh adaptation was NOT successfully performed!" << std::endl;

  *out
      << "Mesh adaptation machinery has returned a FAILURE error code, exiting Albany!"
      << std::endl;

  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "Mesh adaptation failed!\n");

  return false;

}

void AAdapt::AdaptiveSolutionManagerT::resizeMeshDataArrays(
    const Teuchos::RCP<const Tpetra_Map> &mapT,
    const Teuchos::RCP<const Tpetra_Map> &overlapMapT,
    const Teuchos::RCP<const Tpetra_CrsGraph> &overlapJacGraphT)
{

  importerT = Teuchos::rcp(new Tpetra_Import(mapT, overlapMapT));
  exporterT = Teuchos::rcp(new Tpetra_Export(overlapMapT, mapT));

  overlapped_xT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
  overlapped_xdotT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
  overlapped_xdotdotT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
  overlapped_fT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
  overlapped_jacT = Teuchos::rcp(new Tpetra_CrsMatrix(overlapJacGraphT));

  tmp_ovlp_solT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));

  initial_xT = disc_->getSolutionFieldT();
  initial_xdotT = Teuchos::rcp(new Tpetra_Vector(mapT));

}

Teuchos::RCP<Tpetra_Vector>
AAdapt::AdaptiveSolutionManagerT::getOverlapSolutionT(
    const Tpetra_Vector& solutionT)
{
  tmp_ovlp_solT->doImport(solutionT, *importerT, Tpetra::INSERT);
  return tmp_ovlp_solT;
}

void
AAdapt::AdaptiveSolutionManagerT::scatterXT(
    const Tpetra_Vector& xT,
    const Tpetra_Vector* x_dotT,
    const Tpetra_Vector* x_dotdotT)
{

  overlapped_xT->doImport(xT, *importerT, Tpetra::INSERT);

  if (x_dotT) overlapped_xdotT->doImport(*x_dotT, *importerT, Tpetra::INSERT);
  if (x_dotdotT)
    overlapped_xdotdotT->doImport(*x_dotdotT, *importerT, Tpetra::INSERT);

}

void
AAdapt::AdaptiveSolutionManagerT::
projectCurrentSolution()
{

  // grp->getNOXThyraVecRCPX() is the current solution on the old mesh

  // TO provide an example, assume that the meshes are identical and we can just copy the data between them (a Copy Remesh)

  /*
   const Teuchos::RCP<const Tpetra_Vector> testSolution =
   ConverterT::getConstTpetraVector(grp_->getNOXThyraVecRCPX()->getThyraRCPVector());
   */

//    *initial_xT = *testSolution;
}
