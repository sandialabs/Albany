//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_AdaptiveSolutionManagerT.hpp"
#if defined(HAVE_STK)
#include "AAdapt_CopyRemeshT.hpp"
#if defined(ALBANY_LCM) && defined(ALBANY_BGL)
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
#ifdef ALBANY_AMP
#include "AAdapt_SimAdapt.hpp"
#include "AAdapt_SimLayerAdapt.hpp"
#endif
#if (defined(ALBANY_SCOREC) || defined(ALBANY_AMP))
#include "Albany_APFDiscretization.hpp"
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
    num_time_deriv(appParams->sublist("Discretization").get<int>("Number Of Time Derivatives")),
    commT_(commT)
{

  // Create problem PL
  Teuchos::RCP<Teuchos::ParameterList> problemParams =
      Teuchos::sublist(appParams_, "Problem", true);

  // Note that piroParams_ is a member of Thyra_AdaptiveSolutionManager
  piroParams_ = Teuchos::sublist(appParams_, "Piro", true);

  if (problemParams->isSublist("Adaptation")) { // If the user has specified adaptation on input, grab the sublist
    // Note that piroParams_ and adaptiveMesh_ are members of Thyra_AdaptiveSolutionManager
    adaptParams_ = Teuchos::sublist(problemParams, "Adaptation", true);
    adaptiveMesh_ = true;
    buildAdapter(rc_mgr);
  }

  // Want the initial time in the parameter library to be correct
  // if this is a restart solution
  // MJJ (12/06/16) I'll go an remove this conditional "if (disc_->hasRestartSolution())"
  // Independent of restarting analysis you want to have the capability of specify
  // an initial time. Not doing so was causing that the initial time was always set to zero
  // and the latter cause many problem because for time 0, dt is set up as 1.0e-15, creating
  // serious problem when a body source is specified at time 0. For example, in a heat
  // transfer problem at time 0 and specifying a heat source the problem may corresponds to
  // a "thermal shock" problem, causing some instabilities in the time integrator.
//  if (disc_->hasRestartSolution()) {
    if (paramLib_->isParameter("Time")) {

      double initialValue = 0.0;

      if(appParams->sublist("Problem").
         get<std::string>("Solution Method", "Steady") == "Continuation")
      {
        initialValue =
          appParams->sublist("Piro").sublist("LOCA").sublist("Stepper").
          get<double>("Initial Value", 0.0);
      }
      else if(appParams->sublist("Problem").
              get<std::string>("Solution Method", "Steady") == "Transient")
      {
        initialValue =
          appParams->sublist("Piro").sublist("Trapezoid Rule").
          get<double>("Initial Time", 0.0);
      }
      paramLib_->setRealValue<PHAL::AlbanyTraits::Residual>("Time", initialValue);
    }
//  }

  const Teuchos::RCP<const Tpetra_Map> mapT = disc_->getMapT();
  const Teuchos::RCP<const Tpetra_Map> overlapMapT = disc_->getOverlapMapT();
#ifdef ALBANY_AERAS
  //IKT, 1/20/15: the following is needed to ensure Laplace matrix is non-diagonal
  //for Aeras problems that have hyperviscosity and are integrated using an explicit time
  //integration scheme.
  const Teuchos::RCP<const Tpetra_CrsGraph> overlapJacGraphT = disc_
      ->getImplicitOverlapJacobianGraphT();
#else
  const Teuchos::RCP<const Tpetra_CrsGraph> overlapJacGraphT = disc_
      ->getOverlapJacobianGraphT();
#endif

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

      *current_soln->getVectorNonConst(0) = *initial_guessT;

    } else {

      overlapped_soln->getVectorNonConst(0)->doImport(*current_soln->getVector(0), *importerT, Tpetra::INSERT);

      AAdapt::InitialConditionsT(
          overlapped_soln->getVectorNonConst(0), wsElNodeEqID, wsEBNames, coords, neq, numDim,
          problemParams->sublist("Initial Condition"),
          disc_->hasRestartSolution());

      current_soln->getVectorNonConst(0)->doExport(*overlapped_soln->getVector(0), *exporterT, Tpetra::INSERT);

      if(num_time_deriv > 0){
          overlapped_soln->getVectorNonConst(1)->doImport(*current_soln->getVector(1), *importerT, Tpetra::INSERT);
          AAdapt::InitialConditionsT(
             overlapped_soln->getVectorNonConst(1), wsElNodeEqID, wsEBNames, coords, neq, numDim,
             problemParams->sublist("Initial Condition Dot"));
          current_soln->getVectorNonConst(1)->doExport(*overlapped_soln->getVector(1), *exporterT, Tpetra::INSERT);
       }

       if(num_time_deriv > 1){
          overlapped_soln->getVectorNonConst(2)->doImport(*current_soln->getVector(2), *importerT, Tpetra::INSERT);
          AAdapt::InitialConditionsT(
             overlapped_soln->getVectorNonConst(2), wsElNodeEqID, wsEBNames, coords, neq, numDim,
             problemParams->sublist("Initial Condition DotDot"));
          current_soln->getVectorNonConst(2)->doExport(*overlapped_soln->getVector(2), *exporterT, Tpetra::INSERT);
        }

    }
  }
#if (defined(ALBANY_SCOREC) || defined(ALBANY_AMP))
  {
    const Teuchos::RCP< Albany::APFDiscretization > apf_disc =
      Teuchos::rcp_dynamic_cast< Albany::APFDiscretization >(disc_);
    if ( ! apf_disc.is_null()) {
      apf_disc->writeSolutionMVToMeshDatabase(*overlapped_soln, 0, true);
      apf_disc->initTemperatureHack();
    }
  }
#endif
}

void AAdapt::AdaptiveSolutionManagerT::
buildAdapter(const Teuchos::RCP<rc::Manager>& rc_mgr)
{

  std::string& method = adaptParams_->get("Method", "");
  std::string first_three_chars = method.substr(0, 3);

#if defined(HAVE_STK)
  if (method == "Copy Remesh") {
    adapter_ = Teuchos::rcp(new AAdapt::CopyRemeshT(adaptParams_,
        paramLib_,
        stateMgr_,
        commT_));
  } else

# if defined(ALBANY_LCM) && defined(ALBANY_BGL)
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
  if (first_three_chars == "RPI") {
    adapter_ = Teuchos::rcp(
      new AAdapt::MeshAdapt(adaptParams_, paramLib_, stateMgr_, rc_mgr,
                             commT_));
  } else
#endif
#ifdef ALBANY_AMP
  if (method == "Sim") {
    bool add_layer = false;
    if (adaptParams_->isType<bool>("Add Layer"))
      add_layer = adaptParams_->get<bool>("Add Layer");
    if (add_layer) { // add layer
      *out << "************************" << std::endl;
      *out << "    ADDING LAYER ON     " << std::endl;
      *out << "************************" << std::endl;
      adapter_ = Teuchos::rcp(
          new AAdapt::SimLayerAdapt(adaptParams_, paramLib_, stateMgr_, commT_));
    } else { // do not add layer
      adapter_ = Teuchos::rcp(
          new AAdapt::SimAdapt(adaptParams_, paramLib_, stateMgr_, commT_));
    }
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

//Teuchos::RCP<const Teuchos::ParameterList> valid_params =
//  adapter_->getValidAdapterParameters();
//adaptParams_->validateParameters(*valid_params);

  *out << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
      << " Mesh adapter has been initialized:\n"
      << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
      << std::endl;
}

bool
AAdapt::AdaptiveSolutionManagerT::
adaptProblem()
{

  Teuchos::RCP<Thyra::ModelEvaluator<double> > model = this->getState()->getModel();

  // resize problem if the mesh adapts
  if (adapter_->adaptMesh()) {

    resizeMeshDataArrays(disc_->getMapT(),
        disc_->getOverlapMapT(), disc_->getOverlapJacobianGraphT());

    Teuchos::RCP<Thyra::ModelEvaluatorDelegatorBase<ST> > base =
        Teuchos::rcp_dynamic_cast<Thyra::ModelEvaluatorDelegatorBase<ST> >(
            model);

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

    // Build the solution group down in Thyra_AdaptiveSolutionManager.cpp
    this->getState()->buildSolutionGroup();

    // getSolutionMV() below returns the new solution vector with the fields transferred to it

    current_soln = disc_->getSolutionMV();

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

  overlapped_soln = Teuchos::rcp(new Tpetra_MultiVector(overlapMapT, num_time_deriv + 1, false));

  overlapped_fT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
  overlapped_jacT = Teuchos::rcp(new Tpetra_CrsMatrix(overlapJacGraphT));

  // This call allocates the non-overlapped MV
  current_soln = disc_->getSolutionMV();

}

Teuchos::RCP<Tpetra_Vector>
AAdapt::AdaptiveSolutionManagerT::updateAndReturnOverlapSolutionT(
    const Tpetra_Vector& solutionT /* not overlapped */)
{
  overlapped_soln->getVectorNonConst(0)->doImport(solutionT, *importerT, Tpetra::INSERT);
  return overlapped_soln->getVectorNonConst(0);
}

Teuchos::RCP<const Tpetra_MultiVector>
AAdapt::AdaptiveSolutionManagerT::updateAndReturnOverlapSolutionMV(
    const Tpetra_MultiVector& solutionT /* not overlapped */)
{
  overlapped_soln->doImport(solutionT, *importerT, Tpetra::INSERT);
  return overlapped_soln;
}

void
AAdapt::AdaptiveSolutionManagerT::scatterXT(
    const Tpetra_Vector& xT, /* note that none are overlapped */
    const Tpetra_Vector* x_dotT,
    const Tpetra_Vector* x_dotdotT)
{

  overlapped_soln->getVectorNonConst(0)->doImport(xT, *importerT, Tpetra::INSERT);

  if (x_dotT){
     TEUCHOS_TEST_FOR_EXCEPTION(overlapped_soln->getNumVectors() < 2, std::logic_error,
         "AdaptiveSolutionManager error: x_dotT defined but only a single solution vector is available");
     overlapped_soln->getVectorNonConst(1)->doImport(*x_dotT, *importerT, Tpetra::INSERT);
  }

  if (x_dotdotT){
     TEUCHOS_TEST_FOR_EXCEPTION(overlapped_soln->getNumVectors() < 3, std::logic_error,
         "AdaptiveSolutionManager error: x_dotdotT defined but xDotDot isn't defined in the multivector");
     overlapped_soln->getVectorNonConst(2)->doImport(*x_dotdotT, *importerT, Tpetra::INSERT);

	  /*OG uncomment this to enable Laplace calculations in Aeras::Hydrostatic
	 if(overlapped_soln->getNumVectors() == 3)
	    overlapped_soln->getVectorNonConst(2)->doImport(*x_dotdotT, *importerT, Tpetra::INSERT);
	    */
  }

}

void
AAdapt::AdaptiveSolutionManagerT::scatterXT(
    const Tpetra_MultiVector& soln) /* not overlapped */
{

  overlapped_soln->doImport(soln, *importerT, Tpetra::INSERT);

}


Teuchos::RCP<Thyra::MultiVectorBase<double> >
AAdapt::AdaptiveSolutionManagerT::
getCurrentSolution()
{
   return Thyra::createMultiVector<ST, LO, GO, KokkosNode>(current_soln);
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
