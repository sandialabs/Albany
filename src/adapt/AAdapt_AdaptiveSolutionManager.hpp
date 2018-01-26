//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_ADAPTIVESOLUTIONMANAGER
#define AADAPT_ADAPTIVESOLUTIONMANAGER

#include "Piro_Epetra_AdaptiveSolutionManager.hpp"
#include "Piro_SolutionObserverBase.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "AAdapt_AdaptiveModelFactory.hpp"
#include "AAdapt_SolutionObserver.hpp"

#include "Sacado_ScalarParameterLibrary.hpp"
#include "Albany_StateManager.hpp"


namespace AAdapt {

typedef Teuchos::RCP<Piro::SolutionObserverBase<double, const Thyra::VectorBase<double> > > AdaptSolutionObserverType;

class AdaptiveSolutionManager : public Piro::Epetra::AdaptiveSolutionManager {

  public:
    AdaptiveSolutionManager(
      const Teuchos::RCP<Teuchos::ParameterList>& appParams,
      const Teuchos::RCP<Albany::AbstractDiscretization>& disc_,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess);

    virtual ~AdaptiveSolutionManager();

    //! Build a mesh adaptive problem
    void buildAdaptiveProblem(const Teuchos::RCP<ParamLib>& paramLib,
                              Albany::StateManager& StateMgr,
                              const Teuchos::RCP<const Teuchos_Comm>& commT);

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    virtual bool adaptProblem();

    //! Build the model factory that returns the Thyra Model Evaluator wrapping Albany::ModelEvaluator
    virtual Teuchos::RCP<AAdapt::AdaptiveModelFactory> modelFactory() const;

    AdaptSolutionObserverType getSolObserver(){ return solutionObserver; }

    //! Remap the solution
    virtual void
    projectCurrentSolution();

    void scatterX(const Epetra_Vector& x, const Epetra_Vector* xdot, const Epetra_Vector* xdotdot);


  protected:

    //! Element discretization
    Teuchos::RCP<Albany::AbstractDiscretization> disc;

    //! Output stream, defaults to printing just Proc 0
    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! The adaptive thyra model factory object
    Teuchos::RCP<AAdapt::AdaptiveModelFactory> thyra_model_factory;

    Teuchos::RCP<SolutionObserver> solutionObserver;

};

}

#endif //ALBANY_ADAPTIVESOLUTIONMANAGER


