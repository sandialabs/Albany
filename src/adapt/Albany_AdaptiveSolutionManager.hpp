//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ADAPTIVESOLUTIONMANAGER
#define ALBANY_ADAPTIVESOLUTIONMANAGER

#include "Piro_Epetra_AdaptiveSolutionManager.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "Sacado_ScalarParameterLibrary.hpp"
#include "Albany_StateManager.hpp"


namespace Albany {

  class AdaptiveSolutionManager : public Piro::Epetra::AdaptiveSolutionManager {
  
  public:
     AdaptiveSolutionManager (
          const Teuchos::RCP<Teuchos::ParameterList>& appParams,
          const Teuchos::RCP<Albany::AbstractDiscretization> &disc_);

    virtual ~AdaptiveSolutionManager();

    //! Build a mesh adaptive problem
    void buildAdaptiveProblem(const Teuchos::RCP<ParamLib>& paramLib,
                   Albany::StateManager& StateMgr,
		               const Teuchos::RCP<const Epetra_Comm>& comm);

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    virtual bool adaptProblem();

    //! Remap the solution
    virtual void
      projectCurrentSolution();
 
  protected:

    //! Element discretization
    Teuchos::RCP<Albany::AbstractDiscretization> disc;

    //! Output stream, defaults to pronting just Proc 0
    Teuchos::RCP<Teuchos::FancyOStream> out;

    
  };

}

#endif //ALBANY_ADAPTIVESOLUTIONMANAGER


