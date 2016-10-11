#ifndef CTM_APPLICATION_HPP
#define CTM_APPLICATION_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
//
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Workset.hpp"
#include "Phalanx.hpp"
//
#include "Albany_AbstractProblem.hpp"

namespace CTM {

    class SolutionInfo;

    class Application {
    public:
        Application(Teuchos::RCP<Teuchos::ParameterList> p,
                Teuchos::RCP<SolutionInfo> sinfo, 
                Teuchos::RCP<Albany::AbstractProblem> prob,
                Teuchos::RCP<Albany::AbstractDiscretization> d);
        // prohibit copy constructor
        Application(const Application& app) = delete;
        //! Destructor
        ~Application();
    private:
        // Problem parameter list
        Teuchos::RCP<Teuchos::ParameterList> params;
        
        // solution info
        Teuchos::RCP<SolutionInfo> solution_info;

        //! Output stream, defaults to pronting just Proc 0
        Teuchos::RCP<Teuchos::FancyOStream> out;
        
        // Problem to be solved
        Teuchos::RCP<Albany::AbstractProblem> problem;
        
        // discretization
        Teuchos::RCP<Albany::AbstractDiscretization> disc;

        //! Phalanx Field Manager for volumetric fills
        Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > fm;

        //! Phalanx Field Manager for Dirichlet Conditions
        Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm;

        //! Phalanx Field Manager for Neumann Conditions
        Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > nfm;

    };

}

#endif