/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_RESPONSEUTILITIES_HPP
#define ALBANY_RESPONSEUTILITIES_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_AbstractResponseFunction.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

//! Code Base for Quantum Device Simulation Tools LDRD
namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */

  template<typename EvalT, typename Traits>
  class ResponseUtilities {

    public:

    ResponseUtilities(Teuchos::RCP<Albany::Layouts> dl);
  
    //! Utility for parsing response requests and creating response field manager
    void
    constructResponses(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  		       Teuchos::ParameterList& responseList, 
	               Albany::StateManager& stateMgr);


    //! Utility for creating response field manager from responses
    void
    createResponseFieldManager(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
			       const std::vector<std::string>& responseIDs_to_require);

    //! - Returns true if responseName was recognized and response function constructed.
    //! - If p is non-Teuchos::null upon exit, then an evaluator should be build using
    //!   p as the parameter list. 
    bool getStdResponseFn(std::string responseName, int responseIndex,
			  Teuchos::ParameterList& responseList,
			  Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
			  Albany::StateManager& stateMgr,
			  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >& ev);


    //! Helper function for constructResponses and getStdResponseFn
    Teuchos::RCP<Teuchos::ParameterList> setupResponseFnForEvaluator(  
		  Teuchos::ParameterList& responseList, int responseNumber,
		  Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

 
    //! Accessor 
    Teuchos::RCP<Albany::Layouts> get_dl() { return dl;};

   private:

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;
  };
}

#endif
