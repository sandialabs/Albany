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


#ifndef ALBANY_RESPONSE_FACTORY_HPP
#define ALBANY_RESPONSE_FACTORY_HPP

#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_StateInfoStruct.hpp" // contains MeshSpecsStuct
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

  /*!
   * \brief Factory for creating response functions from a parameter list
   */
  class ResponseFactory {
  public:
  
    //! Default constructor
    ResponseFactory(
      const Teuchos::RCP<Albany::Application>& application,
      const Teuchos::RCP<Albany::AbstractProblem>& problem,
      const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&  ms,
      const Teuchos::RCP<Albany::StateManager>& state_manager) :
      app(application), prob(problem), meshSpecs(ms), stateMgr(state_manager) 
      {};

    //! Destructor
    virtual ~ResponseFactory() {};

    //! Create a set of response functions
    virtual Teuchos::Array< Teuchos::RCP<AbstractResponseFunction> >
    createResponseFunctions(Teuchos::ParameterList& responsesList) const;


  private:

    //! Private to prohibit copying
    ResponseFactory(const ResponseFactory&);
    
    //! Private to prohibit copying
    ResponseFactory& operator=(const ResponseFactory&);

  protected:

    //! Application for field manager response functions
    Teuchos::RCP<Albany::Application> app;

    //! Problem class for field manager response functions
    Teuchos::RCP<Albany::AbstractProblem> prob;

    //! Meshspecs for field manager response functions
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs;

    //! State manager for field manager response functions
    Teuchos::RCP<Albany::StateManager> stateMgr;

    //! Create individual response function
    void createResponseFunction(
      const std::string& name,
      Teuchos::ParameterList& responseParams,
      Teuchos::Array< Teuchos::RCP<AbstractResponseFunction> >& responses) const;

  };

}

#endif // ALBANY_SCALAR_RESPONSE_FUNCTION_HPP
