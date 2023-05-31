//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
