//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ADAPTATIONFACTORY_HPP
#define ALBANY_ADAPTATIONFACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractAdapter.hpp"

namespace Albany {

  /*!
   * \brief A factory class to instantiate AbstractAdapter objects
   */
  class AdaptationFactory {
  public:

    //! Default constructor
    AdaptationFactory(const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
                   const Teuchos::RCP<ParamLib>& paramLib,
                   Albany::StateManager& StateMgr,
		               const Teuchos::RCP<const Epetra_Comm>& comm);

    //! Destructor
    virtual ~AdaptationFactory() {}

    virtual Teuchos::RCP<Albany::AbstractAdapter> create();

  private:

    //! Private to prohibit copying
    AdaptationFactory(const AdaptationFactory&);

    //! Private to prohibit copying
    AdaptationFactory& operator=(const AdaptationFactory&);

  protected:

    //! Parameter list specifying what problem to create
    Teuchos::RCP<Teuchos::ParameterList> adaptParams;

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;

    //! MPI Communicator
    Teuchos::RCP<const Epetra_Comm> comm;

    //! State manager (to get ahold of stress, etc)
    Albany::StateManager& StateMgr;

  };

}

#endif // ALBANY_ADAPTATIONFACTORY_HPP
