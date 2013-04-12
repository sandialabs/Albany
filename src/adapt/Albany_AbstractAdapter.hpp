//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTADAPTER_HPP
#define ALBANY_ABSTRACTADAPTER_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"

#include "NOX_Epetra_AdaptManager.H"

#include "Sacado_ScalarParameterLibrary.hpp"
#include "Albany_StateManager.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing the set of adaptation tools available.
   * 
   */
  class AbstractAdapter : public NOX::Epetra::AdaptManager {

  public:
  
    //! Only constructor
    AbstractAdapter( const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    virtual ~AbstractAdapter() {};

    //! Each adapter must generate it's list of valid parameters
    virtual Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const 
      {return getGenericAdapterParams("Generic Adapter List");}

  protected:

    //! List of valid problem params common to all adapters, as 
    //! a starting point for the specific  getValidAdaptaterParameters
    Teuchos::RCP<Teuchos::ParameterList>
      getGenericAdapterParams(std::string listname = "AdapterList") const;

    //! Configurable output stream, defaults to printing on proc=0
    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Adaptation parameters
    Teuchos::RCP<Teuchos::ParameterList> params;

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;

    Albany::StateManager& StateMgr;

    Teuchos::RCP<const Epetra_Comm> comm;

  private:

    //! Private to prohibit default or copy constructor
    AbstractAdapter();
    AbstractAdapter(const AbstractAdapter&);
    
    //! Private to prohibit copying
    AbstractAdapter& operator=(const AbstractAdapter&);


  };


}

#endif // ALBANY_ABSTRACTADAPTATION_HPP
