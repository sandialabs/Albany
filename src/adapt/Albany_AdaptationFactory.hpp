//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#if !defined(Albany_AdaptationFracory_hpp)
#define Albany_AdaptationFracory_hpp

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include "Albany_AbstractAdapter.hpp"

namespace Albany {

  ///
  /// \brief A factory class to instantiate AbstractAdapter objects
  ///
  class AdaptationFactory {
  public:

    ///
    /// Default constructor
    ///
    AdaptationFactory(const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
                      const Teuchos::RCP<ParamLib>& paramLib,
                      Albany::StateManager& StateMgr,
                      const Teuchos::RCP<const Epetra_Comm>& comm);

    ///
    /// Destructor
    ///
    virtual ~AdaptationFactory() {}

    ///
    /// Method to create a specific derived Adapter class
    ///
    virtual Teuchos::RCP<Albany::AbstractAdapter> createAdapter();

  private:

    //! Private to prohibit copying
    AdaptationFactory(const AdaptationFactory&);

    //! Private to prohibit copying
    AdaptationFactory& operator=(const AdaptationFactory&);

  protected:

    ///
    /// Parameter list specifying what problem to create
    ///
    Teuchos::RCP<Teuchos::ParameterList> adapt_params_;

    //! Parameter library
    Teuchos::RCP<ParamLib> param_lib_;

    //! MPI Communicator
    Teuchos::RCP<const Epetra_Comm> epetra_comm_;

    //! State manager (to get ahold of stress, etc)
    Albany::StateManager& state_mgr_;

  };

}

#endif // Albany_AdaptationFracory_hpp
