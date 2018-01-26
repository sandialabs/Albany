//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#if !defined(AAdapt_AdaptationFracory_hpp)
#define AAdapt_AdaptationFracory_hpp

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include "AAdapt_AbstractAdapter.hpp"

namespace AAdapt {

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
                      const Teuchos::RCP<const Teuchos_Comm>& commT);

    ///
    /// Destructor
    ///
    virtual ~AdaptationFactory() {}

    ///
    /// Method to create a specific derived Adapter class
    ///
    virtual Teuchos::RCP<AAdapt::AbstractAdapter> createAdapter();

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
    Teuchos::RCP<const Teuchos_Comm> commT_;

    //! State manager (to get ahold of stress, etc)
    Albany::StateManager& state_mgr_;

};

}

#endif // AdaptationFracory_hpp
