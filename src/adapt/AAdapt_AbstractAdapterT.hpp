//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#if !defined(AAdapt_AbstractAdapterT_hpp)
#define AAdapt_AbstractAdapterT_hpp

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Sacado_ScalarParameterLibrary.hpp>

#include "Albany_StateManager.hpp"

namespace AAdapt {

///
/// \brief Abstract interface for representing the set of adaptation tools available.
///
class AbstractAdapterT {

  public:

    ///
    /// Only constructor
    ///
    AbstractAdapterT(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     const Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Teuchos_Comm>& comm_);

    ///
    /// Destructor
    ///
    virtual ~AbstractAdapterT() {};

    //! Method called by LOCA Solver to determine if the mesh needs adapting
    // A return type of true means that the mesh should be adapted
    virtual bool queryAdaptationCriteria(int iteration) = 0;

    //! Method called by LOCA Solver to actually adapt the mesh
    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    virtual bool adaptMesh(const Teuchos::RCP<const Tpetra_Vector>& solution,
                           const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution) = 0;


    ///
    /// Each adapter must generate its list of valid parameters
    ///
    virtual Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const {
      return getGenericAdapterParams("Generic Adapter List");
    }

  protected:

    ///
    /// List of valid problem params common to all adapters, as
    /// a starting point for the specific  getValidAdaptaterParameters
    ///
    Teuchos::RCP<Teuchos::ParameterList>
    getGenericAdapterParams(std::string listname = "AdapterList") const;

    ///
    /// Configurable output stream, defaults to printing on proc=0
    ///
    Teuchos::RCP<Teuchos::FancyOStream> output_stream_;

    ///
    /// Adaptation parameters
    ///
    Teuchos::RCP<Teuchos::ParameterList> adapt_params_;

    ///
    /// Parameter library
    ///
    Teuchos::RCP<ParamLib> param_lib_;

    ///
    /// State Manager
    ///
    const Albany::StateManager& state_mgr_;

    ///
    /// Teuchos communicator
    ///
    Teuchos::RCP<const Teuchos_Comm> teuchos_comm_;

  private:

    //! Private to prohibit default or copy constructor
    AbstractAdapterT();
    AbstractAdapterT(const AbstractAdapterT&);

    //! Private to prohibit copying
    AbstractAdapterT& operator=(const AbstractAdapterT&);
};
}

#endif // AbstractAdaptationT_hpp
