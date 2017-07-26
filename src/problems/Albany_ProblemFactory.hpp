//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PROBLEMFACTORY_HPP
#define ALBANY_PROBLEMFACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractProblem.hpp"

// Forward declarations.
namespace AAdapt { namespace rc { class Manager; } }

namespace Albany {

  /*!
   * \brief A factory class to instantiate AbstractProblem objects
   */
  class ProblemFactory {
  public:

    //! Default constructor
    ProblemFactory(const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
                   const Teuchos::RCP<ParamLib>& paramLib,
                   const Teuchos::RCP<const Teuchos::Comm<int> >& commT_);

    //! Destructor
    virtual ~ProblemFactory() {}

    virtual Teuchos::RCP<Albany::AbstractProblem>
    create();

    //! Set the ref config manager for use in certain problems.
    void setReferenceConfigurationManager(
      const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr);

  private:

    //! Private to prohibit copying
    ProblemFactory(const ProblemFactory&);

    //! Private to prohibit copying
    ProblemFactory& operator=(const ProblemFactory&);

  protected:

    //! Parameter list specifying what problem to create
    Teuchos::RCP<Teuchos::ParameterList> problemParams;

    //! Parameter list specifying what discretization to use.
    Teuchos::RCP<Teuchos::ParameterList> discretizationParams;

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;

    //! MPI Communicator
    Teuchos::RCP<const Teuchos::Comm<int> > commT;

    Teuchos::RCP<AAdapt::rc::Manager> rc_mgr;

  };

}

#endif // ALBANY_PROBLEMFACTORY_HPP
