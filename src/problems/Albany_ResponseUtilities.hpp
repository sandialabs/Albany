//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_RESPONSEUTILITIES_HPP
#define ALBANY_RESPONSEUTILITIES_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_ProblemUtils.hpp"

#include "Phalanx.hpp"


//! Code Base for Quantum Device Simulation Tools LDRD
namespace Albany {

class StateManager;

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */

  template<typename EvalT, typename Traits>
  class ResponseUtilities {

    public:

    ResponseUtilities(Teuchos::RCP<Albany::Layouts> dl);
    ResponseUtilities(const std::map<std::string,Teuchos::RCP<Albany::Layouts>>& dls);

    //! Utility for parsing response requests and creating response field manager
    Teuchos::RCP<const PHX::FieldTag>
    constructResponses(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      Teuchos::ParameterList& responseList,
      Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem,
      Albany::StateManager& stateMgr,
      // Optionally provide the MeshSpecsStruct. This is relevant only if the
      // response function needs to know whether there are separate field
      // managers for each element block. We can't use an RCP here because at
      // the caller's level meshSpecs is a raw ref, so ownership is unknown.
      const Albany::MeshSpecsStruct* meshSpecs = NULL);

    //! Utility for parsing response requests and creating response field manager
    //! (Convenience overload in the absence of parameters list from problem)
    Teuchos::RCP<const PHX::FieldTag>
    constructResponses(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      Teuchos::ParameterList& responseList,
      Albany::StateManager& stateMgr) {
      return constructResponses(fm0, responseList, Teuchos::null, stateMgr);
    }

    //! Accessor
    Teuchos::RCP<Albany::Layouts> get_dl() { return dl;};

   private:

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;
    std::map<std::string,Teuchos::RCP<Albany::Layouts>> dls;  // Different sides may have different layouts (b/c different cubatures)
  };
}

#endif
