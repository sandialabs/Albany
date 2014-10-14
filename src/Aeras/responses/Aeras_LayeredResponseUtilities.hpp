/*
 * Aeras_ResponseUtilities.hpp
 *
 *  Created on: Jul 30, 2014
 *      Author: swbova
 */

#ifndef AERAS_RESPONSEUTILITIES_HPP_
#define AERAS_RESPONSEUTILITIES_HPP_

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Aeras/problems/Aeras_Layouts.hpp"

#include "Phalanx.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

  template<typename EvalT, typename Traits>
  class LayeredResponseUtilities {

    public:

    LayeredResponseUtilities(Teuchos::RCP<Aeras::Layouts> dl);

    //! Utility for parsing response requests and creating response field manager
    Teuchos::RCP<const PHX::FieldTag>
    constructResponses(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      Teuchos::ParameterList& responseList,
      Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem,
      Albany::StateManager& stateMgr);

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
    Teuchos::RCP<Aeras::Layouts> get_dl() { return dl;};

   private:

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Aeras::Layouts> dl;
  };
}

#endif /* AERAS_RESPONSEUTILITIES_HPP_ */
