//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_UTILS_HPP
#define ATO_UTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "Teuchos_VerboseObject.hpp"


namespace ATO {
  /*!
   * \brief Generic Functions to construct evaluators more succinctly
   */
  template<typename EvalT, typename Traits>
  class Utils {

   public:

    Utils(Teuchos::RCP<Albany::Layouts> dl);

    void
    SaveCellStateField(
       PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &variableName,
       const std::string &elementBlockName,
       const Teuchos::RCP<PHX::DataLayout>& dataLayout, int numDim);

  private:

    Teuchos::RCP<Albany::Layouts> dl;

  };
}

#endif 
