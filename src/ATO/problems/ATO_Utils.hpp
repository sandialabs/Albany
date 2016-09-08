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
#include "Albany_StateManager.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "Teuchos_VerboseObject.hpp"


namespace ATO {
  /*!
   * \brief Generic Functions to construct evaluators more succinctly
   */
  template<typename EvalT, typename Traits>
  class Utils {

   public:

    Utils(Teuchos::RCP<Albany::Layouts> dl_, int numDim_);

    void
    SaveCellStateField(
       PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &variableName,
       const std::string &elementBlockName,
       const Teuchos::RCP<PHX::DataLayout>& dataLayout);

    void 
    constructStressEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName, 
       std::string stressName, std::string strainName);

    void 
    constructBodyForceEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName, 
       std::string bodyForceName);

    void 
    constructResidualStressEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName, 
       std::string residForceName);

    void 
    constructBoundaryConditionEvaluators(
       const Teuchos::ParameterList& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName, 
       std::string boundaryForceName);

  private:

    Teuchos::RCP<Albany::Layouts> dl;
    int numDim;

  };
}

#endif 
