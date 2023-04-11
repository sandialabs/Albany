//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTION_CULLING_STRATEGY_HPP
#define ALBANY_SOLUTION_CULLING_STRATEGY_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_DOFManager.hpp"
#include "Albany_CommTypes.hpp"

namespace Albany {

class SolutionCullingStrategyBase {
public:
  virtual void setup () {}

  virtual Teuchos::Array<GO> selectedGIDs (const Teuchos::RCP<const DOFManager>& source_dof_mgr) const = 0;

  virtual ~SolutionCullingStrategyBase()  = default;
};

class Application; // Forward declaration

//! Factory function
Teuchos::RCP<SolutionCullingStrategyBase>
createSolutionCullingStrategy(const Teuchos::RCP<const Application> &app,
                              Teuchos::ParameterList &params);

} // namespace Albany

#endif // ALBANY_SOLUTION_CULLING_STRATEGY_HPP
