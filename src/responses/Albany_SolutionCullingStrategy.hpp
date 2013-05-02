//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTIONCULLINGSTRATEGY_HPP
#define ALBANY_SOLUTIONCULLINGSTRATEGY_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_BlockMap;

namespace Albany {

class SolutionCullingStrategyBase {
public:
  virtual void setup() {}

  virtual Teuchos::Array<int> selectedGIDs(const Epetra_BlockMap &sourceMap) const = 0;

  virtual ~SolutionCullingStrategyBase() {}
};

class Application; // Forward declaration

//! Factory function
Teuchos::RCP<SolutionCullingStrategyBase>
createSolutionCullingStrategy(
    const Teuchos::RCP<const Application> &app,
    Teuchos::ParameterList &params);

} // namespace Albany

#endif // ALBANY_SOLUTIONCULLINGSTRATEGY_HPP
