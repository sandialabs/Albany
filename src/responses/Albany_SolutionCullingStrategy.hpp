//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLUTIONCULLINGSTRATEGY_HPP
#define ALBANY_SOLUTIONCULLINGSTRATEGY_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_DataTypes.hpp" 

#if defined(ALBANY_EPETRA)
class Epetra_BlockMap;
#endif


namespace Albany {

class SolutionCullingStrategyBase {
public:
#if defined(ALBANY_EPETRA)
  virtual void setup() {}
#endif
  virtual void setupT() {}

#if defined(ALBANY_EPETRA)
  virtual Teuchos::Array<int> selectedGIDs(const Epetra_BlockMap &sourceMap) const = 0;
#endif
  virtual Teuchos::Array<GO> selectedGIDsT(Teuchos::RCP<const Tpetra_Map> sourceMapT) const = 0;

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
