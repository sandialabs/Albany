//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionCullingStrategy.hpp"

#include "Epetra_BlockMap.h"

#include "Epetra_GatherAllV.hpp"

#include "Teuchos_Assert.hpp"

#include <algorithm>

namespace Albany {

class UniformSolutionCullingStrategy : public SolutionCullingStrategyBase {
public:
  explicit UniformSolutionCullingStrategy(int numValues);

  virtual Teuchos::Array<int> selectedGIDs(const Epetra_BlockMap &sourceMap) const;

private:
  int numValues_;
};

} // namespace Albany

Albany::UniformSolutionCullingStrategy::
UniformSolutionCullingStrategy(int numValues) :
  numValues_(numValues)
{
  // Nothing to do
}

Teuchos::Array<int>
Albany::UniformSolutionCullingStrategy::
selectedGIDs(const Epetra_BlockMap &sourceMap) const
{
  Teuchos::Array<int> allGIDs(sourceMap.NumGlobalElements());
  {
    const int ierr = Epetra::GatherAllV(
        sourceMap.Comm(),
        sourceMap.MyGlobalElements(), sourceMap.NumMyElements(),
        allGIDs.getRawPtr(), allGIDs.size());
    TEUCHOS_ASSERT(ierr == 0);
  }
  std::sort(allGIDs.begin(), allGIDs.end());

  Teuchos::Array<int> result(numValues_);
  const int stride = 1 + (allGIDs.size() - 1) / numValues_;
  for (int i = 0; i < numValues_; ++i) {
    result[i] = allGIDs[i * stride];
  }
  return result;
}


namespace Albany {

Teuchos::RCP<SolutionCullingStrategyBase>
createSolutionCullingStrategy(
    const Teuchos::RCP<const Application> &/*app*/,
    Teuchos::ParameterList &params)
{
  const int numValues = params.get("Num Values", 10);
  return Teuchos::rcp(new UniformSolutionCullingStrategy(numValues));
}

} // namespace Albany
