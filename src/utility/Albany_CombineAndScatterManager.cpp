#include "Albany_CombineAndScatterManager.hpp"

#include "Albany_CombineAndScatterManagerTpetra.hpp"
#include "Albany_TpetraThyraUtils.hpp"

namespace Albany
{

// Utility function that returns a concrete manager, depending on the return value
// of Albany::build_type().
Teuchos::RCP<CombineAndScatterManager>
createCombineAndScatterManager (const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tvs = getTpetraMap(owned,false);
  if (!tvs.is_null()) {
    // Check that the second vs is also of tpetra type. This time, throw if cast fails.
    tvs = getTpetraMap(overlapped,true);

    return Teuchos::rcp( new CombineAndScatterManagerTpetra(owned,overlapped) );
  }

  // TODO: add Epetra implementation

  // Dummy return value to silence compiler warning
  Teuchos::RCP<CombineAndScatterManager> manager;
  return manager;
}

} // namespace Albany
