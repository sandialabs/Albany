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
  Teuchos::RCP<CombineAndScatterManager> manager;

  // Allow failure, since we don't know what the underlying linear algebra is
  auto tvs = getTpetraMap(owned,false);
  if (!tvs.is_null()) {
    // Check that the second vs is also of tpetra type. This time, throw if cast fails.
    tvs = getTpetraMap(overlapped,true);

    manager = Teuchos::rcp( new CombineAndScatterManagerTpetra(owned,overlapped) );
  } else {
    // TODO: add Epetra implementation
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! So far, only the Tpetra implementation is available for the CAS manager.\n");
  }

  return manager;
}

} // namespace Albany
