#include "Albany_CombineAndScatterManager.hpp"

#include "Albany_CombineAndScatterManagerTpetra.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#ifdef ALBANY_EPETRA
#include "Albany_CombineAndScatterManagerEpetra.hpp"
#include "Albany_EpetraThyraUtils.hpp"
#endif

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
#ifdef ALBANY_EPETRA
    auto evs = getEpetraMap(owned, false);
    if (!evs.is_null()) {
      // Check that the second vs is also of epetra type. This time, throw if cast fails.
      evs = getEpetraMap(overlapped,true);

      manager = Teuchos::rcp( new CombineAndScatterManagerEpetra(owned,overlapped) );
    }
#endif
  }

  TEUCHOS_TEST_FOR_EXCEPTION (manager.is_null(), std::logic_error, "Error! We were not able to cast the input maps to any of the available concrete implementations (so far, only Epetra and Tpetra).\n");

  return manager;
}

} // namespace Albany
