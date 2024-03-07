#include "Albany_GlobalLocalIndexer.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_GlobalLocalIndexerTpetra.hpp"

namespace Albany
{

Teuchos::RCP<const GlobalLocalIndexer>
createGlobalLocalIndexer (const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  Teuchos::RCP<const GlobalLocalIndexer> indexer;

  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap (vs, false);
  if (!tmap.is_null()) {
    indexer = Teuchos::rcp(new GlobalLocalIndexerTpetra(vs,tmap));
  }

  TEUCHOS_TEST_FOR_EXCEPTION (indexer.is_null(), std::runtime_error,
                              "Error! Could not cast the input vector space to any of the supported concrete types.\n");

  return indexer;
}

} // namespace Albany
