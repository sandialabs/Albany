#include "Albany_CombineAndScatterManager.hpp"

#include "Albany_CombineAndScatterManagerTpetra.hpp"
#include "Albany_Utils.hpp"

namespace Albany
{

// Utility function that returns a concrete manager, depending on the return value
// of Albany::build_type().
Teuchos::RCP<CombineAndScatterManager>
createCombineAndScatterManager (const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
{
  const BuildType bt = build_type();

  Teuchos::RCP<CombineAndScatterManager> manager;
  switch (bt) {
    case BuildType::Tpetra:
      manager = Teuchos::rcp( new CombineAndScatterManagerTpetra(owned,overlapped) );
      break;
    case BuildType::Epetra:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Misssing an Epetra implementation for CombineAndScatterManager.\n");
      break;
    case BuildType::None:
      break;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Albany build type is set to None. Initialize Albany build type first.\n");
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unknown Albany build type. Please, contact developers.\n");
  }

  return manager;
}

} // namespace Albany
