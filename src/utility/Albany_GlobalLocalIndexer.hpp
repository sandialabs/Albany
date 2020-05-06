#ifndef ALBANY_GLOBAL_LOCAL_INDEXER_HPP
#define ALBANY_GLOBAL_LOCAL_INDEXER_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_CommTypes.hpp"

namespace Albany {

class GlobalLocalIndexer
{
public:

  GlobalLocalIndexer (const Teuchos::RCP<const Thyra_VectorSpace>& vs)
   : m_vs (vs)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (m_vs.is_null(), std::logic_error, "Error! Input vector space pointer is null.\n");
  }

  virtual ~GlobalLocalIndexer () = default;

  virtual GO getGlobalElement (const LO lid) const = 0;

  virtual LO getLocalElement (const GO gid) const = 0;

  virtual LO getNumLocalElements () const = 0;

  virtual GO getNumGlobalElements () const = 0;

  virtual Teuchos::RCP<const Teuchos_Comm> getComm () const = 0;

  virtual bool isLocallyOwnedElement (const GO gid) const = 0;

  virtual GO getMaxGlobalGID () const = 0;
  virtual GO getMaxLocalGID  () const = 0;

  Teuchos::RCP<const Thyra_VectorSpace> getVectorSpace () const { return m_vs; }

protected:

  Teuchos::RCP<const Thyra_VectorSpace>   m_vs;
};

// Create an indexer from a vector space
// WARNING: this is a COLLECTIVE operation. All ranks in the comm associated
//          with the vector space MUST call this function.
Teuchos::RCP<const GlobalLocalIndexer>
createGlobalLocalIndexer (const Teuchos::RCP<const Thyra_VectorSpace>& vs);

} // namespace Albany

#endif // ALBANY_GLOBAL_LOCAL_INDEXER_HPP
