#ifndef ALBANY_GLOBAL_LOCAL_INDEXER_TPETRA_HPP
#define ALBANY_GLOBAL_LOCAL_INDEXER_TPETRA_HPP

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_TpetraTypes.hpp"

namespace Albany {

class GlobalLocalIndexerTpetra : public GlobalLocalIndexer
{
public:

  GlobalLocalIndexerTpetra (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                            const Teuchos::RCP<const Tpetra_Map>& tmap)
   : GlobalLocalIndexer(vs)
   , m_tmap (tmap)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (tmap.is_null(), std::logic_error, "Error! Input tpetra map pointer is null.\n");
  }

  GO getGlobalElement (const LO lid) const { return m_tmap->getGlobalElement(lid); }

  LO getLocalElement  (const GO gid) const { return m_tmap->getLocalElement(gid);  }

  LO getNumLocalElements  () const { return m_tmap->getNodeNumElements();   }
  GO getNumGlobalElements () const { return m_tmap->getGlobalNumElements(); }

  GO getMaxGlobalGID () const { return m_tmap->getMaxGlobalIndex(); }
  GO getMaxLocalGID  () const { return m_tmap->getMaxLocalIndex(); }

  bool isLocallyOwnedElement (const GO gid) const { return m_tmap->isNodeGlobalElement(gid); }

  Teuchos::RCP<const Teuchos_Comm> getComm () const { return m_tmap->getComm(); }

protected:

  Teuchos::RCP<const Tpetra_Map>   m_tmap;
};


} // namespace Albany

#endif // ALBANY_GLOBAL_LOCAL_INDEXER_TPETRA_HPP
