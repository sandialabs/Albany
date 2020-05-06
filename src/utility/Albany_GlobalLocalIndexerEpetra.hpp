#ifndef ALBANY_GLOBAL_LOCAL_INDEXER_EPETRA_HPP
#define ALBANY_GLOBAL_LOCAL_INDEXER_EPETRA_HPP

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_EpetraThyraUtils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_Macros.hpp"
#include "Teuchos_OrdinalTraits.hpp"

namespace Albany {

class GlobalLocalIndexerEpetra : public GlobalLocalIndexer
{
public:

  GlobalLocalIndexerEpetra (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                            const Teuchos::RCP<const Epetra_BlockMap>& emap)
   : GlobalLocalIndexer(vs)
   , m_emap (emap)
  {
    m_comm = createTeuchosCommFromEpetraComm(emap->Comm());
    TEUCHOS_TEST_FOR_EXCEPTION (emap.is_null(), std::logic_error, "Error! Input epetra map pointer is null.\n");
  }

  virtual GO getGlobalElement (const LO lid) const { return m_emap->GID(lid); }

  virtual LO getLocalElement (const GO gid) const {
    ALBANY_EXPECT(gid<=static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max()),
                 "Error in getLocalElement! Input gid exceed Epetra_GO ranges.\n");

    return m_emap->LID(static_cast<Epetra_GO>(gid));
  }

  bool isLocallyOwnedElement (const GO gid) const {
    ALBANY_EXPECT(gid<=static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max()),
                 "Error in getLocalElement! Input gid exceed Epetra_GO ranges.\n");
    return m_emap->MyGID(static_cast<Epetra_GO>(gid));
  }

  LO getNumLocalElements  () const { return m_emap->NumMyElements();     }
  GO getNumGlobalElements () const { return m_emap->NumGlobalElements(); }

  GO getMaxGlobalGID () const { return m_emap->MaxAllGID(); }
  GO getMaxLocalGID  () const { return m_emap->MaxMyGID(); }

  Teuchos::RCP<const Teuchos_Comm> getComm () const { return m_comm; }

protected:

  Teuchos::RCP<const Epetra_BlockMap>   m_emap;
  Teuchos::RCP<const Teuchos_Comm>      m_comm;
};

} // namespace Albany

#endif // ALBANY_GLOBAL_LOCAL_INDEXER_EPETRA_HPP
