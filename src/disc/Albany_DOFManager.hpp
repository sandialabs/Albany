//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DOF_MANAGER_HPP
#define ALBANY_DOF_MANAGER_HPP

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_KokkosTypes.hpp"

#include "Panzer_DOFManager.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

class DOFManager : public panzer::DOFManager {
public:
  // Initializes DOF manager
  DOFManager (const Teuchos::RCP<panzer::ConnManager>& conn_mgr,
              const Teuchos::RCP<const Teuchos_Comm>& comm);

  void build ();

  const DualView<const int**>& elem_lids () const;

  const Teuchos::RCP<const GlobalLocalIndexer>& indexer () const;
  const Teuchos::RCP<const GlobalLocalIndexer>& ov_indexer () const;

  Teuchos::RCP<const Thyra_VectorSpace> vs    () const;
  Teuchos::RCP<const Thyra_VectorSpace> ov_vs () const;

  const std::vector<std::string>& parts_names () const { return m_parts_names; }

private:
  Teuchos::RCP<const Teuchos_Comm>          m_comm;
  std::vector<std::string>                  m_parts_names;

  Teuchos::RCP<const GlobalLocalIndexer>    m_indexer;
  Teuchos::RCP<const GlobalLocalIndexer>    m_ov_indexer;
  DualView<const int**>                     m_elem_lids;

  bool m_built = false;
};

} // namespace Albany

#endif // ALBANY_DOF_MANAGER_HPP
