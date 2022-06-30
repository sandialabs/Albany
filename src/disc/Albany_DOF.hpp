#ifndef ALBANY_DOF_HPP
#define ALBANY_DOF_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include <Panzer_DOFManager.hpp>
#include <Teuchos_RCP.hpp>

namespace Albany {

struct DOF {
  std::string mesh_part;

  // The dof_mgr is to be used for elem-wise operations
  Teuchos::RCP<panzer::DOFManager>        dof_mgr;

  // VectorSpaces are mostly to create (Multi)Vector's 
  Teuchos::RCP<const Thyra_VectorSpace>   vs;
  Teuchos::RCP<const Thyra_VectorSpace>   overlapped_vs;

  // Indexers are mostly useful for node-centric operations
  // that do not involve a loop over elements (e.g. Dirichlet)
  Teuchos::RCP<const GlobalLocalIndexer>  indexer;
  Teuchos::RCP<const GlobalLocalIndexer>  overlapped_indexer;

  // To access node information
  Teuchos::RCP<const DOF> node_dof;
};

} // namespace Albany

#endif // ALBANY_DOF_HPP
