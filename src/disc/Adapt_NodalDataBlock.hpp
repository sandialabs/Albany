//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ADAPT_NODALDATABLOCK_HPP
#define ADAPT_NODALDATABLOCK_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_AbstractNodeFieldContainer.hpp"

namespace Adapt {

/*!
 * \brief This is a container class that deals with managing data values at the nodes of a mesh.
 *
 */
class NodalDataBlock {

  public:

    NodalDataBlock(const Teuchos::RCP<Albany::NodeFieldContainer>& container_,
                   const Teuchos::RCP<const Teuchos_Comm>& comm_);

    //! Destructor
    virtual ~NodalDataBlock(){}

    void resizeLocalMap( std::size_t numGlobalNodes,
                         LO blocksize,
                         const std::vector<LO>& local_nodeGIDs);

    void resizeOverlapMap(const std::vector<GO>& overlap_nodeGIDs);

    Teuchos::ArrayRCP<ST> getOverlapNodeView(){ return overlap_node_view; }
    Teuchos::ArrayRCP<ST> getLocalNodeView(){ return local_node_view; }
    Teuchos::ArrayRCP<const ST> getOverlapNodeConstView() const { return const_overlap_node_view; }
    Teuchos::ArrayRCP<const ST> getLocalNodeConstView() const { return const_local_node_view; }

    Teuchos::RCP<const Tpetra_BlockMap> getOverlapMap() const { return overlap_node_map; }
    Teuchos::RCP<const Tpetra_BlockMap> getMap() const { return local_node_map; }

    void initializeVectors(ST value){overlap_node_vec->putScalar(value); local_node_vec->putScalar(value); }

    void initializeExport();

    void exportNodeDataArray(const std::string& field_name);

  private:

    Teuchos::RCP<const Tpetra_BlockMap> overlap_node_map;
    Teuchos::RCP<const Tpetra_BlockMap> local_node_map;

    Teuchos::RCP<Tpetra_BlockMultiVector> overlap_node_vec;
    Teuchos::RCP<Tpetra_BlockMultiVector> local_node_vec;

    Teuchos::RCP<Tpetra_Export> exporter;

    Teuchos::ArrayRCP<ST> overlap_node_view;
    Teuchos::ArrayRCP<ST> local_node_view;
    Teuchos::ArrayRCP<const ST> const_overlap_node_view;
    Teuchos::ArrayRCP<const ST> const_local_node_view;

    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

    //! Tpetra communicator and Kokkos node
    const Teuchos::RCP<const Teuchos_Comm> comm;
    Teuchos::RCP<KokkosNode> node;

    LO blocksize;
    std::size_t numGlobalNodes;

};


}

#endif // ADAPT_NODALDATABLOCK_HPP
