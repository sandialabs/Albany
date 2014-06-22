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
#include "Phalanx_DataLayout.hpp"

namespace Adapt {

/*!
 * \brief This is a container class that deals with managing data values at the nodes of a mesh.
 *
 */
class NodalDataBlock {

  public:

    NodalDataBlock();

    //! Destructor
    virtual ~NodalDataBlock(){}

    void resizeLocalMap(const Teuchos::Array<GO>& local_nodeGIDs, const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

    void resizeOverlapMap(const Teuchos::Array<GO>& overlap_nodeGIDs, const Teuchos::RCP<const Teuchos::Comm<int> >& comm_);

    Teuchos::ArrayRCP<ST> getLocalNodeView(){ return local_node_view; }
    Teuchos::ArrayRCP<ST> getOverlapNodeView(){ return overlap_node_view; }

    Teuchos::ArrayRCP<const ST> getOverlapNodeConstView() const { return const_overlap_node_view; }
    Teuchos::ArrayRCP<const ST> getLocalNodeConstView() const { return const_local_node_view; }

    Teuchos::RCP<const Tpetra_BlockMap> getOverlapMap() const { return overlap_node_map; }
    Teuchos::RCP<const Tpetra_BlockMap> getLocalMap() const { return local_node_map; }

    void initializeVectors(ST value){overlap_node_vec->putScalar(value); local_node_vec->putScalar(value); }

    void initializeExport();

    void exportAddNodalDataBlock();

    void saveNodalDataState() const;

    void getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const;

    void registerState(const std::string &stateName, int ndofs);

    Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer(){ return nodeContainer; }

    void updateNodalGraph(const Teuchos::RCP<const Tpetra_CrsGraph>& nGraph)
         { nodalGraph = nGraph; }

    Teuchos::RCP<const Tpetra_CrsGraph> getNodalGraph()
         { return nodalGraph; }

  private:

    struct NodeFieldSize {

       std::string name;
       int offset;
       int ndofs;

    };

    typedef std::vector<NodeFieldSize> NodeFieldSizeVector;
    typedef std::map<const std::string, std::size_t> NodeFieldSizeMap;

    Teuchos::RCP<const Tpetra_BlockMap> overlap_node_map;
    Teuchos::RCP<const Tpetra_BlockMap> local_node_map;

    Teuchos::RCP<Tpetra_BlockMultiVector> overlap_node_vec;
    Teuchos::RCP<Tpetra_BlockMultiVector> local_node_vec;

    Teuchos::RCP<Tpetra_Import> importer;

    Teuchos::ArrayRCP<ST> overlap_node_view;
    Teuchos::ArrayRCP<ST> local_node_view;
    Teuchos::ArrayRCP<const ST> const_overlap_node_view;
    Teuchos::ArrayRCP<const ST> const_local_node_view;

    Teuchos::RCP<KokkosNode> node;

    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

    NodeFieldSizeVector nodeBlockLayout;
    NodeFieldSizeMap nodeBlockMap;

    Teuchos::RCP<const Tpetra_CrsGraph> nodalGraph;

    LO blocksize;

    bool mapsHaveChanged;

};


}

#endif // ADAPT_NODALDATABLOCK_HPP
