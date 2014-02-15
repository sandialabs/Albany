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

    void resizeLocalMap(const std::vector<int>& local_nodeGIDs, const Epetra_Comm& comm);

    void resizeOverlapMap(const std::vector<int>& overlap_nodeGIDs, const Epetra_Comm& comm);

    Teuchos::RCP<Epetra_Vector> getOverlapNodeVec(){ return overlap_node_vec; }
    Teuchos::RCP<Epetra_Vector> getLocalNodeVec(){ return local_node_vec; }

    Teuchos::RCP<const Epetra_BlockMap> getOverlapMap() const { return overlap_node_map; }
    Teuchos::RCP<const Epetra_BlockMap> getLocalMap() const { return local_node_map; }

    void initializeVectors(double value){overlap_node_vec->PutScalar(value); local_node_vec->PutScalar(value); }

    void initializeExport();

    void exportAddNodalDataBlock();

    void saveNodalDataState() const;

    void saveEpetraNodalDataVector(const std::string& name, const Teuchos::RCP<const Epetra_Vector>& overlap_node_vec,
            int offset, int blocksize) const;

    int getBlocksize(){ return blocksize; }

    void getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const;

    void registerState(const std::string &stateName, int ndofs);

    Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer(){ return nodeContainer; }


  private:

    struct NodeFieldSize {

       std::string name;
       int offset;
       int ndofs;

    };

    typedef std::vector<NodeFieldSize> NodeFieldSizeVector;
    typedef std::map<const std::string, std::size_t> NodeFieldSizeMap;

    Teuchos::RCP<const Epetra_BlockMap> overlap_node_map;
    Teuchos::RCP<const Epetra_BlockMap> local_node_map;

    Teuchos::RCP<Epetra_Vector> overlap_node_vec;
    Teuchos::RCP<Epetra_Vector> local_node_vec;

    Teuchos::RCP<Epetra_Import> importer;

    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

    NodeFieldSizeVector nodeBlockLayout;
    NodeFieldSizeMap nodeBlockMap;

    int blocksize;

    bool mapsHaveChanged;

};


}

#endif // ADAPT_NODALDATABLOCK_HPP
