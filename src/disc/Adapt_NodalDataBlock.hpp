//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ADAPT_NODALDATABLOCK_HPP
#define ADAPT_NODALDATABLOCK_HPP

//#include <boost/tuple/tuple.hpp>

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

    void saveNodalDataState();

    int getBlocksize(){ return blocksize; }

    void registerState(const std::string &stateName, 
			     int ndofs);

    Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer(){ return nodeContainer; }

    typedef std::vector<std::pair<std::string, int> > NodeFieldSizeVector;
//    typedef std::vector<boost::tuple<const std::string, std::size_t, std::size_t> > NodeFieldSizeVector;

  private:

    Teuchos::RCP<const Epetra_BlockMap> overlap_node_map;
    Teuchos::RCP<const Epetra_BlockMap> local_node_map;

    Teuchos::RCP<Epetra_Vector> overlap_node_vec;
    Teuchos::RCP<Epetra_Vector> local_node_vec;

    Teuchos::RCP<Epetra_Export> exporter;

    Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

    NodeFieldSizeVector nodeBlockLayout;

    int blocksize;

};


}

#endif // ADAPT_NODALDATABLOCK_HPP
