//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ADAPT_NODALDATABLOCK_HPP
#define ADAPT_NODALDATABLOCK_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_DataTypes.hpp"

namespace Adapt {

/*!
 * \brief This is a container class that deals with managing data values at the nodes of a mesh.
 *
 */
class NodalDataBlock {

  public:

    NodalDataBlock(const Teuchos::RCP<const Teuchos_Comm>& comm_);

    //! Destructor
    virtual ~NodalDataBlock(){}

    void resizeOverlapMap(const std::vector<GO>& overlap_nodeGIDs);
    void resizeLocalMap(const std::vector<LO>& local_nodeGIDs);

    void setBlockSize(LO blocksize_){ blocksize = blocksize_; }

  private:

    Teuchos::RCP<const Tpetra_BlockMap> overlap_node_map;
    Teuchos::RCP<const Tpetra_BlockMap> local_node_map;

    //! Tpetra communicator and Kokkos node
    const Teuchos::RCP<const Teuchos_Comm> comm;
    Teuchos::RCP<KokkosNode> node;

    LO blocksize;


};


}

#endif // ADAPT_NODALDATABLOCK_HPP
