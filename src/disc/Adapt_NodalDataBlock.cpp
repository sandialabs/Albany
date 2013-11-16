//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBlock.hpp"

Adapt::NodalDataBlock::NodalDataBlock(const Teuchos::RCP<const Teuchos_Comm>& comm_) :
  comm(comm_)
{

  //Create the Kokkos Node instance to pass into Tpetra::Map constructors.
  Teuchos::ParameterList kokkosNodeParams;
  node = Teuchos::rcp(new KokkosNode (kokkosNodeParams));

}

void
Adapt::NodalDataBlock::resizeOverlapMap(const std::vector<GO>& overlap_nodeGIDs){

  std::vector<GO> block_pt(overlap_nodeGIDs.size());
  std::vector<LO> block_sizes(overlap_nodeGIDs.size());

  for (LO i=0; i < overlap_nodeGIDs.size(); i++){
    block_sizes[i] = blocksize; // constant number of entries per node
    block_pt[i] = blocksize * overlap_nodeGIDs[i]; // multiply GID by blocksize
  }

  overlap_node_map = Teuchos::rcp(new Tpetra_BlockMap(overlap_nodeGIDs.size(),
                            Teuchos::arrayViewFromVector(overlap_nodeGIDs),
                            Teuchos::arrayViewFromVector(block_pt),
                            Teuchos::arrayViewFromVector(block_sizes),
                            Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
                            comm,
                            node));

}

void
Adapt::NodalDataBlock::resizeLocalMap(const std::vector<LO>& local_nodeGIDs){

  std::vector<GO> block_pt(local_nodeGIDs.size());
  std::vector<LO> block_sizes(local_nodeGIDs.size());

  for (LO i=0; i < local_nodeGIDs.size(); i++){
    block_sizes[i] = blocksize; // constant number of entries per node
    block_pt[i] = blocksize * local_nodeGIDs[i]; // multiply GID by blocksize
  }

  local_node_map = Teuchos::rcp(new Tpetra_BlockMap(local_nodeGIDs.size(),
                            Teuchos::arrayViewFromVector(local_nodeGIDs),
                            Teuchos::arrayViewFromVector(block_pt),
                            Teuchos::arrayViewFromVector(block_sizes),
                            Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
                            comm,
                            node));

}
