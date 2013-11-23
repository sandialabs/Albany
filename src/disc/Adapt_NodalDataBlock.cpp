//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBlock.hpp"

Adapt::NodalDataBlock::NodalDataBlock(const Teuchos::RCP<Albany::NodeFieldContainer>& container_,
                                      const Teuchos::RCP<const Teuchos_Comm>& comm_) :
  nodeContainer(container_),
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

  // Build the vector and accessors
  overlap_node_vec = Teuchos::rcp(new Tpetra_BlockMultiVector(overlap_node_map, 1));

  overlap_node_view = overlap_node_vec->get1dViewNonConst();
  const_overlap_node_view = overlap_node_vec->get1dView();

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

  // Build the vector and accessors
  local_node_vec = Teuchos::rcp(new Tpetra_BlockMultiVector(local_node_map, 1));

  local_node_view = local_node_vec->get1dViewNonConst();
  const_local_node_view = local_node_vec->get1dView();

}

void
Adapt::NodalDataBlock::initializeExport(){

 exporter = Teuchos::rcp(new Tpetra_Export(overlap_node_map->getPointMap(), local_node_map->getPointMap()));

}

void
Adapt::NodalDataBlock::exportNodeDataArray(const std::string& field_name){

 // Export the data from the local to overlapped decomposition
 local_node_vec->doExport(*overlap_node_vec, *exporter, Tpetra::ADD);

 LO numNodes = overlap_node_map->getNodeNumBlocks();

 // Divide the overlap field through by the weights
 // the weights are located in  overlap_node_2d_view[v][blocksize]

 for (std::size_t v=0; v < numNodes; ++v) 
   for(std::size_t k=0; k < blocksize - 1; ++k)
            overlap_node_view[v * blocksize + k] /= overlap_node_view[v * blocksize + blocksize - 1];

 // Store the overlapped vector data back in stk in the vector field "field_name"

 (*nodeContainer)[field_name]->saveField(overlap_node_vec);

#if 0
 stk::mesh::BucketArray<Field<double,Cartesian> > data_array(metadata->getField(field_name), bucket);

  const int num_vec_components = data_array.dimension(0);
  const int num_nodes_in_bucket = data_array.dimension(1);

  for(std::size_t i = 0; i < num_nodes_in_bucket; i++)  {

    //      const unsigned node_gid = bucket[i].identifier();
    const int node_gid = bucket[i].identifier() - 1;
    int node_lid = node_map->LID(node_gid);

    for(std::size_t j = 0; j < num_vec_components; j++)

      soln[getDOF(node_lid, offset + j)] = solution_array(j, i);

  }
#endif
}

 
