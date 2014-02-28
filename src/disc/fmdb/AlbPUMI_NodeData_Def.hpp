//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AlbPUMI_NodeData.hpp"

Teuchos::RCP<Albany::AbstractNodeFieldContainer>
AlbPUMI::buildPUMINodeField(const std::string& name, const std::vector<int>& dim, const bool output){

  switch(dim.size()){

  case 1: // scalar
    return Teuchos::rcp(new NodeData<double, 1>(name, dim, output));
    break;

  case 2: // vector
    return Teuchos::rcp(new NodeData<double, 2>(name, dim, output));
    break;

  case 3: // tensor
    return Teuchos::rcp(new NodeData<double, 3>(name, dim, output));
    break;

  default:
    TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Error: unexpected argument for dimension");
  }
}


template<typename DataType, unsigned ArrayDim, class traits>
AlbPUMI::NodeData<DataType, ArrayDim, traits>::NodeData(const std::string& name_,
                                const std::vector<int>& dim, const bool output_) :
  name(name_),
  output(output_),
  dims(dim),
  nfield_dofs(1),
  beginning_index(0)
{

  for(std::size_t i = 1; i < dims.size(); i++) // multiply it by the number of dofs per node

    nfield_dofs *= dims[i];

}

template<typename DataType, unsigned ArrayDim, class traits>
void
AlbPUMI::NodeData<DataType, ArrayDim, traits>::resize(const Teuchos::RCP<const Epetra_Map>& local_node_map_){

  local_node_map = local_node_map_;
  std::size_t total_size = local_node_map->NumMyElements() * nfield_dofs;
  buffer.resize(total_size);

  beginning_index = 0;

}

template<typename DataType, unsigned ArrayDim, class traits>
Albany::MDArray
AlbPUMI::NodeData<DataType, ArrayDim, traits>::getMDA(const std::vector<apf::Node>& buck){

  unsigned numNodes = buck.size(); // Total size starts at the number of nodes in the workset

  field_type the_array = traits_type::buildArray(&buffer[beginning_index], numNodes, dims);

  beginning_index += numNodes * nfield_dofs;

  return the_array;

}

template<typename DataType, unsigned ArrayDim, class traits>
void
AlbPUMI::NodeData<DataType, ArrayDim, traits>::saveField(const Teuchos::RCP<const Epetra_Vector>& overlap_node_vec,
    int offset, int blocksize){

  const Epetra_BlockMap& overlap_node_map = overlap_node_vec->Map();
  if(blocksize < 0)
    blocksize = overlap_node_map.ElementSize();

  // loop over all the nodes owned by this processor
  for(std::size_t i = 0; i < local_node_map->NumMyElements(); i++)  {

    int node_gid = local_node_map->GID(i);
    int local_node = overlap_node_map.LID(node_gid); // the current node's location in the block map
    if(local_node < 0) continue; // not on this processor
    int block_start = local_node * blocksize; // there are blocksize dofs per node in the block vector

    for(std::size_t j = 0; j < nfield_dofs; j++) // loop over the dofs at this node

      buffer[i * nfield_dofs + j] = (*overlap_node_vec)[block_start + offset + j];

  }
}
