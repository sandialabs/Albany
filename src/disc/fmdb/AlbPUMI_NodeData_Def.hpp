//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

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
AlbPUMI::NodeData<DataType, ArrayDim, traits>::resize(const Teuchos::RCP<const Tpetra_Map>& local_node_map_){

  local_node_map = local_node_map_;
  std::size_t total_size = local_node_map->getNodeNumElements() * nfield_dofs;
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
AlbPUMI::NodeData<DataType, ArrayDim, traits>::saveFieldBlock(const Teuchos::RCP<const Tpetra_BlockMultiVector>& overlap_node_vec,
     int offset){

  const Teuchos::RCP<const Tpetra_BlockMap>& overlap_node_map = overlap_node_vec->getBlockMap();
  Teuchos::ArrayRCP<const ST> const_overlap_node_view = overlap_node_vec->get1dView();

  // loop over all the nodes owned by this processor
  for(LO i = 0; i < local_node_map->getNodeNumElements(); i++)  {

    GO node_gid = local_node_map->getGlobalElement(i);
    LO local_block_id = overlap_node_map->getLocalBlockID(node_gid);
    // skip the node if it is not owned by me
    if(local_block_id == Teuchos::OrdinalTraits<LO>::invalid()) continue;
    LO block_start = overlap_node_map->getFirstLocalPointInLocalBlock(local_block_id);

    for(std::size_t j = 0; j < nfield_dofs; j++) // loop over the dofs at this node

      buffer[i * nfield_dofs + j] = const_overlap_node_view[block_start + offset + j];

  }
}

template<typename DataType, unsigned ArrayDim, class traits>
void
AlbPUMI::NodeData<DataType, ArrayDim, traits>::saveFieldVector(const Teuchos::RCP<const Tpetra_MultiVector>& overlap_node_vec,
     int offset){


  for(std::size_t j = 0; j < nfield_dofs; j++){ // loop over the dofs at this node

    Teuchos::ArrayRCP<const ST> const_overlap_node_view = overlap_node_vec->getVector(offset + j)->get1dView();

    // loop over all the nodes owned by this processor
    for(LO i = 0; i < local_node_map->getNodeNumElements(); i++)  {

      GO node_gid = local_node_map->getGlobalElement(i);

      buffer[i * nfield_dofs + j] = const_overlap_node_view[node_gid];

    }
  }
}
