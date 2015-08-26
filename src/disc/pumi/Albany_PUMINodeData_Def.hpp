//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMINodeData.hpp"

Albany::PUMINodeMetaData::PUMINodeMetaData (
  const std::string& name, const bool output,
  const std::vector<PHX::DataLayout::size_type>& dims)
  : name(name), output(output), dims(dims)
{
  nfield_dofs = 1;
  // Multiply it by the number of dofs per node.
  for (std::size_t i = 1; i < this->dims.size(); i++)
    nfield_dofs *= dims[i];
}

int Albany::PUMINodeMetaData::ndims () const {
  if (dims.size() == 3) return 2;
  else if (dims.size() == 2) return dims[1] == 1 ? 0 : 1;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "dims[1] is not right");
}

Teuchos::RCP<Albany::AbstractNodeFieldContainer>
Albany::buildPUMINodeField(const std::string& name, const std::vector<PHX::DataLayout::size_type>& dim, const bool output){

  switch(dim.size()){

  case 1: // scalar
    return Teuchos::rcp(new PUMINodeData<double, 1>(name, dim, output));
    break;

  case 2: // vector
    return Teuchos::rcp(new PUMINodeData<double, 2>(name, dim, output));
    break;

  case 3: // tensor
    return Teuchos::rcp(new PUMINodeData<double, 3>(name, dim, output));
    break;

  default:
    TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Error: unexpected argument for dimension");
  }
}


template<typename DataType, unsigned ArrayDim, class traits>
Albany::PUMINodeData<DataType, ArrayDim, traits>::PUMINodeData(const std::string& name_,
                                const std::vector<PHX::DataLayout::size_type>& dims_, const bool output_) :
  PUMINodeDataBase<DataType>(name_, output_, dims_),
  beginning_index(0)
{}

template<typename DataType, unsigned ArrayDim, class traits>
void
Albany::PUMINodeData<DataType, ArrayDim, traits>::resize(const Teuchos::RCP<const Tpetra_Map>& local_node_map_){

  local_node_map = local_node_map_;
  std::size_t total_size = local_node_map->getNodeNumElements() * this->nfield_dofs;
  this->buffer.resize(total_size);

  beginning_index = 0;

}

template<typename DataType, unsigned ArrayDim, class traits>
Albany::MDArray
Albany::PUMINodeData<DataType, ArrayDim, traits>::getMDA(const std::vector<apf::Node>& buck){

  unsigned numNodes = buck.size(); // Total size starts at the number of nodes in the workset

  field_type the_array = traits_type::buildArray(&this->buffer[beginning_index], numNodes, this->dims);

  beginning_index += numNodes * this->nfield_dofs;

  return the_array;

}

template<typename DataType, unsigned ArrayDim, class traits>
void
Albany::PUMINodeData<DataType, ArrayDim, traits>::saveFieldVector(const Teuchos::RCP<const Tpetra_MultiVector>& overlap_node_vec,
     int offset){

  const Teuchos::RCP<const Tpetra_Map>&
    overlap_node_map = overlap_node_vec->getMap();

  for(std::size_t j = 0; j < this->nfield_dofs; j++){ // loop over the dofs at this node

    Teuchos::ArrayRCP<const ST> const_overlap_node_view = overlap_node_vec->getVector(offset + j)->get1dView();

    // loop over all the nodes owned by this processor
    for(LO i = 0; i < local_node_map->getNodeNumElements(); i++)  {

      const GO node_gid = local_node_map->getGlobalElement(i);
      const LO overlap_node_lid = overlap_node_map->getLocalElement(node_gid);

      this->buffer[i * this->nfield_dofs + j] = const_overlap_node_view[overlap_node_lid];

    }
  }
}
