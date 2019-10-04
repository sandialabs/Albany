//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_PUMINodeData.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

namespace Albany
{

PUMINodeMetaData::
PUMINodeMetaData (const std::string& name, const bool output,
                  const std::vector<PHX::DataLayout::size_type>& dims)
 : name(name)
 , output(output)
 , dims(dims)
{
  nfield_dofs = 1;
  // Multiply it by the number of dofs per node.
  for (std::size_t i = 1; i < this->dims.size(); i++)
    nfield_dofs *= dims[i];
}

int PUMINodeMetaData::ndims () const {
  if (dims.size() == 3) return 2;
  else if (dims.size() == 2) return dims[1] == 1 ? 0 : 1;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "dims[1] is not right");
}

Teuchos::RCP<AbstractNodeFieldContainer>
buildPUMINodeField(const std::string& name,
                   const std::vector<PHX::DataLayout::size_type>& dim,
                   const bool output)
{
  Teuchos::RCP<AbstractNodeFieldContainer> nfc;
  switch(dim.size()) {
    case 1: // scalar
      nfc = Teuchos::rcp(new PUMINodeData<double, 1>(name, dim, output));
      break;
    case 2: // vector
      nfc = Teuchos::rcp(new PUMINodeData<double, 2>(name, dim, output));
      break;
    case 3: // tensor
      nfc = Teuchos::rcp(new PUMINodeData<double, 3>(name, dim, output));
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Error: unexpected argument for dimension");
  }
  return nfc;
}

template<typename DataType, unsigned ArrayDim, class traits>
PUMINodeData<DataType, ArrayDim, traits>::
PUMINodeData(const std::string& name_,
             const std::vector<PHX::DataLayout::size_type>& dims_,
             const bool output_)
 : PUMINodeDataBase<DataType>(name_, output_, dims_)
 , beginning_index(0)
{
  // Nothing to do here
}

template<typename DataType, unsigned ArrayDim, class traits>
void PUMINodeData<DataType, ArrayDim, traits>::
resize(const Teuchos::RCP<const Thyra_VectorSpace> local_node_vs)
{
  m_local_node_vs = getSpmdVectorSpace(local_node_vs);

  std::size_t total_size = m_local_node_vs->localSubDim() * this->nfield_dofs;
  this->buffer.resize(total_size);

  beginning_index = 0;
}

template<typename DataType, unsigned ArrayDim, class traits>
MDArray PUMINodeData<DataType, ArrayDim, traits>::
getMDA(const std::vector<apf::Node>& buck)
{
  unsigned numNodes = buck.size(); // Total size starts at the number of nodes in the workset

  field_type the_array = traits_type::buildArray(&this->buffer[beginning_index], numNodes, this->dims);

  beginning_index += numNodes * this->nfield_dofs;

  return the_array;
}

template<typename DataType, unsigned ArrayDim, class traits>
void PUMINodeData<DataType, ArrayDim, traits>::
saveFieldVector(const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec,
                int offset)
{
  const auto& overlap_node_vs = overlap_node_vec->range();

  auto node_vs_indexer    = Albany::createGlobalLocalIndexer(m_local_node_vs);
  auto ov_node_vs_indexer = Albany::createGlobalLocalIndexer(overlap_node_vs);
  // loop over the dofs at this node
  for(int j=0; j<this->nfield_dofs; ++j) {
    auto const_overlap_node_view = getLocalData(overlap_node_vec->col(offset + j));

    // loop over all the nodes owned by this processor
    for(LO i=0; i<m_local_node_vs->localSubDim(); ++i)  {
      const GO node_gid         = node_vs_indexer->getGlobalElement(i);
      const LO overlap_node_lid = ov_node_vs_indexer->getLocalElement(node_gid);

      this->buffer[i * this->nfield_dofs + j] = const_overlap_node_view[overlap_node_lid];
    }
  }
}

} // namespace Albany
