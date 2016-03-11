//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#include "Albany_STKNodeFieldContainer.hpp"
#include "Albany_BucketArray.hpp"

#ifdef ALBANY_SEACAS
#   include <stk_io/IossBridge.hpp>
#endif


#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include "Shards_Array.hpp"


Teuchos::RCP<Albany::AbstractNodeFieldContainer>
Albany::buildSTKNodeField(const std::string& name, const std::vector<PHX::DataLayout::size_type>& dim,
                          const Teuchos::RCP<stk::mesh::MetaData>& metaData,
                          const bool output)
{
  switch(dim.size()) {

  case 1: // scalar
    return Teuchos::rcp(new STKNodeField<double, 1>(name, dim, metaData, output));
    break;

  case 2: // vector
    return Teuchos::rcp(new STKNodeField<double, 2>(name, dim, metaData, output));
    break;

  case 3: // tensor
    return Teuchos::rcp(new STKNodeField<double, 3>(name, dim, metaData, output));
    break;

  default:
    TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Error: unexpected argument for dimension");
  }
}


template<typename DataType, unsigned ArrayDim, class traits>
Albany::STKNodeField<DataType, ArrayDim, traits>::
STKNodeField(const std::string& name_,
             const std::vector<PHX::DataLayout::size_type>& dims_,
             const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
             const bool output) :
  name(name_),
  dims(dims_),
  metaData(metaData_)
{
  //amb-leak Look into this later.
  node_field = traits_type::createField(name, dims, metaData_.get());

#ifdef ALBANY_SEACAS

  if(output)
     stk::io::set_field_role(*node_field, Ioss::Field::TRANSIENT);

#endif

}

template<typename DataType, unsigned ArrayDim, class traits>
void 
Albany::STKNodeField<DataType, ArrayDim, traits>::
saveFieldVector(const Teuchos::RCP<const Tpetra_MultiVector>& mv, int offset)
{
 // Iterate over the processor-visible nodes
 const stk::mesh::Selector select_owned_or_shared = metaData->locally_owned_part() | metaData->globally_shared_part();

 // Iterate over the overlap nodes by getting node buckets and iterating over each bucket.
 stk::mesh::BulkData& mesh = node_field->get_mesh();
 const stk::mesh::BucketVector& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, select_owned_or_shared);

 traits_type::saveFieldData(mv, all_elements, node_field, offset);
}

template<typename DataType, unsigned ArrayDim, class traits>
Albany::MDArray
Albany::STKNodeField<DataType, ArrayDim, traits>::getMDA(const stk::mesh::Bucket& buck){

  BucketArray<field_type> array(*node_field, buck);

  return array;

}
