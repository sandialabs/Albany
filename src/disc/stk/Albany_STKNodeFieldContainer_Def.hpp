//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STKNodeFieldContainer.hpp"


#ifdef ALBANY_SEACAS
#   include <stk_io/IossBridge.hpp>
#endif


#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/fem/FEMMetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include "Shards_Array.hpp"


Teuchos::RCP<Albany::AbstractNodeFieldContainer>
Albany::buildSTKNodeField(const std::string& name, const std::vector<int>& dim,
                          stk_classic::mesh::fem::FEMMetaData* metaData, stk_classic::mesh::BulkData* bulkData,
                          const bool output){

  switch(dim.size()){

  case 1: // scalar
    return Teuchos::rcp(new STKNodeField<double, 1>(name, dim, metaData, bulkData, output));
    break;

  case 2: // vector
    return Teuchos::rcp(new STKNodeField<double, 2>(name, dim, metaData, bulkData, output));
    break;

  case 3: // tensor
    return Teuchos::rcp(new STKNodeField<double, 3>(name, dim, metaData, bulkData, output));
    break;

  default:
    TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Error: unexpected argument for dimension");
  }
}


template<typename DataType, unsigned ArrayDim, class traits>
Albany::STKNodeField<DataType, ArrayDim, traits>::STKNodeField(const std::string& name_,
     const std::vector<int>& dims_, stk_classic::mesh::fem::FEMMetaData* metaData_, stk_classic::mesh::BulkData* bulkData_,
     const bool output) :
  name(name_),
  dims(dims_),
  metaData(metaData_),
  bulkData(bulkData_)
{

  node_field = traits_type::createField(name, dims, metaData_);

#ifdef ALBANY_SEACAS

  if(output)
     stk_classic::io::set_field_role(*node_field, Ioss::Field::TRANSIENT);

#endif

}

template<typename DataType, unsigned ArrayDim, class traits>
void
Albany::STKNodeField<DataType, ArrayDim, traits>::saveField(const Teuchos::RCP<const Epetra_Vector>& block_mv,
        int offset, int blocksize){

 // Iterate over the processor-visible nodes
 stk_classic::mesh::Selector select_owned_or_shared = metaData->locally_owned_part() | metaData->globally_shared_part();

 // Iterate over the overlap nodes by getting node buckets and iterating over each bucket.
 stk_classic::mesh::BucketVector all_elements;
 stk_classic::mesh::get_buckets(select_owned_or_shared, bulkData->buckets(metaData->node_rank()), all_elements);

 traits_type::saveFieldData(block_mv, all_elements, node_field, offset, blocksize);

}

template<typename DataType, unsigned ArrayDim, class traits>
Albany::MDArray
Albany::STKNodeField<DataType, ArrayDim, traits>::getMDA(const stk_classic::mesh::Bucket& buck){

 stk_classic::mesh::BucketArray<field_type> array(*node_field, buck);

 return array;

}
