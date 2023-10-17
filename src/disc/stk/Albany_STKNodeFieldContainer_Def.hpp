//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STKNodeFieldContainer.hpp"

#ifdef ALBANY_SEACAS
#   include <stk_io/IossBridge.hpp>
#endif


#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include "Shards_Array.hpp"

namespace Albany
{

Teuchos::RCP<AbstractNodeFieldContainer>
buildSTKNodeField(const std::string& name,
                  const std::vector<PHX::DataLayout::size_type>& dim,
                  const Teuchos::RCP<stk::mesh::MetaData>& metaData,
                  const bool output)
{
  Teuchos::RCP<AbstractNodeFieldContainer> nfc;
  switch(dim.size()) {
    case 1: // scalar
      nfc = Teuchos::rcp(new STKNodeField<double, 1>(name, dim, metaData, output));
      break;

    case 2: // vector
      nfc = Teuchos::rcp(new STKNodeField<double, 2>(name, dim, metaData, output));
      break;

    case 3: // tensor
      nfc = Teuchos::rcp(new STKNodeField<double, 3>(name, dim, metaData, output));
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Error: unexpected argument for dimension");
  }
  return nfc;
}

template<typename DataType, unsigned ArrayDim, class traits>
STKNodeField<DataType, ArrayDim, traits>::
STKNodeField(const std::string& name_,
             const std::vector<PHX::DataLayout::size_type>& dims_,
             const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
             const bool output)
 : name(name_)
 , dims(dims_)
 , metaData(metaData_)
{
  //amb-leak Look into this later.
  node_field = traits_type::createField(name, dims, metaData_.get());

#ifdef ALBANY_SEACAS
  if(output) {
     stk::io::set_field_role(*node_field, Ioss::Field::TRANSIENT);
  }
#endif
}

template<typename DataType, unsigned ArrayDim, class traits>
StateView STKNodeField<DataType, ArrayDim, traits>::
getMDA(const stk::mesh::Bucket& buck){
  auto data = reinterpret_cast<double*>(buck.field_data_location(*node_field));
  StateView dv;

  int dim0,dim1;
  switch (ArrayDim) {
    case 1:
      dv.reset_from_host_ptr(data,buck.size());
      break;
    case 2:
      dim0 = stk::mesh::field_scalars_per_entity(*node_field,buck);
      dv.reset_from_host_ptr(data,buck.size(),dim0);
      break;
    case 3:
      dim0 = stk::mesh::field_extent0_per_entity(*node_field,buck);
      dim1 = stk::mesh::field_extent1_per_entity(*node_field,buck);
      dv.reset_from_host_ptr(data,buck.size(),dim0,dim1);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
          "Invalid/unsupported array dimension (" << ArrayDim << ")\n");
  }

  return dv;
}

} // namespace Albany
