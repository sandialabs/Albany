//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STK_NODE_FIELD_CONTAINER_HPP
#define ALBANY_STK_NODE_FIELD_CONTAINER_HPP

#include "Albany_AbstractNodeFieldContainer.hpp"
#include "Albany_BucketArray.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_StateInfoStruct.hpp" // For MDArray

#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace Albany {

/*!
 * \brief Abstract interface for an STK NodeField container
 *
 */
class AbstractSTKNodeFieldContainer : public AbstractNodeFieldContainer
{
public:

  AbstractSTKNodeFieldContainer () = default;
  virtual ~AbstractSTKNodeFieldContainer () = default;

  virtual MDArray getMDA(const stk::mesh::Bucket& buck) = 0;
};

Teuchos::RCP<AbstractNodeFieldContainer>
buildSTKNodeField(const std::string& name, const std::vector<PHX::DataLayout::size_type>& dim,
                  const Teuchos::RCP<stk::mesh::MetaData>& metaData,
                  const bool output);

// Helper class for NodeData
template<typename DataType, unsigned ArrayDim>
struct NodeData_Traits { };

template<typename DataType, unsigned ArrayDim,
         class traits = NodeData_Traits<DataType, ArrayDim> >
class STKNodeField : public AbstractSTKNodeFieldContainer
{
public:

  //! Type of traits class being used
  typedef traits traits_type;

  //! Define the field type
  typedef typename traits_type::field_type field_type;


  STKNodeField(const std::string& name, const std::vector<PHX::DataLayout::size_type>& dim,
               const Teuchos::RCP<stk::mesh::MetaData>& metaData,
               const bool output = false);

  virtual ~STKNodeField () = default;

  void saveFieldVector(const Teuchos::RCP<const Thyra_MultiVector>& mv, int offset) override;

  MDArray getMDA(const stk::mesh::Bucket& buck) override;

private:

  std::string name;      // Name of data field
  field_type *node_field;  // stk::mesh::field
  std::vector<PHX::DataLayout::size_type> dims;
  Teuchos::RCP<stk::mesh::MetaData> metaData;
};

// Explicit template definitions in support of the above

// Node Scalar
template <typename T>
struct NodeData_Traits<T, 1> {

  enum { size = 1 }; // One array dimension tag (Node), store type T values
  typedef stk::mesh::Field<T> field_type ;
  static field_type* createField(const std::string& name,
                                 const std::vector<PHX::DataLayout::size_type>& /* dim */,
                                 stk::mesh::MetaData* metaData)
  {
    field_type *fld = & metaData->declare_field<field_type>(stk::topology::NODE_RANK, name);
    // Multi-dim order is Fortran Ordering, so reversed here
    stk::mesh::put_field_on_mesh(*fld , metaData->universal_part(), nullptr);

    return fld; // Address is held by stk
  }

  static void saveFieldData(const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec,
                            const stk::mesh::BucketVector& all_elements,
                            field_type *fld, int offset)
  {
    Teuchos::ArrayRCP<const ST> const_overlap_node_view = getLocalData(overlap_node_vec->col(offset));

    auto indexer = createGlobalLocalIndexer(overlap_node_vec->range());
    for(auto it=all_elements.begin(); it!=all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      const stk::mesh::BulkData& bulkData = bucket.mesh();

      BucketArray<field_type> solution_array(*fld, bucket);

      const int num_nodes_in_bucket = solution_array.dimension(0);

      for (int i = 0; i < num_nodes_in_bucket; i++) {
        const GO global_id = bulkData.identifier(bucket[i]) - 1; // global node in mesh
        const LO local_id = indexer->getLocalElement(global_id);
        solution_array(i) = const_overlap_node_view[local_id];
      }
    }
  }
};

// Node Vector
template <typename T>
struct NodeData_Traits<T, 2> {

  enum { size = 2 }; // Two array dimension tags (Node, Dim), store type T values
  typedef stk::mesh::Field<T, stk::mesh::Cartesian> field_type ;
  static field_type* createField(const std::string& name,
                                 const std::vector<PHX::DataLayout::size_type>& dim,
                                 stk::mesh::MetaData* metaData)
  {
    field_type *fld = & metaData->declare_field<field_type>(stk::topology::NODE_RANK, name);
    // Multi-dim order is Fortran Ordering, so reversed here
    stk::mesh::put_field_on_mesh(*fld , metaData->universal_part(), dim[1], nullptr);

    return fld; // Address is held by stk
  }

  static void saveFieldData(const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec,
                            const stk::mesh::BucketVector& all_elements,
                            field_type *fld, int offset)
  {
    auto indexer = createGlobalLocalIndexer(overlap_node_vec->range());
    for(auto it=all_elements.begin(); it!=all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;

      BucketArray<field_type> solution_array(*fld, bucket);
      stk::mesh::BulkData const& bulkData = bucket.mesh();

      const int num_vec_components = solution_array.dimension(0);
      const int num_nodes_in_bucket = solution_array.dimension(1);

      for(int j=0; j<num_vec_components; ++j) {
        Teuchos::ArrayRCP<const ST> const_overlap_node_view = getLocalData(overlap_node_vec->col(offset + j));

        for(int i=0; i<num_nodes_in_bucket; ++i) {

          const GO global_id = bulkData.identifier(bucket[i]) - 1; // global node in mesh
          const LO local_id = indexer->getLocalElement(global_id);

          solution_array(j, i) = const_overlap_node_view[local_id];
        }
      }
    }
  }
};

// Node Tensor
template <typename T>
struct NodeData_Traits<T, 3> {

  enum { size = 3 }; // Three array dimension tags (Node, Dim, Dim), store type T values
  typedef stk::mesh::Field<T, stk::mesh::Cartesian, stk::mesh::Cartesian> field_type ;
  static field_type* createField(const std::string& name,
                                 const std::vector<PHX::DataLayout::size_type>& dim,
                                 stk::mesh::MetaData* metaData)
  {
    field_type *fld = & metaData->declare_field<field_type>(stk::topology::NODE_RANK, name);
    // Multi-dim order is Fortran Ordering, so reversed here
    stk::mesh::put_field_on_mesh(*fld , metaData->universal_part(), dim[2], dim[1], nullptr);

    return fld; // Address is held by stk
  }

  static void saveFieldData(const Teuchos::RCP<const Thyra_MultiVector>& overlap_node_vec,
                            const stk::mesh::BucketVector& all_elements,
                            field_type *fld, int offset)
  {
    auto indexer = createGlobalLocalIndexer(overlap_node_vec->range());
    for(auto it=all_elements.begin(); it!=all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      stk::mesh::BulkData const& bulkData = bucket.mesh();

      BucketArray<field_type> solution_array(*fld, bucket);

      const int num_i_components = solution_array.dimension(0);
      const int num_j_components = solution_array.dimension(1);
      const int num_nodes_in_bucket = solution_array.dimension(2);

      for(int j=0; j<num_j_components; ++j) {
        for(int k=0; k<num_i_components; ++k) {
          Teuchos::ArrayRCP<const ST> const_overlap_node_view = getLocalData(overlap_node_vec->col(offset + j*num_i_components + k));

          for(int i=0; i<num_nodes_in_bucket; ++i)  {
            const GO global_id = bulkData.identifier(bucket[i]) - 1; // global node in mesh
            const LO local_id = indexer->getLocalElement(global_id);
            solution_array(k, j, i) = const_overlap_node_view[local_id];
      }}}
    }
  }

};

} // namespace Albany

#endif // ALBANY_STK_NODE_FIELD_CONTAINER_HPP
