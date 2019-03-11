//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"

#include "Albany_BucketArray.hpp"
#include <stk_mesh/base/GetBuckets.hpp>

namespace Albany
{

template<typename BucketArrayType>
typename std::conditional<std::is_const<BucketArrayType>::value,const double&, double&>::type
access (BucketArrayType& array, const int i, const int j);

template<>
const double& access<const BucketArray<AbstractSTKFieldContainer::ScalarFieldType>> (const BucketArray<AbstractSTKFieldContainer::ScalarFieldType>& array, const int j, const int i)
{
  ALBANY_EXPECT (j==0, "Error! Attempting to access 1d array with two indices.\n");
  return array(i);
}


template<>
double& access<BucketArray<AbstractSTKFieldContainer::ScalarFieldType>> (BucketArray<AbstractSTKFieldContainer::ScalarFieldType>& array, const int j, const int i)
{
  ALBANY_EXPECT (j==0, "Error! Attempting to access 1d array with two indices.\n");
  return array(i);
}

template<>
const double& access<const BucketArray<AbstractSTKFieldContainer::VectorFieldType>> (const BucketArray<AbstractSTKFieldContainer::VectorFieldType>& array, const int j, const int i)
{
  return array(j,i);
}

template<>
double& access<BucketArray<AbstractSTKFieldContainer::VectorFieldType>> (BucketArray<AbstractSTKFieldContainer::VectorFieldType>& array, const int j, const int i)
{
  return array(j,i);
}

// Get the rank of a field
template<typename FieldType>
constexpr int getRank () {
  return std::is_same<FieldType,AbstractSTKFieldContainer::ScalarFieldType>::value ? 0 :
         (std::is_same<FieldType,AbstractSTKFieldContainer::VectorFieldType>::value ? 1 :
         (std::is_same<FieldType,AbstractSTKFieldContainer::TensorFieldType>::value ? 2 : -1));
}

// Fill the result vector
// Create a multidimensional array view of the
// solution field data for this bucket of nodes.
// The array is two dimensional ( Cartesian X NumberNodes )
// and indexed by ( 0..2 , 0..NumberNodes-1 )

template<class FieldType>
void STKFieldContainerHelper<FieldType>::
fillVector (Thyra_Vector&    field_thyra,
            const FieldType& field_stk,
            const Teuchos::RCP<const Thyra_VectorSpace>& node_vs,
            const stk::mesh::Bucket& bucket,
            const NodalDOFManager& nodalDofManager,
            const int offset)
{
  constexpr int rank = getRank<FieldType>();
  TEUCHOS_TEST_FOR_EXCEPTION(rank==0 || rank==1, std::runtime_error,
                             "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  BucketArray<FieldType> field_array(field_stk, bucket);

  const int num_nodes_in_bucket = field_array.dimension(1);

  const stk::mesh::BulkData& mesh = field_stk.get_mesh();
  auto data = getNonconstLocalData(field_thyra);
  for(int i=0; i<num_nodes_in_bucket; ++i)  {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = getLocalElement(node_vs,node_gid);

    const int num_vec_components = nodalDofManager.numComponents();
    for(int j=0; j<num_vec_components; ++j) {
      data[nodalDofManager.getLocalDOF(node_lid,offset+j)] = access(field_array,j,i);
    }
  }
}

template<class FieldType>
void STKFieldContainerHelper<FieldType>::
saveVector(const Thyra_Vector& field_thyra,
           FieldType& field_stk,
           const Teuchos::RCP<const Thyra_VectorSpace>& node_vs,
           const stk::mesh::Bucket& bucket,
           const NodalDOFManager& nodalDofManager,
           const int offset)
{
  constexpr int rank = getRank<FieldType>();
  TEUCHOS_TEST_FOR_EXCEPTION(rank==0 || rank==1, std::runtime_error,
                             "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  BucketArray<FieldType> field_array(field_stk, bucket);

  const int num_nodes_in_bucket = field_array.dimension(1);

  const stk::mesh::BulkData& mesh = field_stk.get_mesh();
  auto data = getLocalData(field_thyra);
  for(int i=0; i<num_nodes_in_bucket; ++i) {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = getLocalElement(node_vs,node_gid);

    const int num_vec_components = nodalDofManager.numComponents();
    for(int j = 0; j<num_vec_components; ++j) {
      access(field_array,j,i) = data[nodalDofManager.getLocalDOF(node_lid,offset+j)];
    }
  }
}

template<class FieldType>
void STKFieldContainerHelper<FieldType>::
copySTKField(const FieldType& source,
             FieldType& target)
{
  constexpr int rank = getRank<FieldType>();
  TEUCHOS_TEST_FOR_EXCEPTION(rank==0 || rank==1, std::runtime_error,
                             "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  const stk::mesh::BulkData&     mesh = source.get_mesh();
  const stk::mesh::BucketVector& bv   = mesh.buckets(stk::topology::NODE_RANK);

  for(stk::mesh::BucketVector::const_iterator it = bv.begin() ; it != bv.end() ; ++it) {
    const stk::mesh::Bucket& bucket = **it;

    BucketArray<FieldType> source_array(source, bucket);
    BucketArray<FieldType> target_array(target, bucket);

    const int num_source_components = source_array.dimension(0);
    const int num_target_components = target_array.dimension(0);
    const int num_nodes_in_bucket   = source_array.dimension(1);

    const int uneven_downsampling = num_source_components % num_target_components;

    TEUCHOS_TEST_FOR_EXCEPTION((uneven_downsampling) ||
                               (num_nodes_in_bucket != target_array.dimension(1)),
                               std::logic_error,
                               "Error in stk fields: specification of coordinate vector vs. solution layout is incorrect."
                               << std::endl);

    for(int i=0; i<num_nodes_in_bucket; ++i) {
      // In source, j varies over neq (num phys vectors * numDim)
      // We want target to only vary over the first numDim components
      // Not sure how to do this generally...
      for(int j=0; j<num_target_components; ++j) {
        access(target_array, j, i) = access(source_array, j, i);
      }
    }
  }
}

} // namespace Albany
