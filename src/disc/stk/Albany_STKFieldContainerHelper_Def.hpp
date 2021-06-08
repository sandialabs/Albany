//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_STKUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraUtils.hpp"

#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <type_traits>

namespace Albany
{

// Fill the result vector
// Create a multidimensional array view of the
// solution field data for this bucket of nodes.
// The array is two dimensional (NumberNodes, FieldDim)
// where FieldDim=1 for scalar field, and 1+ for vector fields

template<class FieldType>
void STKFieldContainerHelper<FieldType>::
fillVector (      Thyra_Vector&    field_thyra,
            const FieldType& field_stk,
            const Teuchos::RCP<const GlobalLocalIndexer>& indexer,
            const stk::mesh::Bucket& bucket,
            const NodalDOFManager& nodalDofManager,
            const int offset)
{
  constexpr int rank = FieldRank<FieldType>::n;
  static_assert(rank==0 || rank==1,
                "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  using ScalarT = typename FieldScalar<FieldType>::type;
  using ViewT = Kokkos::View<const ScalarT**,Kokkos::LayoutRight,Kokkos::HostSpace>;

  auto stk_data = stk::mesh::field_data(field_stk,bucket);
  const int num_nodes_in_bucket = bucket.size();
  const int num_vec_components = nodalDofManager.numComponents();

  ViewT field_view (stk_data,num_nodes_in_bucket,num_vec_components);

  const stk::mesh::BulkData& mesh = field_stk.get_mesh();
  auto data = getNonconstLocalData(field_thyra);

  for(int i=0; i<num_nodes_in_bucket; ++i)  {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = indexer->getLocalElement(node_gid);

    for(int j=0; j<num_vec_components; ++j) {
      data[nodalDofManager.getLocalDOF(node_lid,offset+j)] = field_view(i,j);
    }
  }
}

template<class FieldType>
void STKFieldContainerHelper<FieldType>::
saveVector(const Thyra_Vector& field_thyra,
           FieldType& field_stk,
           const Teuchos::RCP<const GlobalLocalIndexer>& indexer,
           const stk::mesh::Bucket& bucket,
           const NodalDOFManager& nodalDofManager,
           const int offset)
{
  constexpr int rank = FieldRank<FieldType>::n;
  static_assert(rank==0 || rank==1,
                "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  using ScalarT = typename FieldScalar<FieldType>::type;
  using ViewT = Kokkos::View<ScalarT**,Kokkos::LayoutRight,Kokkos::HostSpace>;
  
  auto stk_data = stk::mesh::field_data(field_stk,bucket);
  const int num_nodes_in_bucket = bucket.size();
  const int num_vec_components = nodalDofManager.numComponents();

  ViewT field_view (stk_data,num_nodes_in_bucket,num_vec_components);

  const auto& mesh = field_stk.get_mesh();
  auto data = getLocalData(field_thyra);

  for(int i=0; i<num_nodes_in_bucket; ++i) {
    const GO node_gid = mesh.identifier(bucket[i]) - 1;
    const LO node_lid = indexer->getLocalElement(node_gid);

    for(int j = 0; j<num_vec_components; ++j) {
      field_view(i,j) = data[nodalDofManager.getLocalDOF(node_lid,offset+j)];
    }
  }
}

template<class FieldType>
void STKFieldContainerHelper<FieldType>::
copySTKField(const FieldType& source,
                   FieldType& target)
{
  constexpr int rank = FieldRank<FieldType>::n;
  static_assert(rank==0 || rank==1,
                "Error! Can only handle ScalarFieldType and VectorFieldType for now.\n");

  using ScalarT = typename FieldScalar<FieldType>::type;
  using SrcViewT = Kokkos::View<const ScalarT**,Kokkos::LayoutRight,Kokkos::HostSpace>;
  using TgtViewT = Kokkos::View<      ScalarT**,Kokkos::LayoutRight,Kokkos::HostSpace>;

  const stk::mesh::BulkData&     mesh = source.get_mesh();
  const stk::mesh::BucketVector& bv   = mesh.buckets(stk::topology::NODE_RANK);

  for (auto it : bv) {
    const stk::mesh::Bucket& bucket = *it;
    const int num_nodes_in_bucket = bucket.size();
    const int num_source_components = stk::mesh::field_scalars_per_entity(target, bucket);
    const int num_target_components = stk::mesh::field_scalars_per_entity(target, bucket);

    const int uneven_downsampling = num_source_components % num_target_components;
    TEUCHOS_TEST_FOR_EXCEPTION(uneven_downsampling, std::logic_error,
        "Error in stk fields: specification of coordinate vector vs. solution layout is incorrect.\n");

    auto src_stk_data = stk::mesh::field_data(source,bucket);
    auto tgt_stk_data = stk::mesh::field_data(target,bucket);

    SrcViewT src_view (src_stk_data,num_nodes_in_bucket,num_source_components);
    TgtViewT tgt_view (tgt_stk_data,num_nodes_in_bucket,num_target_components);

    for(int i=0; i<num_nodes_in_bucket; ++i) {
      // In source, j varies over neq (num phys vectors * numDim)
      // We want target to only vary over the first numDim components
      // Not sure how to do this generally...
      for(int j=0; j<num_target_components; ++j) {
        tgt_view(i,j) = src_view(i,j);
      }
    }
  }
}

} // namespace Albany
