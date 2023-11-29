//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_STKUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraUtils.hpp"

#include <stk_mesh/base/FieldBase.hpp>
#include <type_traits>
#include <numeric>

namespace Albany
{

// Fill the result vector
// Create a multidimensional array view of the
// solution field data for this bucket of nodes.
// The array is two dimensional (NumberNodes, FieldDim)
// where FieldDim=1 for scalar field, and 1+ for vector fields
void STKFieldContainerHelper::
fillVector (      Thyra_Vector&    field_thyra,
            const FieldType& field_stk,
            const stk::mesh::BulkData& bulkData,
            const Teuchos::RCP<const DOFManager>& dof_mgr,
            const bool overlapped,
            const std::vector<int>& components)
{
  constexpr int rank = FieldRank<FieldType>::n;
  static_assert(rank==0 || rank==1,
                "Error! Can only handle Scalar and Vector fields for now.\n");

  using ScalarT = typename FieldScalar<FieldType>::type;
  using ViewT = Kokkos::View<const ScalarT**,Kokkos::LayoutRight,Kokkos::HostSpace>;

  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const bool restricted = dof_mgr->part_name()!=dof_mgr->elem_block_name();

  auto data = getNonconstLocalData(field_thyra);
  constexpr auto ELEM_RANK = stk::topology::ELEM_RANK;
  const auto& elems = dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int num_elems = elems.size();
  const auto indexer = dof_mgr->indexer();

#ifdef ALBANY_DEBUG
  // Safety check
  if (elems.size()>0) {
    const auto& e0 = bulkData.get_entity(ELEM_RANK,elems[0]+1);
    const auto& n0 = *bulkData.begin_nodes(e0);
    TEUCHOS_TEST_FOR_EXCEPTION (stk::mesh::field_scalars_per_entity(field_stk,n0)<components.size(),
        std::runtime_error,
        "Error! Number of components exceeds number of scalars per node of the STK field.\n"
        "  - number of components: " << components.size() << "\n"
        "  - number scalars/node : " << stk::mesh::field_scalars_per_entity(field_stk,n0) << "\n");
  }
#endif

  const auto get_offsets = [&] (const int eq) -> const std::vector<int>&
  {
    return dof_mgr->getGIDFieldOffsets(eq);
  };
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto elem_gid = elems[ielem];
    const auto e = bulkData.get_entity(ELEM_RANK,elem_gid+1);
    const int num_nodes = bulkData.num_nodes(e);
    const auto nodes = bulkData.begin_nodes(e);
    const auto& gids = dof_mgr->getElementGIDs(ielem);
    for (int i=0; i<num_nodes; ++i) {
      if (not overlapped) {
        // Check right away if this node is owned. We can pick fieldId=0 from the
        // dof manager, since we are guaranteed to own all or none of the gids on
        // each node.
        const auto owned_lid = indexer->getLocalElement(gids[get_offsets(0)[i]]);
        if (owned_lid<0) {
          continue;
        }
      }
      auto stk_data = stk::mesh::field_data(field_stk,nodes[i]);
      for (auto fid : components) {
        const auto& offsets = get_offsets(fid);
        const auto lid = elem_dof_lids(ielem,offsets[i]);
        if (!restricted ||lid>=0) {
          data[lid] = stk_data[fid];
        }
      }
    }
  }
}

// Shortcut for 'get all fields'
void STKFieldContainerHelper::
fillVector (      Thyra_Vector&    field_thyra,
            const FieldType& field_stk,
            const stk::mesh::BulkData& bulkData,
            const Teuchos::RCP<const DOFManager>& dof_mgr,
            const bool overlapped)
{
  std::vector<int> components(dof_mgr->getNumFields());
  std::iota(components.begin(),components.end(),0);
  fillVector(field_thyra,field_stk,bulkData,dof_mgr,overlapped,components);
}

void STKFieldContainerHelper::
saveVector(const Thyra_Vector& field_thyra,
                 FieldType& field_stk,
           const stk::mesh::BulkData& bulkData,
           const Teuchos::RCP<const DOFManager>& dof_mgr,
           const bool overlapped,
           const std::vector<int>& components)
{
  constexpr int rank = FieldRank<FieldType>::n;
  static_assert(rank==0 || rank==1,
                "Error! Can only handle scalar and vector fields for now.\n");

  using ScalarT = typename FieldScalar<FieldType>::type;
  using ViewT = Kokkos::View<ScalarT**,Kokkos::LayoutRight,Kokkos::HostSpace>;
  
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const bool restricted = dof_mgr->part_name()!=dof_mgr->elem_block_name();

  auto data = getLocalData(field_thyra);
  constexpr auto ELEM_RANK = stk::topology::ELEM_RANK;
  const auto& elems = dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int num_elems = elems.size();
  const auto indexer = dof_mgr->indexer();

#ifdef ALBANY_DEBUG
  // Safety check
  if (elems.size()>0) {
    const auto& e0 = bulkData.get_entity(ELEM_RANK,elems[0]+1);
    const auto& n0 = *bulkData.begin_nodes(e0);
    TEUCHOS_TEST_FOR_EXCEPTION (stk::mesh::field_scalars_per_entity(field_stk,n0)<components.size(),
        std::runtime_error,
        "Error! Number of components exceeds number of scalars per node of the STK field.\n"
        "  - number of components: " << components.size() << "\n"
        "  - number scalars/node : " << stk::mesh::field_scalars_per_entity(field_stk,n0) << "\n");
  }
#endif

  const auto get_offsets = [&] (const int eq) -> const std::vector<int>&
  {
    return dof_mgr->getGIDFieldOffsets(eq);
  };
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto elem_gid = elems[ielem];
    const auto e = bulkData.get_entity(ELEM_RANK,elem_gid+1);
    const int num_nodes = bulkData.num_nodes(e);
    const auto nodes = bulkData.begin_nodes(e);
    const auto& gids = dof_mgr->getElementGIDs(ielem);
    for (int i=0; i<num_nodes; ++i) {
      if (not overlapped) {
        // Check right away if this node is owned. We can pick fieldId=0 from the
        // dof manager, since we are guaranteed to own all or none of the gids on
        // each node.
        const auto owned_lid = indexer->getLocalElement(gids[get_offsets(0)[i]]);
        if (owned_lid<0) {
          continue;
        }
      }
      auto stk_data = stk::mesh::field_data(field_stk,nodes[i]);
      for (auto fid : components) {
        const auto& offsets = get_offsets(fid);
        const auto lid = elem_dof_lids(ielem,offsets[i]);
        if (!restricted or lid>=0) {
          stk_data[fid] = data[lid];
        }
      }
    }
  }
}

void STKFieldContainerHelper::
copySTKField(const FieldType& source,
                   FieldType& target)
{
  constexpr int rank = FieldRank<FieldType>::n;
  static_assert(rank==0 || rank==1,
                "Error! Can only handle scalar and vector fields for now.\n");

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

// Shortcut for 'save all fields'
void STKFieldContainerHelper::
saveVector (const Thyra_Vector&    field_thyra,
                  FieldType& field_stk,
            const stk::mesh::BulkData& bulkData,
            const Teuchos::RCP<const DOFManager>& dof_mgr,
            const bool overlapped)
{
  std::vector<int> components(dof_mgr->getNumFields());
  std::iota(components.begin(),components.end(),0);
  saveVector(field_thyra,field_stk,bulkData,dof_mgr,overlapped,components);
}


} // namespace Albany
