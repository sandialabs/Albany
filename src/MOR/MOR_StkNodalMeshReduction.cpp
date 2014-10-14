//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_StkNodalMeshReduction.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/BulkModification.hpp"

#include "Teuchos_Ptr.hpp"

#include <boost/iterator/indirect_iterator.hpp>

#include <algorithm>
#include <iterator>

namespace MOR {

class BulkModification {
public:
  explicit BulkModification(stk::mesh::BulkData &target) :
    target_(target)
  { target_.modification_begin(); }

  ~BulkModification() { target_.modification_end(); }

  const stk::mesh::BulkData &target() const { return target_; }
  stk::mesh::BulkData &target() { return target_; }

private:
  stk::mesh::BulkData &target_;

  BulkModification(const BulkModification &);
  BulkModification &operator=(const BulkModification &);
};

void addNodesToPart(
    const Teuchos::ArrayView<const stk::mesh::EntityId> &nodeIds,
    stk::mesh::Part &samplePart,
    stk::mesh::BulkData& bulkData)
{
  const stk::mesh::PartVector samplePartVec(1, &samplePart);
  const stk::mesh::Selector locallyOwned = stk::mesh::MetaData::get(bulkData).locally_owned_part();

  BulkModification mod(bulkData);
  typedef Teuchos::ArrayView<const stk::mesh::EntityId>::const_iterator Iter;
  for (Iter it = nodeIds.begin(), it_end = nodeIds.end(); it != it_end; ++it) {
    stk::mesh::Entity node = bulkData.get_entity(stk::topology::NODE_RANK, *it);
    if (bulkData.is_valid(node) && locallyOwned(bulkData.bucket(node))) {
      bulkData.change_entity_parts(node, samplePartVec);
    }
  }
}

class EntityDestructor : public std::iterator<std::output_iterator_tag, void, void, void, void> {
public:
  EntityDestructor() : modification_() {}
  explicit EntityDestructor(BulkModification &m) : modification_(&m) {}

  // Trivial operations (implemented as noops)
  EntityDestructor &operator++() { return *this; }
  EntityDestructor &operator++(int) { return *this; }
  EntityDestructor &operator*() { return *this; }

  EntityDestructor &operator=(stk::mesh::Entity e) {
    (void) modification_->target().destroy_entity(e); // Ignore return value, may silently fails
    return *this;
  }

private:
  BulkModification *modification_;
};

void performNodalMeshReduction(
    stk::mesh::Part &samplePart,
    stk::mesh::BulkData& bulkData)
{
  const stk::mesh::MetaData &metaData = stk::mesh::MetaData::get(bulkData);

  std::vector<stk::mesh::Entity> sampleNodes;
  stk::mesh::get_selected_entities(samplePart, bulkData.buckets(stk::topology::NODE_RANK), sampleNodes);

  const stk::mesh::Selector locallyOwned = stk::mesh::MetaData::get(bulkData).locally_owned_part();

  std::vector<stk::mesh::Entity> relatedEntities;
  typedef std::vector<stk::mesh::Entity>::const_iterator EntityIterator;
  for (EntityIterator it(sampleNodes.begin()), it_end(sampleNodes.end()); it != it_end; ++it) {
    for (stk::mesh::EntityRank r = stk::topology::NODE_RANK; r < metaData.entity_rank_count(); ++r) {
      stk::mesh::Entity const* relations = bulkData.begin(*it, r);
      const int num_rels = bulkData.num_connectivity(*it, r);
      for (int i = 0; i < num_rels; ++i) {
        stk::mesh::Entity relatedEntity = relations[i];
        if (bulkData.is_valid(relatedEntity) && locallyOwned(bulkData.bucket(relatedEntity))) {
          relatedEntities.push_back(relatedEntity);
        }
      }
    }
  }
  std::sort(relatedEntities.begin(), relatedEntities.end(), stk::mesh::EntityLess(bulkData));
  relatedEntities.erase(
      std::unique(relatedEntities.begin(), relatedEntities.end(), stk::mesh::EntityEqual()),
      relatedEntities.end());

  std::vector<stk::mesh::Entity> sampleClosure;
  stk::mesh::find_closure(bulkData, relatedEntities, sampleClosure);

  // Keep only the closure, remove the rest, by decreasing entityRanks
  {
    const stk::mesh::Selector ownedOrShared = metaData.locally_owned_part() | metaData.globally_shared_part();
    EntityIterator allKeepersEnd(sampleClosure.end());
    const EntityIterator allKeepersBegin(sampleClosure.begin());
    for (size_t candidateRankCount = metaData.entity_rank_count(); candidateRankCount > 0; --candidateRankCount) {
      const stk::mesh::EntityRank candidateRank = static_cast<stk::mesh::EntityRank>(candidateRankCount - 1);
      const EntityIterator keepersBegin = std::lower_bound(allKeepersBegin, allKeepersEnd,
                                                           stk::mesh::EntityKey(candidateRank, 0),
                                                           stk::mesh::EntityLess(bulkData));
      const EntityIterator keepersEnd = allKeepersEnd;
      std::vector<stk::mesh::Entity> candidates;
      stk::mesh::get_selected_entities(ownedOrShared, bulkData.buckets(candidateRank), candidates);
      {
        BulkModification modification(bulkData);
        std::set_difference(candidates.begin(), candidates.end(),
                            keepersBegin.base(), keepersEnd.base(),
                            EntityDestructor(modification),
                            stk::mesh::EntityLess(bulkData));
      }
      allKeepersEnd = keepersBegin;
    }
  }
}

} // end namespace MOR
