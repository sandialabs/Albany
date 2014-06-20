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
  explicit BulkModification(stk_classic::mesh::BulkData &target) :
    target_(target)
  { target_.modification_begin(); }

  ~BulkModification() { target_.modification_end(); }

  const stk_classic::mesh::BulkData &target() const { return target_; }
  stk_classic::mesh::BulkData &target() { return target_; }

private:
  stk_classic::mesh::BulkData &target_;

  BulkModification(const BulkModification &);
  BulkModification &operator=(const BulkModification &);
};

void addNodesToPart(
    const Teuchos::ArrayView<const stk_classic::mesh::EntityId> &nodeIds,
    stk_classic::mesh::Part &samplePart,
    stk_classic::mesh::BulkData& bulkData)
{
  const stk_classic::mesh::EntityRank nodeEntityRank(0);
  const stk_classic::mesh::PartVector samplePartVec(1, &samplePart);
  const stk_classic::mesh::Selector locallyOwned = stk_classic::mesh::MetaData::get(bulkData).locally_owned_part();

  BulkModification mod(bulkData);
  typedef Teuchos::ArrayView<const stk_classic::mesh::EntityId>::const_iterator Iter;
  for (Iter it = nodeIds.begin(), it_end = nodeIds.end(); it != it_end; ++it) {
    const Teuchos::Ptr<stk_classic::mesh::Entity> node(bulkData.get_entity(nodeEntityRank, *it));
    if (Teuchos::nonnull(node) && locallyOwned(*node)) {
      bulkData.change_entity_parts(*node, samplePartVec);
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

  EntityDestructor &operator=(stk_classic::mesh::Entity *&e) {
    (void) modification_->target().destroy_entity(e); // Ignore return value, may silently fails
    return *this;
  }
  EntityDestructor &operator=(stk_classic::mesh::Entity *const &e) {
    stk_classic::mesh::Entity *e_copy = e;
    return this->operator=(e_copy);
  }

private:
  BulkModification *modification_;
};

void performNodalMeshReduction(
    stk_classic::mesh::Part &samplePart,
    stk_classic::mesh::BulkData& bulkData)
{
  const stk_classic::mesh::EntityRank nodeEntityRank(0);
  const stk_classic::mesh::MetaData &metaData = stk_classic::mesh::MetaData::get(bulkData);

  std::vector<stk_classic::mesh::Entity *> sampleNodes;
  stk_classic::mesh::get_selected_entities(samplePart, bulkData.buckets(nodeEntityRank), sampleNodes);

  const stk_classic::mesh::Selector locallyOwned = stk_classic::mesh::MetaData::get(bulkData).locally_owned_part();

  std::vector<stk_classic::mesh::Entity *> relatedEntities;
  typedef boost::indirect_iterator<std::vector<stk_classic::mesh::Entity *>::const_iterator> EntityIterator;
  for (EntityIterator it(sampleNodes.begin()), it_end(sampleNodes.end()); it != it_end; ++it) {
    const stk_classic::mesh::PairIterRelation relations = it->relations();
    typedef stk_classic::mesh::PairIterRelation::first_type RelationIterator;
    for (RelationIterator rel_it = relations.first, rel_it_end = relations.second; rel_it != rel_it_end; ++rel_it) {
      const Teuchos::Ptr<stk_classic::mesh::Entity> relatedEntity(rel_it->entity());
      if (Teuchos::nonnull(relatedEntity) && locallyOwned(*relatedEntity)) {
        relatedEntities.push_back(relatedEntity.get());
      }
    }
  }
  std::sort(relatedEntities.begin(), relatedEntities.end(), stk_classic::mesh::EntityLess());
  relatedEntities.erase(
      std::unique(relatedEntities.begin(), relatedEntities.end(), stk_classic::mesh::EntityEqual()),
      relatedEntities.end());

  std::vector<stk_classic::mesh::Entity *> sampleClosure;
  stk_classic::mesh::find_closure(bulkData, relatedEntities, sampleClosure);

  // Keep only the closure, remove the rest, by decreasing entityRanks
  {
    const stk_classic::mesh::Selector ownedOrShared = metaData.locally_owned_part() | metaData.globally_shared_part();
    typedef boost::indirect_iterator<std::vector<stk_classic::mesh::Entity *>::const_iterator> EntityIterator;
    EntityIterator allKeepersEnd(sampleClosure.end());
    const EntityIterator allKeepersBegin(sampleClosure.begin());
    for (stk_classic::mesh::EntityRank candidateRankCount = metaData.entity_rank_count(); candidateRankCount > 0; --candidateRankCount) {
      const stk_classic::mesh::EntityRank candidateRank = candidateRankCount - 1;
      const EntityIterator keepersBegin = std::lower_bound(allKeepersBegin, allKeepersEnd,
                                                           stk_classic::mesh::EntityKey(candidateRank, 0),
                                                           stk_classic::mesh::EntityLess());
      const EntityIterator keepersEnd = allKeepersEnd;
      std::vector<stk_classic::mesh::Entity *> candidates;
      stk_classic::mesh::get_selected_entities(ownedOrShared, bulkData.buckets(candidateRank), candidates);
      {
        BulkModification modification(bulkData);
        std::set_difference(candidates.begin(), candidates.end(),
                            keepersBegin.base(), keepersEnd.base(),
                            EntityDestructor(modification),
                            stk_classic::mesh::EntityLess());
      }
      allKeepersEnd = keepersBegin;
    }
  }
}

} // end namespace MOR
