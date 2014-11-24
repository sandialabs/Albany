#ifndef percept_PerceptMyPairIterRelation_hpp
#define percept_PerceptMyPairIterRelation_hpp

#include <stk_mesh/base/BulkData.hpp>
  namespace percept {

    class MyPairIterRelation {

      unsigned m_size;
      const stk::mesh::Entity *m_entities;
      const stk::mesh::ConnectivityOrdinal *m_ordinals;

      MyPairIterRelation();
      MyPairIterRelation(const MyPairIterRelation& mp);
      MyPairIterRelation(unsigned size, const stk::mesh::Entity *entities, const stk::mesh::ConnectivityOrdinal *ordinals ) :
        m_size ( size), m_entities(entities), m_ordinals(ordinals) {}

    public:
// AGS: Had to comment this out to get stk_rebalance to compile without all of Percept
      //MyPairIterRelation(PerceptMesh& eMesh, stk::mesh::Entity entity, stk::mesh::EntityRank entity_rank) :
      //  m_size ( eMesh.get_bulk_data()->num_connectivity(entity, entity_rank)),
      //  m_entities ( eMesh.get_bulk_data()->begin(entity, entity_rank) ),
      //  m_ordinals ( eMesh.get_bulk_data()->begin_ordinals(entity, entity_rank) )
      //{}

      MyPairIterRelation(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity, stk::mesh::EntityRank entity_rank) :
        m_size ( bulk_data.num_connectivity(entity, entity_rank)),
        m_entities ( bulk_data.begin(entity, entity_rank) ),
        m_ordinals ( bulk_data.begin_ordinals(entity, entity_rank) )
      {}

      struct MyRelation {
        stk::mesh::Entity m_entity;
        stk::mesh::ConnectivityOrdinal m_ordinal;
        inline stk::mesh::Entity entity() const { return m_entity; }
        inline stk::mesh::ConnectivityOrdinal relation_ordinal() const { return m_ordinal; }
      };

      MyPairIterRelation& operator=(const MyPairIterRelation& mp) {
        m_size  = mp.m_size;
        m_entities = mp.m_entities;
        m_ordinals = mp.m_ordinals;
        return *this;
      }

      inline unsigned size() const { return m_size;}
      inline const MyRelation operator[](int i) const {
        MyRelation mr = { m_entities[i], m_ordinals[i] };
        return mr;
      }
    };
 }
#endif
