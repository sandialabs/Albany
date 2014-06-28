//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Types_h)
#define LCM_Topology_Types_h

// STK includes
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/fem/CreateAdjacentEntities.hpp>
#include <stk_mesh/fem/FEMMetaData.hpp>
#include <stk_mesh/fem/SkinMesh.hpp>

// Boost includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graphviz.hpp>

// Shards includes
#include <Shards_CellTopology.hpp>
#include <Shards_BasicTopologies.hpp>

// Teuchos includes
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_CommandLineProcessor.hpp>

//Intrepid includes
#include <Intrepid_MiniTensor.h>

// Albany includes
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Utils.hpp"

using stk_classic::mesh::Bucket;
using stk_classic::mesh::BulkData;
using stk_classic::mesh::Entity;
using stk_classic::mesh::EntityId;
using stk_classic::mesh::EntityKey;
using stk_classic::mesh::EntityRank;
using stk_classic::mesh::EntityVector;
using stk_classic::mesh::Field;
using stk_classic::mesh::PairIterRelation;
using stk_classic::mesh::Relation;
using stk_classic::mesh::RelationVector;

using Teuchos::RCP;

using Albany::STKDiscretization;

namespace LCM {

typedef stk_classic::mesh::RelationIdentifier EdgeId;

typedef boost::vertex_name_t VertexName;
typedef boost::edge_name_t EdgeName;
typedef boost::property<VertexName, EntityRank> VertexProperty;
typedef boost::property<EdgeName, EdgeId> EdgeProperty;
typedef boost::listS ListS;
typedef boost::vecS VectorS;
typedef boost::bidirectionalS Directed;
typedef boost::undirectedS Undirected;

typedef boost::adjacency_list<
    ListS, ListS, Directed, VertexProperty, EdgeProperty> Graph;

typedef boost::property_map<Graph, VertexName>::type VertexNamePropertyMap;
typedef boost::property_map<Graph, EdgeName>::type EdgeNamePropertyMap;

typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;
typedef boost::graph_traits<Graph>::vertex_iterator VertexIterator;
typedef boost::graph_traits<Graph>::edge_iterator EdgeIterator;
typedef boost::graph_traits<Graph>::out_edge_iterator OutEdgeIterator;
typedef boost::graph_traits<Graph>::in_edge_iterator InEdgeIterator;

typedef Albany::AbstractSTKFieldContainer::IntScalarFieldType
    IntScalarFieldType;

typedef Albany::AbstractSTKFieldContainer::VectorFieldType
    VectorFieldType;

typedef Albany::AbstractSTKFieldContainer::TensorFieldType
    TensorFieldType;

// Specific to topological manipulation
typedef std::pair<Entity*, Entity*> EntityPair;
typedef std::map<Vertex, size_t> ComponentMap;
typedef std::map<Entity*, Entity*> ElementNodeMap;

enum FractureState {CLOSED = 0, OPEN = 1};

enum VTKCellType {INVALID = 0, VERTEX = 1, LINE = 2, TRIANGLE = 5, QUAD = 9};

static EntityRank const
INVALID_RANK = stk_classic::mesh::fem::FEMMetaData::INVALID_RANK;

static EntityRank const
NODE_RANK = stk_classic::mesh::fem::FEMMetaData::NODE_RANK;

static EntityRank const
EDGE_RANK = stk_classic::mesh::fem::FEMMetaData::EDGE_RANK;

static EntityRank const
FACE_RANK = stk_classic::mesh::fem::FEMMetaData::FACE_RANK;

static EntityRank const
VOLUME_RANK = stk_classic::mesh::fem::FEMMetaData::VOLUME_RANK;

///
/// \brief Struct to store the data needed for creation or
///        deletion of an edge in the stk mesh object.
///
/// \param source entity key
/// \param target entity key
/// \param local id of the target entity with respect to the source
///
/// To operate on an stk relation between entities (e.g. deleting
/// a relation), need the source entity, target entity, and local
/// ID of the relation with respect to the source entity.
///
/// Used to create edges from the stk mesh object in a boost graph
///
struct stkEdge {
  EntityKey source;
  EntityKey target;
  EdgeId local_id;
};

///
/// Check if edges are the same
///
struct EdgeLessThan
{
  bool operator()(stkEdge const & a, stkEdge const & b) const
  {
    if (a.source < b.source) return true;
    if (a.source > b.source) return false;
    // source a and b are the same - check target
    return (a.target < b.target);
  }
};

} // namespace LCM

#endif // LCM_Topology_Types_h
