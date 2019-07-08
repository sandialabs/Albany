//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Types_h)
#define LCM_Topology_Types_h

// STK includes
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/CreateAdjacentEntities.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/Types.hpp>

// Boost includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/properties.hpp>
#ifdef __INTEL_COMPILER
#pragma warning(disable : 2196)
// On Intel compiler, disable boost warning
// /projects/albany/intel5.1/include/boost/xpressive/detail/core/adaptor.hpp(75):
// warning #2196:
//    routine is both "inline" and "noinline"
#endif
#include <boost/graph/graphviz.hpp>

// Shards includes
#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopology.hpp>

// Teuchos includes
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <MiniTensor.h>

// Albany includes
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

using EdgeId              = stk::mesh::RelationIdentifier;
using EntityVectorIndex   = stk::mesh::EntityVector::size_type;
using RelationVectorIndex = stk::mesh::RelationVector::size_type;
using Coordinates         = std::vector<minitensor::Vector<double>>;
using CoordinatesIndex    = Coordinates::size_type;
using Connectivity        = std::vector<std::vector<stk::mesh::EntityId>>;
using ConnectivityIndex   = Connectivity::size_type;

using VertexName     = boost::vertex_name_t;
using EdgeName       = boost::edge_name_t;
using VertexProperty = boost::property<VertexName, stk::mesh::EntityRank>;
using EdgeProperty   = boost::property<EdgeName, EdgeId>;
using ListS          = boost::listS;
using VectorS        = boost::vecS;
using Directed       = boost::bidirectionalS;
using Undirected     = boost::undirectedS;
using Graph =
    boost::adjacency_list<ListS, ListS, Directed, VertexProperty, EdgeProperty>;
using VertexNamePropertyMap = boost::property_map<Graph, VertexName>::type;
using EdgeNamePropertyMap   = boost::property_map<Graph, EdgeName>::type;
using Vertex                = boost::graph_traits<Graph>::vertex_descriptor;
using Edge                  = boost::graph_traits<Graph>::edge_descriptor;
using VertexIterator        = boost::graph_traits<Graph>::vertex_iterator;
using EdgeIterator          = boost::graph_traits<Graph>::edge_iterator;
using OutEdgeIterator       = boost::graph_traits<Graph>::out_edge_iterator;
using InEdgeIterator        = boost::graph_traits<Graph>::in_edge_iterator;

using IntScalarFieldType =
    Albany::AbstractSTKFieldContainer::IntScalarFieldType;
using ScalarFieldType = Albany::AbstractSTKFieldContainer::ScalarFieldType;
using VectorFieldType = Albany::AbstractSTKFieldContainer::VectorFieldType;
using TensorFieldType = Albany::AbstractSTKFieldContainer::TensorFieldType;

using EntityPair         = std::pair<stk::mesh::Entity, stk::mesh::Entity>;
using VertexComponentMap = std::map<Vertex, size_t>;
using EntityEntityMap    = std::map<stk::mesh::Entity, stk::mesh::Entity>;

enum FailureState
{
  INTACT = 0,
  FAILED = 1
};

enum BoundaryIndicator
{
  INTERIOR = 0,
  EXTERIOR = 1
};

enum VTKCellType
{
  INVALID  = 0,
  VERTEX   = 1,
  LINE     = 2,
  TRIANGLE = 5,
  QUAD     = 9
};

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
struct STKEdge
{
  stk::mesh::Entity source;
  stk::mesh::Entity target;
  EdgeId            local_id;
};

///
/// Check if edges are the same
///
struct EdgeLessThan
{
  bool
  operator()(STKEdge const& a, STKEdge const& b) const
  {
    if (a.source < b.source) return true;

    // stk::mesh::Entity does not have operator>() (!),
    // thus check for equality next.
    if (a.source == b.source) return (a.target < b.target);

    return false;
  }
};

stk::mesh::Entity const INVALID_ENTITY(
    stk::mesh::Entity::Entity_t::InvalidEntity);

// Forward declarations
class Topology;

}  // namespace LCM

#endif  // LCM_Topology_Types_h
