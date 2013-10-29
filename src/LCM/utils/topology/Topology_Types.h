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

using stk::mesh::Bucket;
using stk::mesh::Entity;
using stk::mesh::EntityId;
using stk::mesh::EntityKey;
using stk::mesh::EntityRank;
using stk::mesh::EntityVector;
using stk::mesh::Field;
using stk::mesh::PairIterRelation;

typedef stk::mesh::RelationIdentifier EdgeId;

// Boost includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graphviz.hpp>

typedef boost::vertex_name_t VertexName;
typedef boost::edge_name_t EdgeName;
typedef boost::property<VertexName, EntityRank> VertexProperty;
typedef boost::property<EdgeName, EdgeId> EdgeProperty;
typedef boost::listS List;
typedef boost::bidirectionalS Undirected;

typedef boost::adjacency_list<
    List, List, Undirected, VertexProperty, EdgeProperty> Graph;

typedef boost::property_map<Graph, VertexName>::type VertexNamePropertyMap;
typedef boost::property_map<Graph, EdgeName>::type EdgeNamePropertyMap;

typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;
typedef boost::graph_traits<Graph>::vertex_iterator VertexIterator;
typedef boost::graph_traits<Graph>::edge_iterator EdgeIterator;
typedef boost::graph_traits<Graph>::out_edge_iterator OutEdgeIterator;
typedef boost::graph_traits<Graph>::in_edge_iterator InEdgeIterator;

// Shards includes
#include <Shards_CellTopology.hpp>
#include <Shards_BasicTopologies.hpp>

// Teuchos includes
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_CommandLineProcessor.hpp>

using Teuchos::RCP;

// Albany includes
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Utils.hpp"

typedef Albany::AbstractSTKFieldContainer::IntScalarFieldType
    IntScalarFieldType;

using Albany::STKDiscretization;

// Specific to topological manipulation

enum FractureState {CLOSED = 0, OPEN = 1};

#endif // LCM_Topology_Types_h
