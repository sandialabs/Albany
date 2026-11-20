//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MockMPAS.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace mock_mpas {

PartitionType partitionTypeFromString (const std::string& name)
{
  if (name=="Strips")    return PartitionType::Strips;
  if (name=="Blocks")    return PartitionType::Blocks;
  if (name=="Scattered") return PartitionType::Scattered;
  throw std::invalid_argument("Error! Unknown partition type '" + name + "'. Valid types: Strips, Blocks, Scattered.\n");
}

void setPhysicalParameters (const PhysicalParameters& p)
{
  velocity_solver_set_physical_parameters__(p.gravity, p.iceDensity, p.oceanDensity, p.seaLevel, p.flowParamA,
      p.flowLawExponent, p.dynamicThickness, p.useGLP, p.clausiusClapeyronCoeff, p.meltingTemperature,
      p.secondsPerYear, p.gasConstant);
}

void setPhysicalParametersDeprecated (const PhysicalParameters& p)
{
  velocity_solver_set_physical_parameters__(p.gravity, p.iceDensity, p.oceanDensity, p.seaLevel, p.flowParamA,
      p.flowLawExponent, p.dynamicThickness, p.useGLP, p.clausiusClapeyronCoeff);
}

// ================================================================================================ //

MockMPAS::MockMPAS (MPI_Comm worldComm, const MeshOptions& options)
 : m_worldComm (worldComm)
 , m_numLayers (options.numLayers)
 , m_ordering (options.ordering)
{
  int worldSize;
  MPI_Comm_rank(m_worldComm, &m_worldRank);
  MPI_Comm_size(m_worldComm, &worldSize);

  if (m_numLayers<1) {
    throw std::invalid_argument("Error! MockMPAS needs at least one layer.\n");
  }
  if (m_ordering!=0 && m_ordering!=1) {
    throw std::invalid_argument("Error! MockMPAS ordering must be 0 (layer-wise) or 1 (column-wise).\n");
  }

  // Vertical levels
  if (options.levels.empty()) {
    m_levels.resize(m_numLayers+1);
    for (int il=0; il<=m_numLayers; ++il) {
      m_levels[il] = static_cast<double>(il)/m_numLayers;
    }
  } else {
    m_levels = options.levels;
    bool valid = static_cast<int>(m_levels.size())==m_numLayers+1 && m_levels.front()==0.0 && m_levels.back()==1.0;
    for (int il=0; valid && il<m_numLayers; ++il) {
      valid = m_levels[il]<m_levels[il+1];
    }
    if (!valid) {
      throw std::invalid_argument("Error! MockMPAS levels must be numLayers+1 increasing values, from 0 to 1.\n");
    }
  }

  // Idle ranks are the first ones, so that reduced ranks differ from world ranks.
  m_numIdleRanks = (worldSize>1) ? std::max(0,std::min(options.numIdleRanks, worldSize-1)) : 0;
  const int numActiveRanks = worldSize - m_numIdleRanks;
  m_active = m_worldRank>=m_numIdleRanks;
  m_reducedRank = m_active ? m_worldRank-m_numIdleRanks : -1;
  MPI_Comm_split(m_worldComm, m_active ? 0 : MPI_UNDEFINED, m_worldRank, &m_reducedComm);

  buildGlobalMesh(options);
  partition(options, numActiveRanks);
  buildLocalMesh(options);
}

MockMPAS::~MockMPAS ()
{
  int finalized;
  MPI_Finalized(&finalized);
  if (m_reducedComm!=MPI_COMM_NULL && !finalized) {
    MPI_Comm_free(&m_reducedComm);
  }
}

void MockMPAS::buildGlobalMesh (const MeshOptions& options)
{
  m_nx = options.nx;
  m_ny = options.ny;
  if (m_nx<2 || m_ny<2) {
    throw std::invalid_argument("Error! MockMPAS grid needs at least 2x2 vertices.\n");
  }
  auto gridIndex = [&](int i, int j) { return i + m_nx*j; };

  m_gridPoints.resize(m_nx*m_ny);
  for (int j=0; j<m_ny; ++j) {
    for (int i=0; i<m_nx; ++i) {
      m_gridPoints[gridIndex(i,j)] = {options.x0 + i*options.dx, options.y0 + j*options.dx};
    }
  }

  // Triangles: each quad is split along alternating diagonals. Vertices are listed counterclockwise.
  m_gTriangleID.clear();
  m_gTriangleVerts.clear();
  for (int j=0; j<m_ny-1; ++j) {
    for (int i=0; i<m_nx-1; ++i) {
      const int a = gridIndex(i,j), b = gridIndex(i+1,j), c = gridIndex(i+1,j+1), d = gridIndex(i,j+1);
      std::array<std::array<int,3>,2> tris;
      if ((i+j)%2==0) {
        tris = {{{a,b,c},{a,c,d}}};
      } else {
        tris = {{{a,b,d},{b,c,d}}};
      }
      for (int t=0; t<2; ++t) {
        const std::array<Point,3> pts = {m_gridPoints[tris[t][0]], m_gridPoints[tris[t][1]], m_gridPoints[tris[t][2]]};
        if (!options.isActiveTriangle || options.isActiveTriangle(pts)) {
          m_gTriangleID.push_back(1 + 2*gridIndex(i,j) + t);
          m_gTriangleVerts.push_back(tris[t]);
        }
      }
    }
  }
  m_numGlobalTriangles = m_gTriangleID.size();
  if (m_numGlobalTriangles==0) {
    throw std::invalid_argument("Error! MockMPAS FE mesh has no triangles.\n");
  }

  m_gVertexTriangles.assign(m_gridPoints.size(), {});
  for (int k=0; k<m_numGlobalTriangles; ++k) {
    for (int v : m_gTriangleVerts[k]) {
      m_gVertexTriangles[v].push_back(k);
    }
  }

  // Edges, with ids of the structured grid: 3 edges (horizontal, vertical, diagonal) per grid vertex
  std::map<std::pair<int,int>,GlobalEdge> edges;
  for (int k=0; k<m_numGlobalTriangles; ++k) {
    const auto& tv = m_gTriangleVerts[k];
    for (int e=0; e<3; ++e) {
      const int v0 = tv[e], v1 = tv[(e+1)%3];
      const auto key = std::make_pair(std::min(v0,v1),std::max(v0,v1));
      auto it = edges.find(key);
      if (it==edges.end()) {
        const int i0 = v0%m_nx, j0 = v0/m_nx, i1 = v1%m_nx, j1 = v1/m_nx;
        int id;
        if (j0==j1) {
          id = 1 + 3*gridIndex(std::min(i0,i1),j0) + 0;
        } else if (i0==i1) {
          id = 1 + 3*gridIndex(i0,std::min(j0,j1)) + 1;
        } else {
          id = 1 + 3*gridIndex(std::min(i0,i1),std::min(j0,j1)) + 2;
        }
        it = edges.emplace(key, GlobalEdge{id, {v0,v1}, {}}).first;
      }
      it->second.triangles.push_back(k);
    }
  }
  m_gEdges.clear();
  for (const auto& e : edges) {
    m_gEdges.push_back(e.second);
  }
  std::sort(m_gEdges.begin(), m_gEdges.end(), [](const GlobalEdge& a, const GlobalEdge& b) { return a.id<b.id; });

  m_gIsBoundaryVertex.assign(m_gridPoints.size(), false);
  for (const auto& e : m_gEdges) {
    if (e.triangles.size()==1) {
      m_gIsBoundaryVertex[e.verts[0]] = m_gIsBoundaryVertex[e.verts[1]] = true;
    }
  }

  // Global strides, computed as MALI does: the largest global id in the FE mesh
  m_numGlobalVertices = 0;
  m_globalVerticesStride = 0;
  for (int v=0; v<static_cast<int>(m_gridPoints.size()); ++v) {
    if (!m_gVertexTriangles[v].empty()) {
      ++m_numGlobalVertices;
      m_globalVerticesStride = std::max(m_globalVerticesStride, v+1);
    }
  }
  m_globalTrianglesStride = *std::max_element(m_gTriangleID.begin(), m_gTriangleID.end());
  m_globalEdgesStride = 0;
  for (const auto& e : m_gEdges) {
    m_globalEdgesStride = std::max(m_globalEdgesStride, e.id);
  }
}

void MockMPAS::partition (const MeshOptions& options, int numActiveRanks)
{
  const int nT = m_numGlobalTriangles;
  if (nT<numActiveRanks) {
    throw std::invalid_argument("Error! MockMPAS FE mesh has fewer triangles than ranks.\n");
  }

  std::vector<Point> centroid(nT);
  for (int k=0; k<nT; ++k) {
    double x=0, y=0;
    for (int v : m_gTriangleVerts[k]) {
      x += m_gridPoints[v].x/3;
      y += m_gridPoints[v].y/3;
    }
    centroid[k] = {x,y};
  }
  auto byX = [&](int a, int b) {
    return std::make_tuple(centroid[a].x,centroid[a].y,m_gTriangleID[a]) < std::make_tuple(centroid[b].x,centroid[b].y,m_gTriangleID[b]);
  };
  auto byY = [&](int a, int b) {
    return std::make_tuple(centroid[a].y,centroid[a].x,m_gTriangleID[a]) < std::make_tuple(centroid[b].y,centroid[b].x,m_gTriangleID[b]);
  };

  std::vector<int> order(nT);
  std::iota(order.begin(), order.end(), 0);
  m_gTriangleOwner.assign(nT, -1);

  switch (options.partition) {
    case PartitionType::Strips:
    {
      std::sort(order.begin(), order.end(), byX);
      for (int k=0; k<nT; ++k) {
        m_gTriangleOwner[order[k]] = static_cast<long>(k)*numActiveRanks/nT;
      }
      break;
    }
    case PartitionType::Blocks:
    {
      int px = 1;
      for (int p=1; p*p<=numActiveRanks; ++p) {
        if (numActiveRanks%p==0) px = p;
      }
      const int py = numActiveRanks/px;
      std::sort(order.begin(), order.end(), byX);
      for (int gx=0; gx<px; ++gx) {
        const int begin = static_cast<long>(gx)*nT/px;
        const int end   = static_cast<long>(gx+1)*nT/px;
        std::sort(order.begin()+begin, order.begin()+end, byY);
        for (int k=begin; k<end; ++k) {
          m_gTriangleOwner[order[k]] = gx*py + static_cast<long>(k-begin)*py/(end-begin);
        }
      }
      break;
    }
    case PartitionType::Scattered:
    {
      std::mt19937 gen(options.seed);
      std::shuffle(order.begin(), order.end(), gen);
      for (int k=0; k<nT; ++k) {
        m_gTriangleOwner[order[k]] = k%numActiveRanks;
      }
      break;
    }
  }

  std::vector<int> count(numActiveRanks,0);
  for (int k=0; k<nT; ++k) {
    ++count[m_gTriangleOwner[k]];
  }
  if (std::find(count.begin(), count.end(), 0)!=count.end()) {
    throw std::invalid_argument("Error! MockMPAS partition left a rank without triangles.\n");
  }

  // A vertex is owned by the owner of the triangle with the smallest id containing it. This often differs
  // from the lowest rank sharing the vertex (the default owner in STK), so Albany has to change the
  // ownership of the nodes to match MPAS's.
  m_gVertexOwner.assign(m_gridPoints.size(), -1);
  for (int v=0; v<static_cast<int>(m_gridPoints.size()); ++v) {
    int minId = std::numeric_limits<int>::max();
    for (int k : m_gVertexTriangles[v]) {
      if (m_gTriangleID[k]<minId) {
        minId = m_gTriangleID[k];
        m_gVertexOwner[v] = m_gTriangleOwner[k];
      }
    }
  }
}

void MockMPAS::buildLocalMesh (const MeshOptions& options)
{
  if (!m_active) return;

  std::mt19937 gen(options.seed + 7919u*(m_reducedRank+1));

  // Local triangles: the owned ones
  m_localTriangleToGlobal.clear();
  for (int k=0; k<m_numGlobalTriangles; ++k) {
    if (m_gTriangleOwner[k]==m_reducedRank) m_localTriangleToGlobal.push_back(k);
  }
  if (options.shuffleLocalOrdering) {
    std::shuffle(m_localTriangleToGlobal.begin(), m_localTriangleToGlobal.end(), gen);
  }
  const int nT = m_localTriangleToGlobal.size();

  // Local vertices: all the vertices of the local triangles
  std::set<int> localVerts;
  for (int k : m_localTriangleToGlobal) {
    localVerts.insert(m_gTriangleVerts[k].begin(), m_gTriangleVerts[k].end());
  }
  m_localVertexToGrid.assign(localVerts.begin(), localVerts.end());
  if (options.shuffleLocalOrdering) {
    std::shuffle(m_localVertexToGrid.begin(), m_localVertexToGrid.end(), gen);
  }
  const int nV = m_localVertexToGrid.size();
  std::vector<int> gridToLocal(m_gridPoints.size(),-1);
  for (int ib=0; ib<nV; ++ib) {
    gridToLocal[m_localVertexToGrid[ib]] = ib;
  }
  std::vector<int> globalToLocalTriangle(m_numGlobalTriangles,-1);
  for (int it=0; it<nT; ++it) {
    globalToLocalTriangle[m_localTriangleToGlobal[it]] = it;
  }

  // Vertices
  m_indexToVertexID.resize(nV);
  m_verticesCoords.resize(3*nV);
  m_vertexProcIDs.resize(nV);
  m_procsSharingVertices.assign(nV, {});
  m_isBoundaryVertex.resize(nV);
  for (int ib=0; ib<nV; ++ib) {
    const int v = m_localVertexToGrid[ib];
    m_indexToVertexID[ib] = v+1;
    m_verticesCoords[3*ib]   = m_gridPoints[v].x;
    m_verticesCoords[3*ib+1] = m_gridPoints[v].y;
    m_verticesCoords[3*ib+2] = 0.0;
    m_vertexProcIDs[ib] = m_gVertexOwner[v];
    m_isBoundaryVertex[ib] = m_gIsBoundaryVertex[v];
    std::set<int> procs;
    for (int k : m_gVertexTriangles[v]) {
      if (m_gTriangleOwner[k]!=m_reducedRank) procs.insert(m_gTriangleOwner[k]);
    }
    m_procsSharingVertices[ib].assign(procs.begin(), procs.end());
  }

  // Triangles
  m_indexToTriangleID.resize(nT);
  m_verticesOnTria.resize(3*nT);
  for (int it=0; it<nT; ++it) {
    const int k = m_localTriangleToGlobal[it];
    m_indexToTriangleID[it] = m_gTriangleID[k];
    for (int j=0; j<3; ++j) {
      m_verticesOnTria[3*it+j] = gridToLocal[m_gTriangleVerts[k][j]];
    }
  }

  // Edges of the local triangles, with the boundary edges first (as in MALI). For each edge, the first
  // entry of trianglesOnEdge is a local triangle; the second is the other triangle if it is local.
  std::vector<int> boundary, interior;
  for (int e=0; e<static_cast<int>(m_gEdges.size()); ++e) {
    bool isLocal = false;
    for (int k : m_gEdges[e].triangles) {
      isLocal = isLocal || globalToLocalTriangle[k]>=0;
    }
    if (!isLocal) continue;
    (m_gEdges[e].triangles.size()==1 ? boundary : interior).push_back(e);
  }
  if (options.shuffleLocalOrdering) {
    std::shuffle(boundary.begin(), boundary.end(), gen);
    std::shuffle(interior.begin(), interior.end(), gen);
  }
  std::vector<int> localEdges(boundary);
  localEdges.insert(localEdges.end(), interior.begin(), interior.end());

  const int nE = localEdges.size();
  m_indexToEdgeID.resize(nE);
  m_isBoundaryEdge.resize(nE);
  m_trianglesOnEdge.resize(2*nE);
  m_verticesOnEdge.resize(2*nE);
  m_iceMarginEdgesIds.clear();
  for (int ie=0; ie<nE; ++ie) {
    const auto& edge = m_gEdges[localEdges[ie]];
    m_indexToEdgeID[ie] = edge.id;
    m_isBoundaryEdge[ie] = edge.triangles.size()==1;
    int first = NotAnId, second = NotAnId;
    for (int k : edge.triangles) {
      const int it = globalToLocalTriangle[k];
      if (it<0) continue;
      (first==NotAnId ? first : second) = it;
    }
    m_trianglesOnEdge[2*ie]   = first;
    m_trianglesOnEdge[2*ie+1] = second;
    m_verticesOnEdge[2*ie]   = gridToLocal[edge.verts[0]];
    m_verticesOnEdge[2*ie+1] = gridToLocal[edge.verts[1]];
    if (m_isBoundaryEdge[ie]) {
      const auto& p0 = m_gridPoints[edge.verts[0]];
      const auto& p1 = m_gridPoints[edge.verts[1]];
      if (!options.isIceMarginEdge || options.isIceMarginEdge(p0,p1)) {
        m_iceMarginEdgesIds.push_back(edge.id);
      }
    }
  }

  // Dirichlet nodes, with the ids of the 3D nodes of the Albany mesh
  const int vertexColumnShift = (m_ordering==1) ? 1 : m_globalVerticesStride;
  const int vertexLayerShift  = (m_ordering==0) ? 1 : m_numLayers+1;
  m_dirichletNodesIds.clear();
  m_isDirichlet.assign((m_numLayers+1)*nV, false);
  if (options.isDirichletNode) {
    for (int ib=0; ib<nV; ++ib) {
      for (int il=0; il<=m_numLayers; ++il) {
        if (options.isDirichletNode(vertex(ib), il, m_isBoundaryVertex[ib])) {
          m_dirichletNodesIds.push_back(il*vertexColumnShift + vertexLayerShift*(m_indexToVertexID[ib]-1) + 1);
          m_isDirichlet[vertex3D(il,ib)] = true;
        }
      }
    }
  }
}

Point MockMPAS::triangleCentroid (int it) const
{
  Point c = {0,0};
  for (int j=0; j<3; ++j) {
    const auto p = vertex(m_verticesOnTria[3*it+j]);
    c.x += p.x/3;
    c.y += p.y/3;
  }
  return c;
}

void MockMPAS::allocate (FieldData& f) const
{
  const int nV = numVertices();
  const int nT = numTriangles();
  f.regulThickness.assign(nV, 0.0);
  f.elevation.assign(nV, 0.0);
  f.thickness.assign(nV, 0.0);
  f.beta.assign(nV, 0.0);
  f.bedTopography.assign(nV, 0.0);
  f.smb.assign(nV, 0.0);
  f.stiffeningFactor.assign(nV, 1.0);
  f.effectivePressure.assign(nV, 0.0);
  f.mu.assign(nV, 0.0);
  f.bedRoughness.clear();
  f.bulkFriction.clear();
  f.basalDebris.clear();
  f.temperatureOnPrisms.assign(m_numLayers*nT, 0.0);
  f.bodyForceOnBasalCell.assign(nT, 0.0);
  f.dissipationHeatOnPrisms.assign(m_numLayers*nT, 0.0);
  f.velocityOnVertices.assign(2*numVertices3D(), 0.0);
}

void MockMPAS::setVertexField (std::vector<double>& field, const std::function<double(const Point&)>& f) const
{
  field.resize(numVertices());
  for (int ib=0; ib<numVertices(); ++ib) {
    field[ib] = f(vertex(ib));
  }
}

void MockMPAS::setPrismField (std::vector<double>& field, const std::function<double(const Point&, double, int)>& f) const
{
  field.resize(numPrisms());
  for (int it=0; it<numTriangles(); ++it) {
    const auto c = triangleCentroid(it);
    for (int il=0; il<m_numLayers; ++il) {
      field[prism(il,it)] = f(c, 0.5*(m_levels[il]+m_levels[il+1]), il);
    }
  }
}

void MockMPAS::setVelocity (std::vector<double>& velocity,
                            const std::function<std::array<double,2>(const Point&, double, int)>& f) const
{
  const int nV3D = numVertices3D();
  velocity.resize(2*nV3D);
  for (int ib=0; ib<numVertices(); ++ib) {
    const auto p = vertex(ib);
    for (int il=0; il<=m_numLayers; ++il) {
      const auto v = f(p, m_levels[il], il);
      velocity[vertex3D(il,ib)] = v[0];
      velocity[vertex3D(il,ib)+nV3D] = v[1];
    }
  }
}

// ---------------------------- Calls to the interface ---------------------------- //

namespace {
void checkActive (bool active, const std::string& name) {
  if (!active) {
    throw std::logic_error("Error! MockMPAS::" + name + " called on a rank without triangles.\n");
  }
}
} // anonymous namespace

void MockMPAS::computeGrid ()
{
  checkActive(m_active, "computeGrid");
  velocity_solver_compute_2d_grid__(m_reducedComm);
}

void MockMPAS::extrudeGrid ()
{
  checkActive(m_active, "extrudeGrid");
  velocity_solver_extrude_3d_grid__(m_numLayers, m_globalTrianglesStride, m_globalVerticesStride,
      m_globalEdgesStride, m_ordering, m_reducedComm, m_indexToVertexID, m_vertexProcIDs, m_verticesCoords,
      m_verticesOnTria, m_procsSharingVertices, m_isBoundaryEdge, m_trianglesOnEdge, m_verticesOnEdge,
      m_indexToEdgeID, m_indexToTriangleID, m_dirichletNodesIds, m_iceMarginEdgesIds);
}

int MockMPAS::solve (FieldData& f, bool firstTimeStep, double deltat)
{
  checkActive(m_active, "solve");
  // MALI passes max(thickness, 1e-4) as regularized thickness (not used by Albany)
  f.regulThickness.resize(f.thickness.size());
  for (std::size_t i=0; i<f.thickness.size(); ++i) {
    f.regulThickness[i] = std::max(1e-4, f.thickness[i]);
  }
  const double minBeta = 1e-5;
  int error = 0;
  velocity_solver_solve_fo__(m_numLayers, m_globalVerticesStride, m_globalTrianglesStride, m_ordering!=0,
      firstTimeStep, m_indexToVertexID, m_indexToTriangleID, minBeta, f.regulThickness, m_levels,
      f.elevation, f.thickness, f.beta, f.bedTopography, f.smb, f.stiffeningFactor, f.effectivePressure,
      f.mu, f.bedRoughness, f.bulkFriction, f.basalDebris, f.temperatureOnPrisms, f.bodyForceOnBasalCell,
      f.dissipationHeatOnPrisms, f.velocityOnVertices, error, deltat);
  return error;
}

int MockMPAS::solveDeprecated (FieldData& f, bool firstTimeStep, double deltat)
{
  checkActive(m_active, "solveDeprecated");
  f.regulThickness.resize(f.thickness.size());
  for (std::size_t i=0; i<f.thickness.size(); ++i) {
    f.regulThickness[i] = std::max(1e-4, f.thickness[i]);
  }
  const double minBeta = 1e-5;
  const std::vector<double>& effectivePressure = f.effectivePressure;
  int error = 0;
  velocity_solver_solve_fo__(m_numLayers, m_globalVerticesStride, m_globalTrianglesStride, m_ordering!=0,
      firstTimeStep, m_indexToVertexID, m_indexToTriangleID, minBeta, f.regulThickness, m_levels,
      f.elevation, f.thickness, f.beta, f.bedTopography, f.smb, f.stiffeningFactor, effectivePressure,
      f.mu, f.temperatureOnPrisms, f.bodyForceOnBasalCell, f.dissipationHeatOnPrisms, f.velocityOnVertices,
      error, deltat);
  return error;
}

void MockMPAS::exportVelocity ()
{
  checkActive(m_active, "exportVelocity");
  velocity_solver_export_fo_velocity__(m_reducedComm);
}

// ---------------------------------- Checks ---------------------------------- //

std::string MockMPAS::checkConsistency () const
{
  std::ostringstream err;
  int worldSize;
  MPI_Comm_size(m_worldComm, &worldSize);
  const int numActiveRanks = worldSize - m_numIdleRanks;

  if (m_active) {
    const int nV = numVertices(), nT = numTriangles(), nE = numEdges();
    auto checkUnique = [&](std::vector<int> ids, const std::string& name) {
      std::sort(ids.begin(), ids.end());
      if (std::adjacent_find(ids.begin(), ids.end())!=ids.end()) err << "  duplicated " << name << " ids\n";
    };
    checkUnique(m_indexToVertexID, "vertex");
    checkUnique(m_indexToTriangleID, "triangle");
    checkUnique(m_indexToEdgeID, "edge");

    if (nT==0) err << "  active rank without triangles\n";
    for (int id : m_indexToVertexID)   if (id<1 || id>m_globalVerticesStride)  err << "  vertex id " << id << " out of [1,stride]\n";
    for (int id : m_indexToTriangleID) if (id<1 || id>m_globalTrianglesStride) err << "  triangle id " << id << " out of [1,stride]\n";
    for (int id : m_indexToEdgeID)     if (id<1 || id>m_globalEdgesStride)     err << "  edge id " << id << " out of [1,stride]\n";

    std::vector<int> numTriaOfVertex(nV,0);
    for (int it=0; it<nT; ++it) {
      const auto p0 = vertex(m_verticesOnTria[3*it]);
      const auto p1 = vertex(m_verticesOnTria[3*it+1]);
      const auto p2 = vertex(m_verticesOnTria[3*it+2]);
      const double area = 0.5*((p1.x-p0.x)*(p2.y-p0.y) - (p2.x-p0.x)*(p1.y-p0.y));
      if (area<=0) err << "  triangle " << m_indexToTriangleID[it] << " is not counterclockwise\n";
      for (int j=0; j<3; ++j) ++numTriaOfVertex[m_verticesOnTria[3*it+j]];
    }
    for (int ib=0; ib<nV; ++ib) {
      if (numTriaOfVertex[ib]==0) err << "  vertex " << m_indexToVertexID[ib] << " is in no local triangle\n";
      const auto& procs = m_procsSharingVertices[ib];
      for (int p : procs) {
        if (p<0 || p>=numActiveRanks || p==m_reducedRank) err << "  invalid rank " << p << " sharing vertex " << m_indexToVertexID[ib] << "\n";
      }
      const int owner = m_vertexProcIDs[ib];
      if (owner!=m_reducedRank && std::find(procs.begin(), procs.end(), owner)==procs.end()) {
        err << "  owner of vertex " << m_indexToVertexID[ib] << " is not a rank sharing it\n";
      }
    }

    // Boundary edges: first local triangle contains the edge
    for (int ie=0; ie<nE; ++ie) {
      const int it = m_trianglesOnEdge[2*ie];
      if (it<0 || it>=nT) { err << "  edge " << m_indexToEdgeID[ie] << " has no local triangle\n"; continue; }
      const int a = m_verticesOnEdge[2*ie], b = m_verticesOnEdge[2*ie+1];
      int found = 0;
      for (int j=0; j<3; ++j) found += (m_verticesOnTria[3*it+j]==a) + (m_verticesOnTria[3*it+j]==b);
      if (found!=2) err << "  edge " << m_indexToEdgeID[ie] << " is not an edge of its first triangle\n";
    }
    for (int id : m_iceMarginEdgesIds) {
      const auto pos = std::find(m_indexToEdgeID.begin(), m_indexToEdgeID.end(), id) - m_indexToEdgeID.begin();
      if (pos==nE || !m_isBoundaryEdge[pos]) err << "  ice margin edge " << id << " is not a local boundary edge\n";
    }

    // Dirichlet ids must be ids of local 3D nodes
    const int vertexColumnShift = (m_ordering==1) ? 1 : m_globalVerticesStride;
    const int vertexLayerShift  = (m_ordering==0) ? 1 : m_numLayers+1;
    std::set<int> local3D;
    for (int ib=0; ib<nV; ++ib) {
      for (int il=0; il<=m_numLayers; ++il) {
        local3D.insert(il*vertexColumnShift + vertexLayerShift*(m_indexToVertexID[ib]-1) + 1);
      }
    }
    for (int id : m_dirichletNodesIds) {
      if (local3D.count(id)==0) err << "  Dirichlet node " << id << " is not a local node\n";
    }
  }

  // Cross-rank checks: every rank storing a vertex lists all the other ranks storing it, and all agree on its owner
  int nLocal = m_active ? numVertices() : 0;
  std::vector<int> counts(worldSize), displs(worldSize,0);
  MPI_Allgather(&nLocal, 1, MPI_INT, counts.data(), 1, MPI_INT, m_worldComm);
  for (int r=1; r<worldSize; ++r) displs[r] = displs[r-1]+counts[r-1];
  const int nAll = displs.back()+counts.back();
  std::vector<int> allIds(nAll), allOwners(nAll);
  std::vector<int> myIds(m_active ? m_indexToVertexID : std::vector<int>());
  std::vector<int> myOwners(m_active ? m_vertexProcIDs : std::vector<int>());
  MPI_Allgatherv(myIds.data(), nLocal, MPI_INT, allIds.data(), counts.data(), displs.data(), MPI_INT, m_worldComm);
  MPI_Allgatherv(myOwners.data(), nLocal, MPI_INT, allOwners.data(), counts.data(), displs.data(), MPI_INT, m_worldComm);
  if (m_active) {
    std::map<int,std::set<int>> ranksOfVertex;
    std::map<int,std::set<int>> ownersOfVertex;
    for (int r=0; r<worldSize; ++r) {
      for (int k=displs[r]; k<displs[r]+counts[r]; ++k) {
        ranksOfVertex[allIds[k]].insert(r-m_numIdleRanks);
        ownersOfVertex[allIds[k]].insert(allOwners[k]);
      }
    }
    for (int ib=0; ib<numVertices(); ++ib) {
      const int id = m_indexToVertexID[ib];
      auto others = ranksOfVertex[id];
      others.erase(m_reducedRank);
      const std::set<int> listed(m_procsSharingVertices[ib].begin(), m_procsSharingVertices[ib].end());
      if (others!=listed) err << "  ranks sharing vertex " << id << " are not consistent across ranks\n";
      if (ownersOfVertex[id].size()!=1) err << "  ranks storing vertex " << id << " disagree on its owner\n";
      if (ranksOfVertex[id].count(m_vertexProcIDs[ib])==0) err << "  owner of vertex " << id << " does not store it\n";
    }
  }

  int nTria = m_active ? numTriangles() : 0, nTriaTot = 0;
  MPI_Allreduce(&nTria, &nTriaTot, 1, MPI_INT, MPI_SUM, m_worldComm);
  if (nTriaTot!=m_numGlobalTriangles) err << "  the ranks store " << nTriaTot << " triangles instead of " << m_numGlobalTriangles << "\n";

  int localFail = err.str().empty() ? 0 : 1, globalFail = 0;
  MPI_Allreduce(&localFail, &globalFail, 1, MPI_INT, MPI_MAX, m_worldComm);
  if (localFail) return "MockMPAS consistency check failed on world rank " + std::to_string(m_worldRank) + ":\n" + err.str();
  if (globalFail) return "MockMPAS consistency check failed on another rank.\n";
  return "";
}

} // namespace mock_mpas
