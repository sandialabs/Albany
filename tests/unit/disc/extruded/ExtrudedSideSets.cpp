//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Side sets of the logically extruded mesh (ExtrudedDiscretization::computeSideSets), with both orderings.
//
// The basal mesh (TestBasalMesh below) has the features that the numbering of the extruded sides used to get wrong:
//  - a boundary side with GID 0. The lateral sides are numbered with a LayeredMeshNumbering of the basal sides
//    GIDs, which was sized with the max basal side GID rather than max GID+1, so, with the layer-wise ordering,
//    the side with the max GID in layer il got the same GID as the side with GID 0 in layer il+1, and one of the
//    two was dropped (as a duplicate);
//  - a side set contained in another one. That numbering was sized for each side set (and rank), so, with the
//    layer-wise ordering, a lateral side got different GIDs in different side sets;
//  - non-contiguous element GIDs. The GIDs of the upper and lateral sides were shifted by the number of basal
//    elements, so they could coincide with the GIDs of other sides.

#include "Albany_UnitTestSession.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_ExtrudedMesh.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_UnitTestHelpers.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>

#include <Shards_BasicTopologies.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int nx = 4;           // squares along x
constexpr int ny = 4;           // squares along y
constexpr int num_tria = 2*nx*ny;
constexpr int num_layers = 3;
constexpr int num_boundary_edges = 2*(nx+ny);
constexpr int num_south_edges = nx;

// Triangulation of [0,nx]x[0,ny]: the unit square (i,j) is split along the diagonal from (i,j) to (i+1,j+1) in
// triangle 0 (below the diagonal) and triangle 1 (above). The ids are those of a grid with nx+1 vertices per row:
//  - vertex (i,j): 1+v, with v = i + (nx+1)*j;
//  - triangle t of square (i,j): 1 + 2*v + t, with v the vertex (i,j). They are not contiguous, since there is
//    no square (nx,j);
//  - edge: 1 + 3*v + k, from vertex v along x (k=0) or along y (k=1), or the diagonal (k=2). The edge from (0,0)
//    along x has id 1 (GID 0), and is a boundary edge.
// The triangles are dealt to the ranks in order of id (strips along y). Side sets: "boundary" (all the boundary
// edges) and "south" (the boundary edges at y=0), contained in "boundary".
class TestBasalMesh : public Albany::GenericSTKMeshStruct
{
public:
  TestBasalMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<const Teuchos_Comm>& comm)
   : GenericSTKMeshStruct(params, 2, 0)
  {
    const int rank = comm->getRank();
    const int size = comm->getSize();
    for (int j=0; j<ny; ++j) {
      for (int i=0; i<nx; ++i) {
        for (int t=0; t<2; ++t) {
          const int k = 2*(i+nx*j) + t;
          const int owner = static_cast<long>(k)*size/num_tria;
          const auto verts = vertices(i,j,t);
          for (int v : verts) {
            m_vertex_procs[v].insert(owner);
          }
          if (owner==rank) {
            m_triangles.push_back({i,j,t});
          }
        }
      }
    }

    const std::string ebn = "Element Block 0";
    partVec.push_back(&metaData->declare_part_with_topology(ebn, stk::topology::TRI_3_2D));
    shards::CellTopology shards_ctd = stk::mesh::get_cell_topology(stk::topology::TRI_3_2D);
    this->addElementBlockInfo(0, ebn, partVec[0], shards_ctd);

    const std::vector<std::string> ssNames = {"boundary", "south"};
    for (const auto& ssn : ssNames) {
      ssPartVec[ssn] = &metaData->declare_part(ssn, metaData->side_rank());
      stk::mesh::set_topology(*ssPartVec[ssn], stk::topology::LINE_2);
    }

    numDim = 2;
    const int worksetSize = Albany::computeWorksetSize(params->get<int>("Workset Size",-1), m_triangles.size());
    const CellTopologyData& ctd = *shards_ctd.getCellTopologyData();
    this->meshSpecs[0] = Teuchos::rcp(
        new Albany::MeshSpecsStruct(Albany::MeshType::Unstructured, ctd, numDim, {}, ssNames,
                                    worksetSize, ebn, ebNameToIndex));
    this->initializeSideSetMeshSpecs(comm);
  }

  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm) override
  {
    const auto SIDE_RANK = metaData->side_rank();
    const int rank = comm->getRank();

    metaData->commit();
    bulkData->modification_begin();

    // Nodes of the local triangles, shared with the other ranks that have triangles containing them
    auto coordinates_field = fieldContainer->getCoordinatesField();
    std::map<int,stk::mesh::Entity> nodes;
    for (const auto& tria : m_triangles) {
      for (int v : vertices(tria[0],tria[1],tria[2])) {
        if (nodes.count(v)==1) continue;
        auto node = nodes[v] = bulkData->declare_node(1+v, stk::mesh::PartVector{});
        for (int proc : m_vertex_procs.at(v)) {
          if (proc!=rank) bulkData->add_node_sharing(node, proc);
        }
        double* coord = stk::mesh::field_data(*coordinates_field, node);
        coord[0] = v % (nx+1);
        coord[1] = v / (nx+1);
      }
    }

    // Triangles and their boundary sides. Side s of a triangle joins its nodes s and (s+1)%3.
    const auto& ctd = this->meshSpecs[0]->ctd;
    for (const auto& tria : m_triangles) {
      const int i = tria[0], j = tria[1], t = tria[2];
      const int v = i + (nx+1)*j;
      const auto verts = vertices(i,j,t);
      auto elem = bulkData->declare_element(1 + 2*v + t, stk::mesh::PartVector{partVec[0]});
      for (int n=0; n<3; ++n) {
        bulkData->declare_relation(elem, nodes.at(verts[n]), n);
      }

      // (side position, edge id, in "south")
      std::vector<std::array<int,3>> sides;
      if (t==0) {
        if (j==0)    sides.push_back({0, 1 + 3*verts[0] + 0, 1});  // along x from (i,j)
        if (i==nx-1) sides.push_back({1, 1 + 3*verts[1] + 1, 0});  // along y from (i+1,j)
      } else {
        if (j==ny-1) sides.push_back({1, 1 + 3*verts[2] + 0, 0});  // along x from (i,j+1)
        if (i==0)    sides.push_back({2, 1 + 3*verts[0] + 1, 0});  // along y from (i,j)
      }
      for (const auto& s : sides) {
        stk::mesh::PartVector parts {ssPartVec["boundary"]};
        if (s[2]==1) parts.push_back(ssPartVec["south"]);
        auto side = bulkData->declare_entity(SIDE_RANK, s[1], parts);
        bulkData->declare_relation(elem, side, s[0]);
        const auto elem_nodes = bulkData->begin_nodes(elem);
        for (int n=0; n<2; ++n) {
          bulkData->declare_relation(side, elem_nodes[ctd.side[s[0]].node[n]], n);
        }
      }
    }

    bulkData->modification_end();

    this->setDefaultCoordinates3d();
    this->loadRequiredInputFields(comm);
    m_bulk_data_set = true;
  }

  bool hasRestartSolution () const override { return false; }
  double restartDataTime () const override { return -1.0; }

private:
  // Grid indices of the vertices of triangle t of square (i,j), counterclockwise
  static std::array<int,3> vertices (const int i, const int j, const int t) {
    const int a = i + (nx+1)*j, b = a+1, c = b+nx+1, d = a+nx+1;
    return t==0 ? std::array<int,3>{a,b,c} : std::array<int,3>{a,c,d};
  }

  std::vector<std::array<int,3>> m_triangles;    // (i,j,t) of the local triangles
  std::map<int,std::set<int>>    m_vertex_procs; // ranks with a triangle containing each vertex
};

Teuchos::RCP<Albany::AbstractDiscretization>
create_extruded_disc (const Teuchos::RCP<const Teuchos_Comm>& comm, const bool columnwise)
{
  using namespace Albany;

  auto top_level_params = Teuchos::rcp(new Teuchos::ParameterList(""));
  auto disc_params = Teuchos::sublist(top_level_params,"Discretization");
  disc_params->set<std::string>("Method", "Extruded");
  disc_params->set<int>("NumLayers", num_layers);
  disc_params->set<int>("Number Of Time Derivatives", 0);
  disc_params->set<int>("Workset Size", -1);
  disc_params->set<bool>("Columnwise Ordering", columnwise);
  disc_params->sublist("Side Set Discretizations").set("Side Sets",Teuchos::Array<std::string>{"basalside"});
  auto basal_params = Teuchos::sublist(Teuchos::sublist(disc_params,"Side Set Discretizations"),"basalside");
  basal_params->set<int>("Number Of Time Derivatives", 0);
  basal_params->set<int>("Workset Size", -1);

  // Thickness and surface height, needed for the coordinates of the 3d mesh
  auto& basal_req = basal_params->sublist("Required Fields Info");
  basal_req.set<int>("Number Of Fields",2);
  int ifield = 0;
  for (const std::string name : {"thickness", "surface_height"}) {
    auto& req = basal_req.sublist("Field " + std::to_string(ifield++));
    req.set<std::string>("Field Name",name);
    req.set<std::string>("Field Type","Node Scalar");
    req.set<std::string>("Field Origin","File");
    req.set<double>("Field Value",1.0);
  }
  std::map<std::string, Teuchos::RCP<StateInfoStruct> > ss_sis;
  auto& bsis = ss_sis["basalside"] = Teuchos::rcp(new StateInfoStruct());
  bsis->push_back(Teuchos::rcp(new StateStruct("thickness",StateStruct::NodalDataToElemNode,{num_tria,3})));
  bsis->push_back(Teuchos::rcp(new StateStruct("surface_height",StateStruct::NodalDataToElemNode,{num_tria,3})));
  auto sis = Teuchos::rcp(new StateInfoStruct());

  auto basal_mesh = Teuchos::rcp(new TestBasalMesh(basal_params,comm));
  auto mesh = Teuchos::rcp(new ExtrudedMesh(basal_mesh,disc_params,comm));

  DiscretizationFactory factory(top_level_params,comm,false);
  factory.createMeshSpecs(mesh);
  return factory.createDiscretization(1,{},sis,ss_sis);
}

} // anonymous namespace

TEUCHOS_UNIT_TEST (ExtrudedSideSets, GIDs)
{
  using namespace Albany;

  auto comm = getDefaultComm();
  const auto mpi_comm = getMpiCommFromTeuchosComm(comm);

  auto global_sum = [&](const long long value) {
    long long result;
    MPI_Allreduce(&value, &result, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm);
    return result;
  };

  for (bool columnwise : {false, true}) {
    out << "  Columnwise Ordering: " << (columnwise ? "true" : "false") << "\n";
    auto disc = create_extruded_disc(comm,columnwise);

    std::map<std::string,std::vector<long long>> gids;
    for (int ws=0; ws<disc->getNumWorksets(); ++ws) {
      for (const auto& ss : disc->getSideSets(ws)) {
        for (const auto& side : ss.second) {
          gids[ss.first].push_back(side.side_GID);
        }
      }
    }

    // Number of sides
    TEST_EQUALITY (global_sum(gids["basalside"].size()), num_tria);
    TEST_EQUALITY (global_sum(gids["upperside"].size()), num_tria);
    TEST_EQUALITY (global_sum(gids["lateralside"].size()), num_layers*num_boundary_edges);
    TEST_EQUALITY (global_sum(gids["extruded_boundary"].size()), num_layers*num_boundary_edges);
    TEST_EQUALITY (global_sum(gids["extruded_south"].size()), num_layers*num_south_edges);

    // The extruded side sets have the lateral sides with the same GIDs as in "lateralside" (on the same rank,
    // since each side belongs to one element)
    const std::set<long long> lateral(gids["lateralside"].begin(), gids["lateralside"].end());
    long long not_lateral = 0;
    for (const std::string ss : {"extruded_boundary", "extruded_south"}) {
      for (auto gid : gids[ss]) {
        not_lateral += lateral.count(gid)==0;
      }
    }
    TEST_EQUALITY (global_sum(not_lateral), 0);

    // All the basal, upper and lateral sides have different GIDs, across all ranks
    std::vector<long long> local;
    for (const std::string ss : {"basalside", "upperside", "lateralside"}) {
      local.insert(local.end(), gids[ss].begin(), gids[ss].end());
    }
    const int size = comm->getSize();
    const int n = local.size();
    std::vector<int> counts(size), displs(size,0);
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi_comm);
    for (int r=1; r<size; ++r) displs[r] = displs[r-1]+counts[r-1];
    std::vector<long long> all(displs.back()+counts.back());
    MPI_Allgatherv(local.data(), n, MPI_LONG_LONG, all.data(), counts.data(), displs.data(), MPI_LONG_LONG, mpi_comm);
    std::sort(all.begin(), all.end());
    const long long num_duplicates = all.end() - std::unique(all.begin(), all.end());
    out << "    duplicated side GIDs: " << num_duplicates << "\n";
    TEST_EQUALITY (num_duplicates, 0);
  }
}
