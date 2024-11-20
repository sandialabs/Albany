//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DUMMY_MESH_HPP
#define ALBANY_DUMMY_MESH_HPP

#include "Albany_ExtrudedMesh.hpp"

#include <numeric>

namespace Albany {

// This dummy mesh in 2d or 3d. The 2d part is made up of triangles,
// obtained by dividing a strip of squares into triangle pairs,like so
//
//   +--+--+--+
//   | /| /| /|...
//   |/ |/ |/ |
//   +--+--+--+
// This allows to be able to compute node/edge GIDs from the elem GID. In particular:
//  - node gids are left to right, bottom row first
//  - edge gids are left to right, bottom row first, then top, then vertical, then oblique
//  - face gids are left to right, square by square, with upside-down triangle first
// For 3d mesh, we number all horiz entities first, bottom to top, and then all vertical ones,
// using same ordering as in 2d
struct DummyMesh : public AbstractMeshStruct {
public:
  DummyMesh (const int ne_x, const int ne_z,
             const Teuchos::RCP<const Teuchos_Comm>& comm)
   : m_mesh2d(new DummyMesh(ne_x,comm))
   , m_num_local_nodes (m_mesh2d->get_num_local_nodes()*(ne_z+1))
   , m_num_local_elems (m_mesh2d->get_num_local_elements()*ne_z)
  {
    const auto ctd = shards::getCellTopologyData<shards::Wedge<6>>();
    meshSpecs.resize(1);
    meshSpecs[0] = Teuchos::rcp(new MeshSpecsStruct(
          MeshType::Structured, *ctd,
          ctd->dimension, {}, {}, 1000, "eb0", { {"eb0",0} }));

    m_num_global_elems = m_mesh2d->num_global_elems()*ne_z;
    m_num_global_nodes = m_mesh2d->num_global_nodes()*(ne_z+1);
    m_num_global_edges = m_mesh2d->num_global_edges()*(ne_z+1)
                       + m_mesh2d->num_global_nodes()*ne_z;
    m_num_global_faces = m_mesh2d->num_global_elems()*(ne_z+1)
                       + m_mesh2d->num_global_edges()*ne_z;

    m_my_elems.resize(m_num_local_elems);

    auto ordering = LayeredMeshOrdering::LAYER;
    LayeredMeshNumbering<LO> cell_layers_lid(m_mesh2d->get_num_local_elements(),ne_z,ordering);
    LayeredMeshNumbering<GO> cell_layers_gid(m_mesh2d->num_global_elems(),ne_z,ordering);
    LayeredMeshNumbering<GO> node_layers_gid(m_mesh2d->num_global_nodes(),ne_z+1,ordering);
    LayeredMeshNumbering<GO> edge_layers_gid(m_mesh2d->num_global_edges(),ne_z+1,ordering);
    for (int icol=0; icol<m_mesh2d->get_num_local_elements(); ++icol) {
      const auto& nodes2d = m_mesh2d->elem2node().at(icol);
      const auto& edges2d = m_mesh2d->elem2edge().at(icol);
      GO gelem2d = m_mesh2d->my_elems()[icol];
      for (int ilev=0; ilev<ne_z; ++ilev) {
        LO ie = cell_layers_lid.getId(icol,ilev);
        
        // Nodes: bot face, then top face
        auto& nodes = m_elem2node[ie];
        for (auto n : nodes2d) {
          nodes.push_back(node_layers_gid.getId(n,ilev));
        }
        for (auto n : nodes2d) {
          nodes.push_back(node_layers_gid.getId(n,ilev+1));
        }

        // Edges: bot face edges, then top face edges, then vert edges
        auto& edges = m_elem2edge[ie];
        for (auto e : edges2d) {
          edges.push_back(edge_layers_gid.getId(e,ilev));
        }
        for (auto e : edges2d) {
          edges.push_back(edge_layers_gid.getId(e,ilev+1));
        }
        for (auto n : nodes2d) {
          GO offset = 2*m_mesh2d->num_global_edges();
          GO vertId = node_layers_gid.getId(n,ilev);
          edges.push_back(offset+vertId);
        }

        // Faces: bot, top, then lateral, in same order of edges2d
        auto& faces = m_elem2face[ie];
        faces.push_back(cell_layers_gid.getId(gelem2d,ilev));
        faces.push_back(cell_layers_gid.getId(gelem2d,ilev+1));
        GO offset_vfaces = m_mesh2d->num_global_elems() * (ne_z+1);
        for (auto e : edges2d) {
          faces.push_back(edge_layers_gid.getId(e,ilev) + offset_vfaces);
        }
      }
    }
  }

  DummyMesh (const int ne_x,
             const Teuchos::RCP<const Teuchos_Comm>& comm)
  {
    auto lcl_ne_x = ne_x / comm->getSize();
    auto rem = ne_x % comm->getSize();
    if (comm->getRank()<rem) {
      ++lcl_ne_x;
    }

    m_num_local_nodes = 2*(lcl_ne_x+1);
    m_num_local_elems = 2*lcl_ne_x;

    const auto ctd = shards::getCellTopologyData<shards::Triangle<3>>();
    meshSpecs.resize(1);
    meshSpecs[0] = Teuchos::rcp(new MeshSpecsStruct(
          MeshType::Structured, *ctd,
          ctd->dimension, {}, {}, 1000, "eb0", { {"eb0",0} }));

    GO beg = 0;
    for (int pid=0; pid<comm->getSize(); ++pid) {
      int pid_ne = m_num_local_elems;
      Teuchos::broadcast(*comm,pid,1,&pid_ne);
      if (pid<comm->getRank()) {
        beg += pid_ne/2;
      }
    }
    std::iota(m_my_elems.begin(),m_my_elems.end(),beg);

    GO ngnodes_x = ne_x+1;
    m_num_global_elems = 2*ne_x;
    m_num_global_nodes = 2*(ne_x+1);
    m_num_global_edges = 4*ne_x+1;

    const GO offset_edge_top =   ne_x;
    const GO offset_edge_vrt = 2*ne_x;
    const GO offset_edge_obl = 3*ne_x + 1;
    m_my_elems.resize(m_num_local_elems);
    for (LO isquare=0; isquare<lcl_ne_x; ++isquare) {
      GO gsquare = isquare + beg;
      for (LO isplit : {0,1}) {
        int ie = 2*isquare + isplit;

        auto& nodes = m_elem2node[ie];
        auto& edges = m_elem2edge[ie];
        if (isplit==0) {
          nodes.push_back(gsquare);
          nodes.push_back(gsquare+ngnodes_x);
          nodes.push_back(gsquare+ngnodes_x+1);

          edges.push_back(offset_edge_obl+gsquare);
          edges.push_back(offset_edge_top+gsquare);
          edges.push_back(offset_edge_vrt+gsquare);
        } else {
          nodes.push_back(gsquare);
          nodes.push_back(gsquare+1);
          nodes.push_back(gsquare+ngnodes_x+1);

          edges.push_back(gsquare);
          edges.push_back(offset_edge_vrt+gsquare+1);
          edges.push_back(offset_edge_obl+gsquare);
        }
      }
    }
  }

  const std::vector<GO>& my_elems () const { return m_my_elems; }
  const std::map<LO,std::vector<GO>>& elem2node () const { return m_elem2node; }
  const std::map<LO,std::vector<GO>>& elem2edge () const { return m_elem2edge; }
  const std::map<LO,std::vector<GO>>& elem2face () const { return m_elem2face; }

  GO num_global_nodes () const { return m_num_global_nodes; }
  GO num_global_edges () const { return m_num_global_edges; }
  GO num_global_faces () const { return m_num_global_faces; }
  GO num_global_elems () const { return m_num_global_elems; }

  LO get_num_local_nodes () const override { return m_num_local_nodes; }
  LO get_num_local_elements () const override { return m_num_local_elems; }
  GO get_max_node_gid () const override { return num_global_nodes()-1; }
  GO get_max_elem_gid () const override { return num_global_elems()-1; }
  Teuchos::RCP<AbstractMeshFieldAccessor> get_field_accessor () const override { return Teuchos::null; }

  //! Internal mesh specs type needed
  std::string meshLibName() const override { return "dummy"; }

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& /* comm */,
                     const Teuchos::RCP<StateInfoStruct>& /* sis */) override {}

  void setBulkData(const Teuchos::RCP<const Teuchos_Comm>& /* comm */) override {}

protected:

  // If this mesh is 3d, store the 2d basal mesh, to help numbering
  Teuchos::RCP<DummyMesh> m_mesh2d;

  std::map<LO,std::vector<GO>> m_elem2node;
  std::map<LO,std::vector<GO>> m_elem2edge;
  std::map<LO,std::vector<GO>> m_elem2face;
  std::vector<GO>              m_my_elems;

  GO  m_num_global_nodes = 0;
  GO  m_num_global_edges = 0;
  GO  m_num_global_faces = 0;
  GO  m_num_global_elems = 0;
  LO  m_num_local_nodes  = 0;
  LO  m_num_local_elems  = 0;
};

} // Namespace Albany

#endif // ALBANY_DUMMY_MESH_HPP
