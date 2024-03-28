#include "Albany_ExtrudedConnManager.hpp"
#include "Albany_SerialConnManager1d.hpp"

#include <Panzer_FieldAggPattern.hpp>
#include <Panzer_GeometricAggFieldPattern.hpp>
#include <Panzer_IntrepidFieldPattern.hpp>
#include <Intrepid2_TensorBasis.hpp>

namespace Albany {

ExtrudedConnManager::
ExtrudedConnManager(const Teuchos::RCP<ConnManager>&         conn_mgr_h,
                    const Teuchos::RCP<const ExtrudedMesh>&  mesh)
 : m_conn_mgr_h (conn_mgr_h)
 , m_mesh(mesh)
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_conn_mgr_h.is_null(), std::invalid_argument,
      "[ExtrudedConnManager] Error! Invalid basal conn manager pointer.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (m_mesh.is_null(), std::invalid_argument,
      "[ExtrudedConnManager] Error! Invalid extruded mesh pointer.\n");

  const auto tri_topo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3>>());
  TEUCHOS_TEST_FOR_EXCEPTION (
      m_conn_mgr_h->get_topology().getName()!=std::string(tri_topo.getName()), std::runtime_error,
      "[ExtrudedConnManager::getElementBlockTopologies] Unsupported basal topology.\n"
      "  Basal topology: " << m_conn_mgr_h->get_topology().getName() << "\n"
      "  Supported basal topologies: " << tri_topo.getName() << "\n");

  auto layers_data = mesh->layers_data_lid();
  m_num_elems = layers_data->numHorizEntities*layers_data->numLayers;
}

Teuchos::RCP<panzer::ConnManager>
ExtrudedConnManager::noConnectivityClone() const
{
  return Teuchos::rcp(new ExtrudedConnManager(m_conn_mgr_h,m_mesh));
}

std::vector<GO>
ExtrudedConnManager::
getElementsInBlock (const std::string& blockId) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (blockId!=m_elem_blocks_names[0],std::logic_error,
      "[ExtrudedConnManager::getElementBlock] Error! Invalid elem block name: " + blockId + ".\n");

  const auto& elems_basal = m_conn_mgr_h->getElementBlock();

  std::vector<GO> elems;
  auto layers_data = m_mesh->layers_data_lid();
  elems.reserve(m_num_elems);
  for (auto bgid : elems_basal) {
    for (int ilayer=0; ilayer<layers_data->numLayers; ++ilayer) {
      elems.push_back (layers_data->getId(bgid,ilayer));
    }
  }
  return elems;
}

std::string ExtrudedConnManager::
getBlockId(LO localElmtId) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (localElmtId<0 or localElmtId>m_num_elems, std::runtime_error,
      "[ExtrudedConnManager::getBlockId] Element LID (" << localElmtId << ") out of bounds [0," << m_num_elems << ")\n");

  return elem_block_name();
}

void ExtrudedConnManager::
getElementBlockTopologies(std::vector<shards::CellTopology>& elementBlockTopologies) const
{
  const auto& topo = shards::CellTopology(&m_mesh->meshSpecs[0]->ctd);
  elementBlockTopologies.resize(numElementBlocks(),topo);
}

const std::vector<LO>&
ExtrudedConnManager::getElementBlock(const std::string& blockId) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (blockId!=elem_block_name(),std::logic_error,
      "[ExtrudedConnManager::getElementBlock] Error! Invalid elem block name: " + blockId + ".\n");

  return m_elem_lids;
}

const GO* ExtrudedConnManager::
getConnectivity (const LO ielem) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_is_connectivity_built, std::logic_error,
      "Error! Cannot call getConnectivity before connectivity is build.\n");
  return m_connectivity.data() + getConnectivityStart(ielem);
}

std::vector<int>
ExtrudedConnManager::
getConnectivityMask (const std::string& sub_part_name) const
{
  std::vector<int> m;
  TEUCHOS_TEST_FOR_EXCEPTION (true, NotYetImplemented,
      "ExtrudedConnManager::getConnectivityMask");
  return m;
}

int ExtrudedConnManager::
part_dim (const std::string& part_name) const
{
  const auto& ms = m_mesh->meshSpecs[0];
  const auto& ss_names = ms->ssNames;
  if (part_name==elem_block_name()) {
    return ms->numDim;
  } else if (std::find(ss_names.begin(),ss_names.end(),part_name)!=ss_names.end()) {
    return ms->numDim - 1;
  } else {
    try {
      return m_conn_mgr_h->part_dim(m_mesh->get_basal_part_name(part_name));
    } catch (...) {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
          "[ExtrudedConnManager::part_dim] Invalid part name: " + part_name + "\n");
    }
  }
}
  
const Ownership*
ExtrudedConnManager::getOwnership(LO localElmtId) const
{
  return m_ownership.data() + getConnectivityStart(localElmtId);
}

void ExtrudedConnManager::
buildConnectivity(const panzer::FieldPattern & fp)
{
  TEUCHOS_TEST_FOR_EXCEPTION (m_is_connectivity_built, std::logic_error,
      "[ExtrudedConnManager::buildConnectivity] Connectivity was already built.\n");

  using basis_type = Intrepid2::Basis<PHX::Device,RealType,RealType>;
  using tensor_basis_type = Intrepid2::Basis_TensorBasis<basis_type>;

  auto fp_rcp = Teuchos::rcpFromRef(fp);
  auto fp_agg = Teuchos::rcp_dynamic_cast<const panzer::FieldAggPattern>(fp_rcp,true);

  // Unfortunately, FieldAggPattern does not expose how many fields are in it,
  // so use try/catch block with one of the getters;
  int nfields = 0;
  while (true) {
    try {
      fp_agg->getFieldType(nfields);
      ++nfields;
    } catch(...) {
      break;
    }
  };

  Teuchos::RCP<const panzer::Intrepid2FieldPattern> intrepid_fp;
  const auto line = shards::CellTopology(shards::getCellTopologyData<shards::Line<2>>());
  // using fa_tuple_t = std::vector<std::tuple<int,panzer::FieldType,Teuchos::RCP<const panzer::FieldPattern> > >;
  using gfa_vec_t = std::vector<std::pair<panzer::FieldType,Teuchos::RCP<const panzer::FieldPattern>>>;
  gfa_vec_t patterns_h,patterns_v;
  constexpr auto CG = panzer::FieldType::CG;
  auto basis2fp = [](const Teuchos::RCP<basis_type>& basis) {
    return Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  };
  for (int ifield=0; ifield<nfields; ++ifield) {
    auto fpi = fp_agg->getFieldPattern(ifield);
    auto intrepid_fpi = Teuchos::rcp_dynamic_cast<const panzer::Intrepid2FieldPattern>(fpi,true);

    TEUCHOS_TEST_FOR_EXCEPTION (
        intrepid_fp!=Teuchos::null and intrepid_fpi!=intrepid_fp and intrepid_fpi->equals(*intrepid_fp),
        std::runtime_error,
        "ExtrudedConnManager expects a FieldPattern consisting of N copies of the same Intrepid2FieldPattern.\n");
    intrepid_fp = intrepid_fpi;

    // Now build the horiz/vert patterns
    auto basis = intrepid_fpi->getIntrepidBasis();
    auto tbasis = Teuchos::rcp_dynamic_cast<tensor_basis_type>(basis,true);
    auto comps = tbasis->getTensorBasisComponents();

    TEUCHOS_TEST_FOR_EXCEPTION (comps.size()!=2, std::runtime_error,
        "ExtrudedConnManager can only handle a tensor basis with 2 tensor components.\n"
        "  - tensor basis name  : " << tbasis->getName() << "\n"
        "  - num tensorial comps: " << comps.size() << "\n");

    auto basis_h = comps[0];
    auto basis_v  = comps[1];
    TEUCHOS_TEST_FOR_EXCEPTION (basis_v->getBaseCellTopology()!=line, std::runtime_error,
        "ExtrudedConnManager expects a tensor product of the form BasisBasal X Line.\n"
        "  - vert basis name: " << basis_v->getName() << "\n"
        "  - vert basis base cell topo: " << basis_v->getBaseCellTopology().getName() << "\n");

    const auto& basalShape = basis_h->getBaseCellTopology();
    TEUCHOS_TEST_FOR_EXCEPTION (basalShape.getKey()!=m_conn_mgr_h->get_topology().getKey(), std::invalid_argument,
        "[ExtrudedConnManager] Intrepid field pattern for field " << ifield << " is invalid\n"
        "  - basal conn manager cell topo : " << m_conn_mgr_h->get_topology().getName() << "\n"
        "  - field pattern basal cell topo: " << basalShape.getName() << "\n");

    patterns_h.emplace_back(CG,basis2fp(basis_h));
    patterns_v.emplace_back(CG,basis2fp(basis_v));
  }

  auto layers_data_gid = m_mesh->layers_data_gid();
  auto layers_data_lid = m_mesh->layers_data_lid();

  // Create a serial 1d conn mgr for vertical
  m_conn_mgr_v = Teuchos::rcp(new SerialConnManager1d(layers_data_gid->numLayers));

  // Build horiz and vertical connectivities
  panzer::GeometricAggFieldPattern fp_h(patterns_h); 
  panzer::GeometricAggFieldPattern fp_v(patterns_v); 
  m_conn_mgr_h->buildConnectivity(fp_h);
  m_conn_mgr_v->buildConnectivity(fp_v);

  // Compute basal max gid
  const auto& elems_h = m_conn_mgr_h->getElementsInBlock();
  const int nelems_h = elems_h.size();
  GO my_max_gid = 0;
  for (int ie=0; ie<nelems_h; ++ie) {
    const int ndofs = m_conn_mgr_h->getConnectivitySize(ie);
    const GO* dofs  = m_conn_mgr_h->getConnectivity(ie);
    for (int idof=0; idof<ndofs; ++idof) {
      my_max_gid = std::max(my_max_gid,dofs[idof]);
    }
  }
  GO max_gid_h;
  auto comm = m_mesh->comm();
  Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&my_max_gid,&max_gid_h);

  GO num_gids_v = (layers_data_gid->numLayers+1) * nfields
                +  layers_data_gid->numLayers    * nfields;

  // We assume same number of dofs in all cells!
  const LO ndofs_h = m_conn_mgr_h->getConnectivitySize(0);
  const LO ndofs_v = m_conn_mgr_v->getConnectivitySize(0);
  m_num_dofs_per_elem = ndofs_h*ndofs_v;

  // The strategy to number dofs is the following:
  //  1. use cell layers data (LO) to get icol/ilev of element
  //  2. compute horiz/vert conn for that icol/ilev
  //  3. add dofs for node, edges, faces, elem (if any). For each of these
  //    a. get entity local id from topo
  //    b. get basal entity id and layer id from shards
  //    c. get dof id using basal/layer id from b.
  // The following two layers numbering objects are for part b and c of step 3

  LayeredMeshNumbering<LO> shards_layers_data(ndofs_h,ndofs_v,layers_data_gid->ordering);
  LayeredMeshNumbering<GO> dofs_layers_data(max_gid_h+1,num_gids_v,layers_data_gid->ordering);

  const auto& topo_h = m_conn_mgr_h->get_topology();
  int ih, iv;
  for (int ielem=0; ielem<m_num_elems; ++ielem) {
    int icol,ilev;
    layers_data_lid->getIndices(ielem,icol,ilev);

    // Resize connectivity, and get pointer to current elem connectivity
    const GO ielem_offset = m_connectivity.size();
    m_connectivity.resize(ielem_offset + m_num_dofs_per_elem);
    auto ielem_conn = m_connectivity.data()+ielem_offset;

    std::cout << "ExtrudedConnManager, ie=" << ielem << ", icol=" << icol << ", ilev=" << ilev << "\n";
    auto conn_h = m_conn_mgr_h->getConnectivity(icol);
    auto conn_v = m_conn_mgr_v->getConnectivity(ilev);

    // We want extruded dofs to be added layer by layer, with layers
    // ordered exactly as 1d dofs are ordered in fp_v

    for (int idof_v=0; idof_v<ndofs_v; ++idof_v) {
      int v_idx = fp_v.getSubcellIndices(0,idof_v)[0];
      std::cout << "  idof_v=" << idof_v << ", conn_v=" << conn_v[idof_v] << ":\n";
      int v_offset = ndofs_h*v_idx;
      for (int idof_h=0; idof_h<ndofs_h; ++idof_h) {
        for (int ifield=0; ifield<nfields; ++ifield) {
          GO& idx = ielem_conn[v_offset + nfields*idof_h + ifield];
          idx = dofs_layers_data.getId(conn_h[idof_h],conn_v[v_idx]);
          std::cout << "    idof_h=" << idof_h << ", conn_h=" << conn_h[v_idx] << ", dof=" << m_connectivity.back() << "\n";
        }
        // m_connectivity.push_back(dofs_layers_data.getId(conn_h[idof_h],conn_v[v_pos]));
      }
    }

    // // Things may seem "weird", but work out. Say you have P1Tria X P2Line.
    // // The numbering is [0,1,2] at bot, [3,4,5] at top, and [6,7,8] at middle.
    // // Say ordering by layer. When you ask shards_layers_data the col/lev ids for, say,
    // // 7, you get col=1,lev=2. But that's consistent with vert_conn, since the middle
    // // layer correspond to the middle node of P2Line, which has ordinal=2.
    // if (dim>0) {
    //   // Add node indices
    //   std::cout << "  node dofs:";
    //   for (int inode=0; inode<topo.getNodeCount(); ++inode) {
    //     shards_layers_data.getIndices(inode,ih,iv);
    //     auto ids_h = fp_h.getSubcellIndices(0,ih);
    //     auto ids_v = fp_v.getSubcellIndices(0,iv);
    //     for (int id=0; id<nodeIdCnt; ++id) {
    //       m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
    //       printf(" gid(%lld,%lld)=%lld",horiz_conn[ih],vert_conn[iv],m_connectivity.back());
          
    //     }
    //   }
    //   std::cout << "\n";
    // }

    // if (dim>1) {
    //   // Add edge indices
    //   for (int iedge=0; iedge<topo.getEdgeCount(); ++iedge) {
    //     auto ids = fp.getSubcellIndices(1,iedge);
    //     for (auto id : ids) {
    //       shards_layers_data.getIndices(id,ih,iv);
    //       m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
    //     }
    //   }
    // }

    // if (dim>2) {
    //   // Add face indices
    //   for (int iface=0; iface<topo.getFaceCount(); ++iface) {
    //     auto ids = fp.getSubcellIndices(2,iface);
    //     for (auto id : ids) {
    //       shards_layers_data.getIndices(id,ih,iv);
    //       m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
    //     }
    //   }
    // }

    // // Add cell indices
    // auto ids = fp.getSubcellIndices(dim,0);
    // for (auto id : ids) {
    //   shards_layers_data.getIndices(id,ih,iv);
    //   m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
    // }
  }

  m_is_connectivity_built = true;
}

} // namespace Albany
