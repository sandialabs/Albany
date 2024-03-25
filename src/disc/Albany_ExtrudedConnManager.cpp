#include "Albany_ExtrudedConnManager.hpp"
#include "Albany_SerialConnManager1d.hpp"

#include <Panzer_FieldAggPattern.hpp>
#include <Panzer_IntrepidFieldPattern.hpp>
#include <Intrepid2_TensorBasis.hpp>

namespace Albany {

ExtrudedConnManager::
ExtrudedConnManager(const Teuchos::RCP<ConnManager>&         basal_conn_mgr,
                    const Teuchos::RCP<const ExtrudedMesh>&  mesh)
 : m_basal_conn_mgr (basal_conn_mgr)
 , m_mesh(mesh)
{
  TEUCHOS_TEST_FOR_EXCEPTION (basal_conn_mgr.is_null(), std::invalid_argument,
      "[ExtrudedConnManager] Error! Invalid basal conn manager pointer.\n");

  const auto tri_topo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3>>());
  TEUCHOS_TEST_FOR_EXCEPTION (
      basal_conn_mgr->get_topology().getName()==std::string(tri_topo.getName()), std::runtime_error,
      "[ExtrudedConnManager::getElementBlockTopologies] Unsupported basal topology.\n"
      "  Basal topology: " << basal_conn_mgr->get_topology().getName() << "\n"
      "  Supported basal topologies: " << tri_topo.getName() << "\n");

  auto layers_data = mesh->layers_data_lid();
  m_num_elems = layers_data->numHorizEntities*layers_data->numLayers;
}


Teuchos::RCP<panzer::ConnManager>
ExtrudedConnManager::noConnectivityClone() const
{
  return Teuchos::rcp(new ExtrudedConnManager(m_basal_conn_mgr,m_mesh));
}

std::vector<GO>
ExtrudedConnManager::
getElementsInBlock (const std::string& blockId) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (blockId!=m_elem_blocks_names[0],std::logic_error,
      "[ExtrudedConnManager::getElementBlock] Error! Invalid elem block name: " + blockId + ".\n");

  const auto& elems_basal = m_basal_conn_mgr->getElementBlock();

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
      return m_basal_conn_mgr->part_dim(m_mesh->get_basal_part_name(part_name));
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
  using fa_tuple_t = std::vector<std::tuple<int,panzer::FieldType,Teuchos::RCP<const panzer::FieldPattern> > >;
  fa_tuple_t horiz_patterns, vert_patterns;
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

    TEUCHOS_TEST_FOR_EXCEPTION (tbasis->getNumTensorialExtrusions()!=1, std::runtime_error,
        "ExtrudedConnManager can only handle a single tensor extrusion");

    auto comps = tbasis->getTensorBasisComponents();
    auto horiz_basis = comps[0];
    auto vert_basis  = comps[1];
    TEUCHOS_TEST_FOR_EXCEPTION (vert_basis->getBaseCellTopology()==line, std::runtime_error,
        "ExtrudedConnManager expects a tensor product of the form BasisBasal X Line");

    const auto& basalShape = horiz_basis->getBaseCellTopology();
    TEUCHOS_TEST_FOR_EXCEPTION (basalShape==m_basal_conn_mgr->get_topology(), std::invalid_argument,
        "[ExtrudedConnManager] Intrepid field pattern for field " << ifield << " is invalid\n"
        "  - basal conn manager cell topo : " << m_basal_conn_mgr->get_topology().getName() << "\n"
        "  - field pattern basal cell topo: " << basalShape.getName() << "\n");

    horiz_patterns.emplace_back(ifield,CG,basis2fp(horiz_basis));
    vert_patterns.emplace_back(ifield,CG,basis2fp(vert_basis));
  }

  auto layers_data_gid = m_mesh->layers_data_gid();
  auto layers_data_lid = m_mesh->layers_data_lid();

  // Create a serial 1d conn mgr for vertical
  auto vert_conn_mgr = Teuchos::rcp(new SerialConnManager1d(layers_data_gid->numLayers));

  // Build horiz and vertical connectivities
  panzer::FieldAggPattern vert_fp(vert_patterns); 
  panzer::FieldAggPattern horiz_fp(horiz_patterns); 
  vert_conn_mgr->buildConnectivity(vert_fp);
  m_basal_conn_mgr->buildConnectivity(horiz_fp);

  // Compute basal max gid
  const auto& basal_elems = m_basal_conn_mgr->getElementsInBlock();
  const int num_basal_elems = basal_elems.size();
  GO my_max_gid = 0;
  for (int ie=0; ie<num_basal_elems; ++ie) {
    const int ndofs = m_basal_conn_mgr->getConnectivitySize(ie);
    const GO* dofs  = m_basal_conn_mgr->getConnectivity(ie);
    for (int idof=0; idof<ndofs; ++idof) {
      my_max_gid = std::max(my_max_gid,dofs[idof]);
    }
  }
  GO max_gid_h;
  auto comm = m_mesh->comm();
  Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&my_max_gid,&max_gid_h);

  GO num_gids_v = (layers_data_gid->numLayers+1) * vert_fp.getSubcellIndices(0,0).size()
                + layers_data_gid->numLayers * vert_fp.getSubcellIndices(1,0).size();

  // We assume same number of dofs in all cells!
  const LO ndofs_horiz = m_basal_conn_mgr->getConnectivitySize(0);
  const LO ndofs_vert  = vert_conn_mgr->getConnectivitySize(0);
  m_num_dofs_per_elem = ndofs_horiz*ndofs_vert;

  // The strategy to number dofs is the following:
  //  1. use cell layers data (LO) to get icol/ilev of element
  //  2. compute horiz/vert conn for that icol/ilev
  //  3. add dofs for node, edges, faces, elem (if any). For each of these
  //    a. get entity local id from topo
  //    b. get basal entity id and layer id from shards
  //    c. get dof id using basal/layer id from b.
  // The following two layers numbering objects are for part b and c of step 3

  LayeredMeshNumbering<LO> shards_layers_data(ndofs_horiz,ndofs_vert,layers_data_gid->ordering);
  LayeredMeshNumbering<GO> dofs_layers_data(max_gid_h,num_gids_v,layers_data_gid->ordering);

  const auto& topo = get_topology();
  const int dim = topo.getDimension();
  int ih, iv;
  for (int ielem=0; ielem<m_num_elems; ++ielem) {
    int icol,ilev;
    layers_data_lid->getIndices(ielem,icol,ilev);
    m_connectivity.reserve(m_connectivity.size()+ndofs_horiz*ndofs_vert);

    auto horiz_conn = m_basal_conn_mgr->getConnectivity(icol);
    auto vert_conn  = vert_conn_mgr->getConnectivity(icol);

    // Things may seem "weird", but work out. Say you have P1Tria X P2Line.
    // The numbering is [0,1,2] at bot, [3,4,5] at top, and [6,7,8] at middle.
    // Say ordering by layer. When you ask shards_layers_data the col/lev ids for, say,
    // 7, you get col=1,lev=2. But that's consistent with vert_conn, since the middle
    // layer correspond to the middle node of P2Line, which has ordinal=2.
    if (dim>0) {
      // Add node indices
      for (int inode=0; inode<topo.getNodeCount(); ++inode) {
        auto ids = fp.getSubcellIndices(0,inode);
        for (auto id : ids) {
          shards_layers_data.getIndices(id,ih,iv);
          m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
        }
      }
    }

    if (dim>1) {
      // Add edge indices
      for (int iedge=0; iedge<topo.getEdgeCount(); ++iedge) {
        auto ids = fp.getSubcellIndices(1,iedge);
        for (auto id : ids) {
          shards_layers_data.getIndices(id,ih,iv);
          m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
        }
      }
    }

    if (dim>2) {
      // Add face indices
      for (int iface=0; iface<topo.getFaceCount(); ++iface) {
        auto ids = fp.getSubcellIndices(2,iface);
        for (auto id : ids) {
          shards_layers_data.getIndices(id,ih,iv);
          m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
        }
      }
    }

    // Add cell indices
    auto ids = fp.getSubcellIndices(dim,0);
    for (auto id : ids) {
      shards_layers_data.getIndices(id,ih,iv);
      m_connectivity.push_back(dofs_layers_data.getId(horiz_conn[ih],vert_conn[iv]));
    }
  }
}

} // namespace Albany
