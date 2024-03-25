#include "Albany_ExtrudedConnManager.hpp"

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
  auto panzer_conn_mgr_h = m_conn_mgr_h->noConnectivityClone();
  auto conn_mgr_h = Teuchos::rcp_dynamic_cast<ConnManager>(panzer_conn_mgr_h);
  return Teuchos::rcp(new ExtrudedConnManager(conn_mgr_h,m_mesh));
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
  // To build the 3d connectivity we make a few assumptions:
  //   - the input pattern is a FieldAggPattern
  //   - all patterns in the aggregate pattern are the same
  //   - all patterns in the aggregate pattern are Intrepid2FieldPattern
  //   - the basis in the Intrepid2FieldPattern is a tensor basis
  // We may remove the second assumption at some point, which means some parts
  // of the impl below will have to change.

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

  using IFP = panzer::Intrepid2FieldPattern;
  const auto line = shards::CellTopology(shards::getCellTopologyData<shards::Line<2>>());
  auto basis2fp = [](const Teuchos::RCP<basis_type>& basis) {
    return Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  };

  // Check that all fields have the same pattern
  Teuchos::RCP<const IFP> intrepid_fp;
  for (int ifield=0; ifield<nfields; ++ifield) {
    auto fpi = fp_agg->getFieldPattern(ifield);
    auto this_fp = Teuchos::rcp_dynamic_cast<const IFP>(fpi,true);

    // TODO: allowing fields with different patterns would be nice, but much more complicated.
    //       For now, require same pattern, and move on. Later, we may generalize.
    TEUCHOS_TEST_FOR_EXCEPTION (
        intrepid_fp!=Teuchos::null and this_fp!=intrepid_fp and not this_fp->equals(*intrepid_fp),
        std::runtime_error,
        "ExtrudedConnManager expects a FieldPattern consisting of N copies of the same Intrepid2FieldPattern.\n");
    intrepid_fp = this_fp;
  }

  // Now build the horiz/vert patterns
  auto basis = intrepid_fp->getIntrepidBasis();
  auto tbasis = Teuchos::rcp_dynamic_cast<tensor_basis_type>(basis,true);
  auto comps = tbasis->getTensorBasisComponents();

  TEUCHOS_TEST_FOR_EXCEPTION (comps.size()!=2, std::runtime_error,
      "ExtrudedConnManager can only handle a tensor basis with 2 tensor components.\n"
      "  - tensor basis name  : " << tbasis->getName() << "\n"
      "  - num tensorial comps: " << comps.size() << "\n");

  auto basis_h = comps[0];
  auto basis_v = comps[1];
  TEUCHOS_TEST_FOR_EXCEPTION (basis_v->getBaseCellTopology()!=line, std::runtime_error,
      "ExtrudedConnManager expects a tensor product of the form BasisBasal X Line.\n"
      "  - vert basis name: " << basis_v->getName() << "\n"
      "  - vert basis base cell topo: " << basis_v->getBaseCellTopology().getName() << "\n");

  const auto& basalShape = basis_h->getBaseCellTopology();
  TEUCHOS_TEST_FOR_EXCEPTION (basalShape.getKey()!=m_conn_mgr_h->get_topology().getKey(), std::invalid_argument,
      "[ExtrudedConnManager] Intrepid field pattern is incompatible with basal conn manager:\n"
      "  - basal conn manager cell topo : " << m_conn_mgr_h->get_topology().getName() << "\n"
      "  - field pattern basal cell topo: " << basalShape.getName() << "\n");

  auto layers_data_gid = m_mesh->layers_data_gid();
  auto layers_data_lid = m_mesh->layers_data_lid();

  // Build horiz and vertical intrepid patterns.
  // TODO: if you generalize to fields of different patterns,
  //       you'll need a vector of patterns (one per field)
  auto fp_v = Teuchos::rcp(new IFP (basis_v));
  auto fp_h = Teuchos::rcp(new IFP (basis_h));

  // Build SCALAR horiz connectivity
  // NOTE: if you generalize to fields of different patterns, fp_h will be a vector
  m_conn_mgr_h->buildConnectivity(*fp_h);

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

  // These are either 0 or 1, since fp_v is a scalar FP
  // NOTE: if you generalize to fields with different vert pattern, you will
  //       have to turn these into vectors
  const LO nodeIdCnt_v = fp_v->getSubcellIndices(0,0).size();
  const LO cellIdCnt_v = fp_v->getSubcellIndices(1,0).size();

  // Total number of dofs in "vertical grid" (for a scalar field)
  GO numDofLayers = (layers_data_gid->numLayers+1) * nodeIdCnt_v
                  +  layers_data_gid->numLayers    * cellIdCnt_v;

  // const LO ndofs_v = m_conn_mgr_v->getConnectivitySize(0);
  m_num_dofs_per_elem = fp.numberIds();

  // The strategy to number dofs is the following:
  //  1. use cell layers data (LO) to get icol/ilev of element
  //  2. compute horiz/vert conn for that icol/ilev
  //  3. add dofs for node, edges, faces, elem (if any). For each of these
  //    a. get entity local id from topo
  //    b. get basal entity id and layer id from shards
  //    c. get dof id using basal/layer id from b.
  // The following layer numbering object is for part b and c of step 3

  // LayeredMeshNumbering<LO> shards_layers_data(ndofs_h,ndofs_v,layers_data_gid->ordering);
  LayeredMeshNumbering<GO> dofs_layers_data(nfields*(max_gid_h+1),numDofLayers,layers_data_gid->ordering);

  const int ndofs_v = fp_v->numberIds();
  const int ndofs_h = fp_h->numberIds();
  for (int ielem=0; ielem<m_num_elems; ++ielem) {
    int icol,ilev;
    layers_data_lid->getIndices(ielem,icol,ilev);

    // Resize connectivity, and get pointer to current elem connectivity
    const GO ielem_offset = m_connectivity.size();
    m_connectivity.resize(ielem_offset + m_num_dofs_per_elem);
    auto conn3d = m_connectivity.data()+ielem_offset;
    auto conn_h = m_conn_mgr_h->getConnectivity(icol);

    // We want extruded dofs to be added layer by layer, with layers
    // ordered in ascending order of z coordinate
    auto add_layer_conn = [&](const int dof_layer, const int offset3d) {
      // int v_offset = nfields*ndofs_h*v_idx;
      for (int idof_h=0; idof_h<ndofs_h; ++idof_h) {
        // auto idx = dofs_layers_data.getId(conn_h[idof_h],v_idx);
        for (int ifield=0; ifield<nfields; ++ifield) {
          auto dof2d = conn_h[idof_h]*nfields+ifield;
          conn3d[offset3d+idof_h*nfields+ifield] = dofs_layers_data.getId(dof2d,dof_layer);
        }
      }
    };

    for (int ilay=0; ilay<ndofs_v; ++ilay) {
      int dof_lev = ilev*cellIdCnt_v+ilev*nodeIdCnt_v + ilay;
      int offset3d = ilay*ndofs_h*nfields;
      add_layer_conn(dof_lev,offset3d);
    }
  }

  m_is_connectivity_built = true;
}

} // namespace Albany
