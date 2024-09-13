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

  auto layers_data = mesh->cell_layers_lid();
  m_num_elems = layers_data->numHorizEntities*layers_data->numLayers;

  m_elem_blocks_names.resize(1, mesh->meshSpecs[0]->ebName);
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
  auto layers_data = m_mesh->cell_layers_gid();
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
  TEUCHOS_TEST_FOR_EXCEPTION (not is_connectivity_built(), std::logic_error,
      "Error! Cannot call getConnectivity before connectivity is build.\n");
  return m_connectivity.data() + getConnectivityStart(ielem);
}

std::vector<int>
ExtrudedConnManager::
getConnectivityMask (const std::string& sub_part_name) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not is_connectivity_built(), std::logic_error,
      "Error! Cannot call getConnectivityMask before connectivity is build.\n");
  std::vector<int> mask(m_connectivity.size(),0);
  const auto& cell_layers_lid = m_mesh->cell_layers_lid();
  const int num_basal_elems = m_mesh->basal_mesh()->get_num_local_elements();
  const int num_layers = cell_layers_lid->numLayers;
  if (sub_part_name=="basalside" or sub_part_name=="upperside" or
      sub_part_name=="bottom" or sub_part_name=="top") {
    const int ilay = sub_part_name=="basalside" or sub_part_name=="bottom" ? 0 : num_layers-1;
    int num_side_dofs = m_num_dofs_per_elem / m_num_vdofs_per_elem;

    const int lid_offset = ilay==0 ? 0 : num_side_dofs*(m_num_vdofs_per_elem-1);
    for (int ie=0; ie<num_basal_elems; ++ie) {
      const int ielem3d = cell_layers_lid->getId(ie,ilay);
      auto m = &mask[getConnectivityStart(ielem3d)];
      for (int idof=0; idof<num_side_dofs; ++idof) {
        m[lid_offset+idof] = 1;
      }
    }
  } else if (sub_part_name=="lateral" or sub_part_name=="lateralside" or
             sub_part_name.substr(0,9)=="extruded_") {
    std::vector<std::string> basal_ss_names;
    if (sub_part_name=="lateral" or sub_part_name=="lateralside") {
      basal_ss_names = m_mesh->basal_mesh()->meshSpecs[0]->ssNames;
    } else {
      basal_ss_names.push_back(m_mesh->get_basal_part_name(sub_part_name));
    }
    LayeredMeshNumbering<LO> dofs_layers(m_num_hdofs_per_elem*m_num_fields,m_num_vdofs_per_elem,LayeredMeshOrdering::LAYER);
    for (const auto& basal_ssn : basal_ss_names) {
      const auto basal_mask = m_conn_mgr_h->getConnectivityMask(basal_ssn);
      for (int ie=0; ie<num_basal_elems; ++ie) {
        const int start_h = m_conn_mgr_h->getConnectivityStart(ie);
        const int size_h  = m_conn_mgr_h->getConnectivitySize(ie);
        for (int idof=0; idof<size_h; ++idof) {
          if (basal_mask[start_h+idof]==1) {
            for (int ilay=0; ilay<num_layers; ++ilay) {
              const int ielem3d = cell_layers_lid->getId(ie,ilay);
              const int start3d = getConnectivityStart(ielem3d);
              auto m = &mask[start3d];
              for (int ilev=0; ilev<m_num_vdofs_per_elem; ++ilev) {
                const int lid = dofs_layers.getId(idof,ilev);
                m[lid] = 1;
              }
            }
          }
        }
      }
    }
  } else if (sub_part_name.substr(0,5)=="basal") {
    std::cout << "basal NS not yet implemented, but not erroring out for now...\n";
  } else {
    throw NotYetImplemented("ExtrudedConnManager::getConnectivityMask for generic sub-part");
  }
  return mask;
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

  using basis_type = Intrepid2::Basis<PHX::Device,RealType,RealType>;
  using tensor_basis_type = Intrepid2::Basis_TensorBasis<basis_type>;

  auto fp_rcp = Teuchos::rcpFromRef(fp);
  auto fp_agg = Teuchos::rcp_dynamic_cast<const panzer::FieldAggPattern>(fp_rcp,true);

  // Unfortunately, FieldAggPattern does not expose how many fields are in it,
  // so use try/catch block with one of the getters;
  m_num_fields = 0;
  while (true) {
    try {
      fp_agg->getFieldType(m_num_fields);
      ++m_num_fields;
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
  for (int ifield=0; ifield<m_num_fields; ++ifield) {
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

  auto cell_layers_gid = m_mesh->cell_layers_gid();
  auto cell_layers_lid = m_mesh->cell_layers_lid();

  // Build horiz and vertical intrepid patterns.
  // TODO: if you generalize to fields of different patterns,
  //       you'll need a vector of patterns (one per field)
  auto fp_v = Teuchos::rcp(new IFP (basis_v));
  auto fp_h = Teuchos::rcp(new IFP (basis_h));

  // Build SCALAR horiz connectivity
  // NOTE: if you generalize to fields of different patterns, fp_h will be a vector
  m_conn_mgr_h->buildConnectivity(fp_h);

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
  GO numDofLayers = (cell_layers_gid->numLayers+1) * nodeIdCnt_v
                  +  cell_layers_gid->numLayers    * cellIdCnt_v;
  m_num_dofs_layers = numDofLayers;

  m_num_dofs_per_elem = fp.numberIds();

  // The strategy to number dofs is the following:
  //  1. loop over num belems and num levs, compute ielem3d with layered data
  //  2. compute horiz conn for that ibelem
  //  3. add dofs for 3d elem on a per-layer basis (local ordering), but honoring
  //     the user requests when it comes to global ids.
  LayeredMeshNumbering<GO> dofs_layers_data(m_num_fields*(max_gid_h+1),numDofLayers,cell_layers_gid->ordering);

  const int ndofs_v = fp_v->numberIds();
  const int ndofs_h = fp_h->numberIds();
  m_ownership.resize(m_num_elems*m_num_dofs_per_elem);
  m_connectivity.resize(m_num_elems*m_num_dofs_per_elem);
  for (int ibelem=0; ibelem<m_mesh->basal_mesh()->get_num_local_elements(); ++ibelem) {
    auto conn_h = m_conn_mgr_h->getConnectivity(ibelem);
    auto ownership_h = m_conn_mgr_h->getOwnership(ibelem);

    auto add_layer_conn = [&](const int dof_layer, int elem_offset,
                              Ownership* ownership, GO* connectivity) {
      for (int idof_h=0; idof_h<ndofs_h; ++idof_h) {
        for (int ifield=0; ifield<m_num_fields; ++ifield) {
          auto dof2d = conn_h[idof_h]*m_num_fields+ifield;
          connectivity[elem_offset+idof_h*m_num_fields+ifield] = dofs_layers_data.getId(dof2d,dof_layer);
          ownership[elem_offset+idof_h*m_num_fields+ifield] = ownership_h[idof_h*m_num_fields+ifield];
        }
      }
    };

    for (int ilay=0; ilay<cell_layers_lid->numLayers; ++ilay) {
      int ielem = cell_layers_lid->getId(ibelem,ilay);

      const GO ielem_offset = ielem*m_num_dofs_per_elem;
      auto conn3d = &m_connectivity[ielem_offset];
      for (int ilev_dof=0; ilev_dof<ndofs_v; ++ilev_dof) {
        int dof_lev = ilay*cellIdCnt_v+ilay*nodeIdCnt_v + ilev_dof;
        int elem_offset = ilev_dof*ndofs_h*m_num_fields;
        add_layer_conn(dof_lev, elem_offset,
                       &m_ownership[ielem_offset],
                       &m_connectivity[ielem_offset]);
      }
    }
  }

  m_num_vdofs_per_elem = ndofs_v;
  m_num_hdofs_per_elem = ndofs_h;
}

} // namespace Albany
