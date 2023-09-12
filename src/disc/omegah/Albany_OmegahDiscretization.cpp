#include "Albany_OmegahDiscretization.hpp"

#include "OmegahConnManager.hpp"

#include <Panzer_IntrepidFieldPattern.hpp>

namespace Albany {

OmegahDiscretization::
OmegahDiscretization(
  const Teuchos::RCP<Teuchos::ParameterList>& discParams,
  const int                                   neq,
  Teuchos::RCP<OmegahAbstractMesh>&           mesh,
  const Teuchos::RCP<const Teuchos_Comm>&     comm,
  const Teuchos::RCP<RigidBodyModes>& /* rigidBodyModes */,
  const std::map<int, std::vector<std::string>>& sideSetEquations)
 : m_disc_params (discParams)
 , m_mesh_struct(mesh)
 , m_comm (comm)
 , m_side_set_equations(sideSetEquations)
 , m_neq (neq)
{
  m_num_time_deriv = m_disc_params->get("Number Of Time Derivatives",0);

  // TODO: get solution names from param list
  m_sol_names.resize(m_num_time_deriv+1,solution_dof_name());
  if (m_num_time_deriv>0) {
    m_sol_names[1] += "_dot";
    if (m_num_time_deriv>1) {
      m_sol_names[2] += "_dotdot";
    }
  }
}

void OmegahDiscretization::
updateMesh ()
{
  printf ("TODO: change name to the method?\n");
  
  auto sol_dof_mgr  = create_dof_mgr(solution_dof_name(),"",FE_Type::HGRAD,1,m_neq);
  auto node_dof_mgr = create_dof_mgr(nodes_dof_name(),"",FE_Type::HGRAD,1,1);

  m_dof_managers[solution_dof_name()][""] = sol_dof_mgr;
  m_dof_managers[nodes_dof_name()][""]     = node_dof_mgr;
  m_node_dof_managers[""]     = node_dof_mgr;

  // TODO: our Thyra VS is such that owned GIDs come first, so we can store ov fields,
  //       and subview them for non-ov queries
  auto create_fc = [&] (const Teuchos::RCP<const Thyra_VectorSpace>& vs) {
    auto fc = Teuchos::rcp(new OmegahFieldContainer(m_mesh_struct));
    fc->add_field (m_sol_names[0],vs);
    if (m_num_time_deriv>0) {
      fc->add_field (m_sol_names[1],vs);
      if (m_num_time_deriv>1) {
        fc->add_field (m_sol_names[2],vs);
      }
    }
    return fc;
  };
  m_ov_field_container = create_fc (sol_dof_mgr->ov_vs());
  if (m_comm->getSize()>1) {
    m_field_container = create_fc (sol_dof_mgr->ov_vs());
  } else {
    m_field_container = m_ov_field_container;
  }

  // Compute workset information
  // NOTE: these arrays are all of size 1, for the foreseable future.
  //       Still, make impl generic (where possible), in case things change.
  const auto& ms = m_mesh_struct->getMeshSpecs()[0];
  int nelems = m_mesh_struct->getOmegahMesh().nelems();
  int max_ws_size = ms->worksetSize;
  int num_ws = 1 + (nelems-1) / max_ws_size;
  TEUCHOS_TEST_FOR_EXCEPTION (num_ws!=1, std::runtime_error,
      "Error! We are not yet supporting 2+ worksets with Omega_h.\n");

  m_workset_sizes.resize(num_ws);
  int min_ws_size = nelems / num_ws;
  int remainder = nelems % num_ws;
  for (int ws=0;ws<num_ws; ++ws) {
    m_workset_sizes[ws] = min_ws_size + (ws<remainder ? 1 : 0);
  }

  m_workset_elements = DualView<int**>("ws_elems",1,max_ws_size);
  for (int i=0; i<nelems; ++i) {
    m_workset_elements.host()(0,i) = i;
  }
  m_workset_elements.sync_to_dev();

  m_wsEBNames.resize(1,ms->ebName);
  m_wsPhysIndex.resize(num_ws);
  for (int i=0; i<num_ws; ++i) {
    m_wsPhysIndex[i] = ms->ebNameToIndex[m_wsEBNames[i]];
  }

  m_ws_elem_coords.resize(num_ws);
  auto coords_h  = m_mesh_struct->coords_host();
  auto node_gids = Omega_h::HostRead<Omega_h::GO>(m_mesh_struct->getOmegahMesh().globals(0));
  std::vector<LO> lid2pos (node_gids.size());
  auto node_indexer = getOverlapNodeGlobalLocalIndexer();
  for (size_t i=0; i<lid2pos.size(); ++i) {
    auto gid = node_gids[i];
    auto lid = node_indexer->getLocalElement(gid);
    lid2pos[lid] = i;
  }
  int num_elem_nodes = node_dof_mgr->get_topology().getNodeCount();
  const auto& node_elem_dof_lids = node_dof_mgr->elem_dof_lids().host();
  for (int ws=0; ws<num_ws; ++ws) {
    m_ws_elem_coords[ws].resize(m_workset_sizes[ws]);
    for (int ielem=0; ielem<m_workset_sizes[ws]; ++ielem) {
      m_ws_elem_coords[ws][ielem].resize(num_elem_nodes);
      for (int inode=0; inode<num_elem_nodes; ++inode) {
        LO node_lid = node_elem_dof_lids(ielem,inode);
        int omh_pos = lid2pos[node_lid];
        m_ws_elem_coords[ws][ielem][inode] = &coords_h[omh_pos*3];
      }
    }
  }

  computeGraphs ();
}

void OmegahDiscretization::
computeGraphs ()
{
  const auto vs = getVectorSpace();
  const auto ov_vs = getOverlapVectorSpace();
  m_jac_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(vs, vs, ov_vs, ov_vs));

  // Determine which equations are defined on the whole domain,
  // as well as what eqn are on each sideset
  std::vector<int> volumeEqns;
  std::map<std::string,std::vector<int>> ss_to_eqns;
  for (int k=0; k < m_neq; ++k) {
    if (m_side_set_equations.find(k) == m_side_set_equations.end()) {
      volumeEqns.push_back(k);
    }
  }
  const int numVolumeEqns = volumeEqns.size();

  // The global solution dof manager
  const auto sol_dof_mgr = getDOFManager();
  const int num_elems = sol_dof_mgr->cell_indexer()->getNumLocalElements();

  // Handle the simple case, and return immediately
  if (numVolumeEqns==m_neq) {
    // This is the easy case: couple everything with everything
    for (int icell=0; icell<num_elems; ++icell) {
      const auto& elem_gids = sol_dof_mgr->getElementGIDs(icell);
      m_jac_factory->insertGlobalIndices(elem_gids,elem_gids,true);
    }
    m_jac_factory->fillComplete();
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
      "Error! SideSet equation support not yet added for Omega_h discretization.\n");
}

void OmegahDiscretization::
setFieldData(const Teuchos::RCP<StateInfoStruct>& sis)
{
  printf ("TODO: add code to save states in disc field container, if needed.\n");
}

Teuchos::RCP<Thyra_MultiVector>
OmegahDiscretization::
getSolutionMV (bool overlapped) const
{
  auto soln = Thyra::createMembers(getVectorSpace(), m_num_time_deriv + 1);
  auto fc = overlapped ? m_ov_field_container : m_field_container;
  for (int i=0; i<m_num_time_deriv; ++i) {
    soln->col(i)->assign(*fc->get_field(m_sol_names[i]));
  }
  return soln;
}

Teuchos::RCP<DOFManager>
OmegahDiscretization::
create_dof_mgr (const std::string& field_name,
                const std::string& part_name,
                const FE_Type fe_type,
                const int order,
                const int dof_dim) const
{
  const auto& mesh_specs = m_mesh_struct->getMeshSpecs()[0];

  // Create conn and dof managers
  auto conn_mgr = Teuchos::rcp(new OmegahConnManager(m_mesh_struct->getOmegahMesh()));
  auto dof_mgr  = Teuchos::rcp(new DOFManager(conn_mgr,m_comm,part_name));

  shards::CellTopology topo (&mesh_specs->ctd);
  Teuchos::RCP<panzer::FieldPattern> fp;
  if (topo.getName()==std::string("Particle")) {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
        "Error! Particle cell topology not supported in Omegah discretization yet.\n");
  } else {
    // For space-dependent equations, we rely on Intrepid2 for patterns
    const auto basis = getIntrepid2Basis(*topo.getBaseCellTopologyData(),fe_type,order);
    fp = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  }
  // NOTE: we add $dof_dim copies of the field pattern to the dof mgr,
  //       and call the fields ${field_name}_n, n=0,..,$dof_dim-1
  for (int i=0; i<dof_dim; ++i) {
    dof_mgr->addField(field_name + "_" + std::to_string(i),fp);
  }

  dof_mgr->build();

  return dof_mgr;
}

}  // namespace Albany
