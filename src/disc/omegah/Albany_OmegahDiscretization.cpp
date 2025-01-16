#include "Albany_OmegahDiscretization.hpp"
#include "Albany_OmegahUtils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_ThyraUtils.hpp"

#include "OmegahConnManager.hpp"

#include <Panzer_IntrepidFieldPattern.hpp>

namespace Albany {

OmegahDiscretization::
OmegahDiscretization (const Teuchos::RCP<Teuchos::ParameterList>& discParams,
                      const int                                   neq,
                      const Teuchos::RCP<OmegahGenericMesh>&      mesh,
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

  // Create DOF managers
  auto sol_dof_mgr  = create_dof_mgr(solution_dof_name(),"",FE_Type::HGRAD,1,m_neq);
  auto node_dof_mgr = create_dof_mgr(nodes_dof_name(),"",FE_Type::HGRAD,1,1);

  m_dof_managers[solution_dof_name()][""] = sol_dof_mgr;
  m_dof_managers[nodes_dof_name()][""]     = node_dof_mgr;
  m_node_dof_managers[""]     = node_dof_mgr;

  // Compute workset information
  // NOTE: these arrays are all of size 1, for the foreseable future.
  //       Still, make impl generic (where possible), in case things change.
  const auto& ms = m_mesh_struct->meshSpecs[0];
  const auto& mesh = *m_mesh_struct->getOmegahMesh();
  int nelems = mesh.nelems();
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
  auto node_gids = hostRead(mesh.globals(0));
  auto node_indexer = getOverlapNodeGlobalLocalIndexer();
  m_node_lid_to_omegah_pos.resize(mesh.nverts());
  for (int i=0; i<mesh.nverts(); ++i) {
    auto gid = node_gids[i];
    auto lid = node_indexer->getLocalElement(gid);
    m_node_lid_to_omegah_pos[lid] = i;
  }
  int num_elem_nodes = node_dof_mgr->get_topology().getNodeCount();
  const auto& node_elem_dof_lids = node_dof_mgr->elem_dof_lids().host();

  const int mdim = mesh.dim();
  m_nodes_coordinates.resize(3 * getLocalSubdim(getOverlapNodeVectorSpace()));
  for (int ws=0; ws<num_ws; ++ws) {
    m_ws_elem_coords[ws].resize(m_workset_sizes[ws]);
    for (int ielem=0; ielem<m_workset_sizes[ws]; ++ielem) {
      m_ws_elem_coords[ws][ielem].resize(num_elem_nodes);
      for (int inode=0; inode<num_elem_nodes; ++inode) {
        LO node_lid = node_elem_dof_lids(ielem,inode);
        int omh_pos = m_node_lid_to_omegah_pos[node_lid];
        m_ws_elem_coords[ws][ielem][inode] = &coords_h[omh_pos*mdim];
        auto coords = &m_nodes_coordinates[node_lid*mdim];
        for (int idim=0; idim<mdim; ++idim) {
          coords[idim] = m_ws_elem_coords[ws][ielem][inode][idim];
        }
      }
    }
  }

  m_side_sets.resize(num_ws);
  for (int ws=0; ws<num_ws; ++ws) {
    m_side_set_views[ws] = {};
    m_ws_local_dof_views[ws] = {};
  }

  computeNodeSets ();
  computeGraphs ();
}

void OmegahDiscretization::
computeNodeSets ()
{
  const auto& nsNames = getMeshStruct()->meshSpecs[0]->nsNames;
  using Omega_h::I32;
  using Omega_h::I8;

  auto& mesh = *m_mesh_struct->getOmegahMesh();

  auto v2e = mesh.ask_up(0,mesh.dim());
  auto v2e_a2ab = hostRead(v2e.a2ab);
  auto v2e_ab2b = hostRead(v2e.ab2b);

  auto e2v = hostRead(mesh.ask_elem_verts());
  int nodes_per_elem = e2v.size() / mesh.nelems();

  auto owned_host = hostRead(mesh.owned(0));
  for (const auto& nsn : nsNames) {
    auto is_on_ns_host = hostRead(mesh.get_array<Omega_h::I8>(0,nsn));
    std::vector<int> owned_on_ns;
    for (int i=0; i<is_on_ns_host.size(); ++i) {
      if (owned_host[i] and is_on_ns_host[i]) {
        owned_on_ns.push_back(i);
      }
    }

    auto& ns_elem_pos = m_node_sets[nsn];

    ns_elem_pos.reserve(owned_on_ns.size());
    for (auto i : owned_on_ns) {
      auto node_adj_start = v2e_a2ab[i];
      auto ielem = v2e_ab2b[node_adj_start];

      bool found = false;
      for (int j=0; j<nodes_per_elem; ++j) {
        if (e2v[nodes_per_elem*ielem+j]==i) {
          ns_elem_pos.push_back(std::make_pair(ielem,j));
          found = true;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION (not found, std::runtime_error,
          "Something went wrong while locating a node in an element.\n"
          " - node set: " << nsn << "\n"
          " - node lid (osh): " << i << "\n"
          " - ielem: " << ielem << "\n");
    }
  }
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
setFieldData(const Teuchos::RCP<StateInfoStruct>& /* sis */)
{
  auto field_accessor = Teuchos::rcp_dynamic_cast<OmegahMeshFieldAccessor>(m_mesh_struct->get_field_accessor());
  field_accessor->addFieldOnMesh (solution_dof_name(),FE_Type::HGRAD,m_neq);
  auto mesh_fields = m_mesh_struct->get_field_accessor();
  for (auto st : mesh_fields->getNodalParameterSIS()) {
    // TODO: get mesh part from st, create dof mgr on that part for st.name dof
    int numComps;
    switch (st->dim.size()) {
      case 2: numComps = 1; break;
      case 3: numComps = st->dim[2]; break;
      default:
        throw std::runtime_error(
            "[OmegahDiscretization::setFieldData] Error! Unsupported nodal state rank.\n"
            "  - state name: " + st->name + "\n"
            "  - input dims: (" + util::join(st->dim,",") + ")\n");
    }
    auto dof_mgr = create_dof_mgr (st->name,st->meshPart,FE_Type::HGRAD,1,numComps);
    m_dof_managers[st->name][st->meshPart] = dof_mgr;

    if (m_node_dof_managers.find(st->meshPart)==m_node_dof_managers.end()) {
      auto node_dof_mgr = create_dof_mgr (nodes_dof_name(),st->meshPart,FE_Type::HGRAD,1,1);
      m_node_dof_managers[st->meshPart] = node_dof_mgr;
    }
  }
}

void
OmegahDiscretization::
getSolutionMV (Thyra_MultiVector& solution, bool /* overlapped */) const
{
  std::vector<std::string> names = {
    solution_dof_name(),
    solution_dof_name() + std::string("_dot"),
    solution_dof_name() + std::string("_dotdot")
  };
  auto accessor = m_mesh_struct->get_field_accessor();
  auto dof_mgr = getDOFManager();
  for (int icol=0; icol<m_num_time_deriv; ++icol) {
    auto col = solution.col(icol);
    accessor->fillVector(*col,names[icol],dof_mgr,false);
  }
}

void
OmegahDiscretization::
getField (Thyra_Vector& field_vector, const std::string& field_name) const
{
  auto accessor = m_mesh_struct->get_field_accessor();
  auto dof_mgr = getDOFManager(field_name);
  accessor->fillVector(field_vector,field_name,dof_mgr,false);
}

void
OmegahDiscretization::
setField (const Thyra_Vector& field_vector,
          const std::string&  field_name,
          bool                /* overlapped */)
{
  auto accessor = m_mesh_struct->get_field_accessor();
  auto dof_mgr = getDOFManager(field_name);
  accessor->saveVector(field_vector,field_name,dof_mgr,false);
}

Teuchos::RCP<DOFManager>
OmegahDiscretization::
create_dof_mgr (const std::string& field_name,
                const std::string& part_name,
                const FE_Type fe_type,
                const int order,
                const int dof_dim) const
{
  const auto& mesh_specs = m_mesh_struct->meshSpecs[0];

  // Create conn and dof managers
  auto conn_mgr = Teuchos::rcp(new OmegahConnManager(m_mesh_struct));
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

Teuchos::RCP<AdaptationData>
OmegahDiscretization::
checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& /* solution */,
                    const Teuchos::RCP<const Thyra_Vector>& /* solution_dot */,
                    const Teuchos::RCP<const Thyra_Vector>& /* solution_dotdot */,
                    const Teuchos::RCP<const Thyra_MultiVector>& /* dxdp */) const
{
  throw NotYetImplemented("OmegaDiscretization::checkForAdaptation");
}

void OmegahDiscretization::
adapt (const Teuchos::RCP<AdaptationData>& adaptData)
{
  // Not sure if we allow calling adapt in general, but just in case
  if (adaptData->type==AdaptationType::None) {
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (adaptData->type!=AdaptationType::Topology, std::runtime_error,
      "Error! Adaptation type not supported. Only 'None' and 'Topology' are currently supported.\n");

  return;
//
//  // Solution oscillates. We need to half dx
//  auto mesh1d = Teuchos::rcp_dynamic_cast<TmplSTKMeshStruct<1>>(stkMeshStruct);
//  int num_params = mesh1d->getNumParams();
//  int ne_x = discParams->get<int>("1D Elements");
//  auto& adapt_params = discParams->sublist("Mesh Adaptivity");
//  discParams->set("Workset Size", stkMeshStruct->meshSpecs()[0]->worksetSize);
//  int factor = adapt_params.get("Refining Factor",2);
//  discParams->set("1D Elements",factor*ne_x);
//  stkMeshStruct = Teuchos::rcp(new TmplSTKMeshStruct<1>(discParams,comm,num_params));
//  stkMeshStruct->setFieldData(comm,mesh1d->sis_);
//  this->setFieldData(mesh1d->sis_);
//  stkMeshStruct->setBulkData(comm);
//
//  updateMesh();
//
//  int num_time_deriv = discParams->get<int>("Number Of Time Derivatives");
//  auto x_mv_new = Thyra::createMembers(getVectorSpace(),num_time_deriv);
//
//  for (int ideriv=0; ideriv<num_time_deriv; ++ideriv) {
//    auto data_new = getNonconstLocalData(x_mv_new->col(ideriv));
//    auto x = ideriv==0 ? adaptData->x : (ideriv==1 ? adaptData->x_dot : adaptData->x_dotdot);
//    auto data_old = getLocalData(x);
//    int num_nodes_new = data_new.size();
//
//    for (int inode=0; inode<num_nodes_new; ++inode) {
//      int coarse = inode / factor;
//      int rem    = inode % factor;
//      if (rem == 0) {
//        // Same node as coarse mesh
//        data_new[inode] = data_old[coarse];
//      } else {
//        // Convex interpolation of two coarse points
//        double alpha = static_cast<double>(rem) / factor;
//        data_new[inode] = data_old[coarse]*(1-alpha) + data_old[coarse+1]*alpha;
//      }
//    }
//  }
//
//  writeSolutionMVToMeshDatabase(*x_mv_new, Teuchos::null, 0, false);
}



}  // namespace Albany
