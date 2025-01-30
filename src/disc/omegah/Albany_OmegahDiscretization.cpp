#include "Albany_OmegahDiscretization.hpp"
#include "Albany_OmegahUtils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_ThyraUtils.hpp"

#include "OmegahConnManager.hpp"
#include "Omega_h_adapt.hpp"
#include "Omega_h_array_ops.hpp"

#include <Panzer_IntrepidFieldPattern.hpp>

namespace debug {
template <typename T>
void printTagInfo(Omega_h::Mesh mesh, std::ostringstream& oss, int dim, int tag, std::string type) {
    auto tagbase = mesh.get_tag(dim, tag);
    auto array = Omega_h::as<T>(tagbase)->array();

    Omega_h::Real min = get_min(array);
    Omega_h::Real max = get_max(array);

    oss << std::setw(18) << std::left << tagbase->name().c_str()
        << std::setw(5) << std::left << dim 
        << std::setw(7) << std::left << type 
        << std::setw(5) << std::left << tagbase->ncomps() 
        << std::setw(10) << std::left << min 
        << std::setw(10) << std::left << max 
        << "\n";
}

void printAllTags(Omega_h::Mesh& mesh) {
  std::ostringstream oss;
  // always print two places to the right of the decimal
  // for floating point types (i.e., imbalance)
  oss.precision(2);
  oss << std::fixed;

  oss << "\nTag Properties by Dimension: (Name, Dim, Type, Number of Components, Min. Value, Max. Value)\n";
  for (int dim=0; dim <= mesh.dim(); dim++) {
    for (int tag=0; tag < mesh.ntags(dim); tag++) {
      auto tagbase = mesh.get_tag(dim, tag);
      if (tagbase->type() == OMEGA_H_I8)
        printTagInfo<Omega_h::I8>(mesh, oss, dim, tag, "I8");
      if (tagbase->type() == OMEGA_H_I32)
        printTagInfo<Omega_h::I32>(mesh, oss, dim, tag, "I32");
      if (tagbase->type() == OMEGA_H_I64)
        printTagInfo<Omega_h::I64>(mesh, oss, dim, tag, "I64");
      if (tagbase->type() == OMEGA_H_F64)
        printTagInfo<Omega_h::Real>(mesh, oss, dim, tag, "F64");
    }
  }

  std::cout << oss.str();
}
}

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
  int ws_size = m_mesh_struct->meshSpecs[0]->worksetSize;
  int num_ws = 1 + (nelems-1) / ws_size;

  m_workset_sizes.resize(num_ws);
  int min_ws_size = nelems / num_ws;
  int remainder = nelems % num_ws;
  for (int ws=0;ws<num_ws; ++ws) {
    m_workset_sizes[ws] = min_ws_size + (ws<remainder ? 1 : 0);
  }

  m_workset_elements = DualView<int**>("ws_elems",num_ws,ws_size);
  for (int iws=0,ielem=0; iws<num_ws; ++iws) {
    for (int i=0; i<m_workset_sizes[iws]; ++i,++ielem) {
      m_workset_elements.host()(0,i) = ielem;
    }
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
  m_nodes_coordinates.resize(mdim * getLocalSubdim(getOverlapNodeVectorSpace()));
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

  computeNodeSets (); //FIXME fails here
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
  fprintf(stderr, "OmegahDiscretization::setFieldData\n");
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
    accessor->fillVector(*col,names[icol],dof_mgr,false); //fails here
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

void OmegahDiscretization::
writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                             const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                             const bool          overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION (solution_dxdp != Teuchos::null, std::runtime_error,
      "OmegahDiscretization::writeSolutionToMeshDatabase does not support writing sensitivities yet.");

  const auto& dof_mgr = getDOFManager();
  auto field_accessor = m_mesh_struct->get_field_accessor();
  field_accessor->saveVector(solution, solution_dof_name(), dof_mgr, overlapped);
}

void OmegahDiscretization::
writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                             const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                             const Thyra_Vector& solution_dot,
                             const bool          overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION (solution_dxdp != Teuchos::null, std::runtime_error,
      "OmegahDiscretization::writeSolutionToMeshDatabase does not support writing sensitivities yet.");

  const auto& dof_mgr = getDOFManager();
  auto field_accessor = m_mesh_struct->get_field_accessor();
  field_accessor->saveVector(solution,     solution_dof_name(),          dof_mgr, overlapped);
  field_accessor->saveVector(solution_dot, solution_dof_name() + "_dot", dof_mgr, overlapped);
}

void OmegahDiscretization::
writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                             const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                             const Thyra_Vector& solution_dot,
                             const Thyra_Vector& solution_dotdot,
                             const bool          overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION (solution_dxdp != Teuchos::null, std::runtime_error,
      "OmegahDiscretization::writeSolutionToMeshDatabase does not support writing sensitivities yet.");

  const auto& dof_mgr = getDOFManager();
  auto field_accessor = m_mesh_struct->get_field_accessor();
  field_accessor->saveVector(solution,        solution_dof_name(),             dof_mgr, overlapped);
  field_accessor->saveVector(solution_dot,    solution_dof_name() + "_dot",    dof_mgr, overlapped);
  field_accessor->saveVector(solution_dotdot, solution_dof_name() + "_dotdot", dof_mgr, overlapped);
}

void OmegahDiscretization::
writeSolutionMVToMeshDatabase (const Thyra_MultiVector& solution,
                               const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                               const bool               overlapped)
{
  switch (m_num_time_deriv) {
    case 0:
      writeSolutionToMeshDatabase(*solution.col(0),solution_dxdp,overlapped);
      break;
    case 1:
      writeSolutionToMeshDatabase(*solution.col(0),solution_dxdp,*solution.col(1),overlapped);
      break;
    case 2:
      writeSolutionToMeshDatabase(*solution.col(0),solution_dxdp,*solution.col(1),*solution.col(2),overlapped);
      break;
    default:
      throw std::runtime_error("Unexpected value for m_num_time_deriv:" + std::to_string(m_num_time_deriv) + "\n");
  }
}

//! Write the solution to file. Must call writeSolution first.
void OmegahDiscretization::
writeMeshDatabaseToFile (const double /* time */,
                         const bool   force_write_solution)
{
  std::cout << "WARNING OmegahDiscretization::writeMeshDatabaseToFile not yet implemented\n";
  // throw NotYetImplemented("OmegahDiscretization::writeMeshDatabaseToFile");
}

Teuchos::RCP<AdaptationData>
OmegahDiscretization::
checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& solution ,
                    const Teuchos::RCP<const Thyra_Vector>& solution_dot,
                    const Teuchos::RCP<const Thyra_Vector>& solution_dotdot,
                    const Teuchos::RCP<const Thyra_MultiVector>& dxdp) const
{
  auto adapt_data = Teuchos::rcp(new AdaptationData());

  // Only do adaptation for simple 1d problems
  auto mesh = m_mesh_struct->getOmegahMesh();
  if (mesh->dim() != 1) {
    std::cout << "NOT a 1D Omega_h mesh...\n";
    return adapt_data;
  }
  auto& adapt_params = m_disc_params->sublist("Mesh Adaptivity");
  auto adapt_type = adapt_params.get<std::string>("Type","None");
  if (adapt_type=="None") {
    return adapt_data;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (adapt_type!="Minimally-Oscillatory", std::runtime_error,
      "Error! Adaptation type '" << adapt_type << "' not supported.\n"
      " - valid choices: None, Minimally-Oscillatory\n");

  double tol = adapt_params.get<double>("Max Hessian");
  auto data = getLocalData(solution);
  // Simple check: refine if a proxy of the hessian of x is larger than a tolerance
  // TODO: replace with
  //  1. if |C_i| > threshold, mark for refinement the whole mesh
  //  2. Interpolate solution (and all elem/node fields if possible, but not necessary for adv-diff example)
  int num_nodes = data.size();
  adapt_data->x = solution;
  adapt_data->x_dot = solution_dot;
  adapt_data->x_dotdot = solution_dotdot;
  adapt_data->dxdp = dxdp;
  for (int i=1; i<num_nodes-1; ++i) {
    auto h_prev = m_nodes_coordinates[i] - m_nodes_coordinates[i-1];
    auto h_next = m_nodes_coordinates[i+1] - m_nodes_coordinates[i];
    auto hess = (data[i-1] - 2*data[i] + data[i+1]) / (h_prev*h_next);
    auto grad_prev = (data[i]-data[i-1]) / h_prev;
    auto grad_next = (data[i+1]-data[i]) / h_next;
    if (std::fabs(hess)>tol and grad_prev*grad_next<0) {
      adapt_data->type = AdaptationType::Topology;
      break;
    }
  }

  return adapt_data;

}

void OmegahDiscretization::
adapt (const Teuchos::RCP<AdaptationData>& adaptData)
{
  fprintf(stderr,"OmegahDiscretization::adapt\n");
  // Not sure if we allow calling adapt in general, but just in case
  if (adaptData->type==AdaptationType::None) {
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (adaptData->type!=AdaptationType::Topology, std::runtime_error,
      "Error! Adaptation type not supported. Only 'None' and 'Topology' are currently supported.\n");


  auto ohMesh = m_mesh_struct->getOmegahMesh();
  if (ohMesh->has_tag(0, solution_dof_name())) {
    fprintf(stderr,"OmegahDiscretization::adapt has tag %s\n", solution_dof_name());
  } else {
    fprintf(stderr,"OmegahDiscretization::adapt does NOT have tag %s\n", solution_dof_name());
  }
  //debug::printAllTags(*ohMesh);
  auto nelems = ohMesh->nglobal_ents(ohMesh->dim());
  const auto desired_nelems = nelems*2;

  Omega_h::AdaptOpts opts(&(*ohMesh));
  opts.xfer_opts.type_map[solution_dof_name()] = OMEGA_H_LINEAR_INTERP;
  while (double(nelems) < desired_nelems) {
    std::cout << "element count " << nelems << " < target "
      << desired_nelems << ", will adapt\n";
    if (!ohMesh->has_tag(0, "metric")) {
      std::cout << "mesh had no metric, adding implied and adapting to it\n";
      Omega_h::add_implied_metric_tag(ohMesh.get());
      Omega_h::adapt(ohMesh.get(), opts);
      nelems = ohMesh->nglobal_ents(ohMesh->dim());
      std::cout << "mesh now has " << nelems << " total elements\n";
    }
    auto metrics = ohMesh->get_array<double>(0, "metric");
    metrics = Omega_h::multiply_each_by(metrics, 1.2);
    auto const metric_ncomps =
      Omega_h::divide_no_remainder(metrics.size(), ohMesh->nverts());
    ohMesh->add_tag(0, "metric", metric_ncomps, metrics);
    std::cout << "adapting to scaled metric\n";
    Omega_h::adapt(ohMesh.get(), opts);
    nelems = ohMesh->nglobal_ents(ohMesh->dim());
    std::cout << "mesh now has " << nelems << " total elements\n";
  }

  if (ohMesh->has_tag(0, solution_dof_name())) {
    fprintf(stderr,"OmegahDiscretization::adapt post adapt has tag %s\n", solution_dof_name());
  } else {
    fprintf(stderr,"OmegahDiscretization::adapt post adapt does NOT have tag %s\n", solution_dof_name());
  }

  updateMesh();
  return;
}

void OmegahDiscretization::
writeSolution(
    const Thyra_Vector& solution,
    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
    const double        time,
    const bool          overlapped,
    const bool          force_write_solution) {
  writeSolutionToMeshDatabase(solution, solution_dxdp, time, overlapped);
}

void OmegahDiscretization::
writeSolution(
    const Thyra_Vector& solution,
    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
    const Thyra_Vector& solution_dot,
    const double        time,
    const bool          overlapped,
    const bool          force_write_solution) {
  writeSolutionToMeshDatabase(solution, solution_dxdp, solution_dot, time, overlapped);
}

void OmegahDiscretization::
writeSolution(
      const Thyra_Vector& /* solution */,
      const Teuchos::RCP<const Thyra_MultiVector>& /* solution_dxdp */,
      const Thyra_Vector& /* solution_dot */,
      const Thyra_Vector& /* solution_dotdot */,
      const double        /* time */,
      const bool          /* overlapped */,
      const bool          /* force_write_solution */) {
  TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::writeSolution");
}

void OmegahDiscretization::
writeSolutionMV( //MV = multivector
    const Thyra_MultiVector& /* solution */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* solution_dxdp */,
    const double             /* time */,
    const bool               /* overlapped */,
    const bool               /* force_write_solution */) {
  TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::writeSolutionMV");
}

//! Write the solution to the mesh database.
void OmegahDiscretization::
writeSolutionToMeshDatabase(
    const Thyra_Vector& solution,
    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
    const double        time,
    const bool          overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION(solution_dxdp != Teuchos::null, std::runtime_error, "OmegahDiscretization::writeSolutionToMeshDatabase does not support Thyra_MultiVector");
  const auto& dof_mgr = getDOFManager();
  auto field_accessor = m_mesh_struct->get_field_accessor();
  field_accessor->saveVector(solution, "solution", dof_mgr, overlapped);
}

void OmegahDiscretization::
writeSolutionToMeshDatabase(
    const Thyra_Vector& solution,
    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
    const Thyra_Vector& solution_dot,
    const double        time,
    const bool          overlapped) {
  TEUCHOS_TEST_FOR_EXCEPTION(solution_dxdp != Teuchos::null, std::runtime_error, "OmegahDiscretization::writeSolutionToMeshDatabase does not support Thyra_MultiVector");
  const auto& dof_mgr = getDOFManager();
  auto field_accessor = m_mesh_struct->get_field_accessor();
  field_accessor->saveVector(solution, "solution", dof_mgr, overlapped);
  field_accessor->saveVector(solution_dot, "solution_dot", dof_mgr, overlapped);
}

void OmegahDiscretization::
writeSolutionToMeshDatabase(
    const Thyra_Vector& /* solution */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* solution_dxdp */,
    const Thyra_Vector& /* solution_dot */,
    const Thyra_Vector& /* solution_dotdot */,
    const double        /* time */,
    const bool          /* overlapped */) {
  TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::writeSolutionToMeshDatabase");
}

void OmegahDiscretization::
writeSolutionMVToMeshDatabase(
    const Thyra_MultiVector& /* solution */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* solution_dxdp */,
    const double             /* time */,
    const bool               /* overlapped */) {
  TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::writeSolutionMVToMeshDatabase");
}

//! Write the solution to file. Must call writeSolution first.
void OmegahDiscretization::
writeSolutionToFile(
    const Thyra_Vector& /* solution */,
    const double        /* time */,
    const bool          /* overlapped */,
    const bool          /* force_write_solution */) {
  TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::writeSolutionToFile");
}

void OmegahDiscretization::
writeSolutionMVToFile(
    const Thyra_MultiVector& /* solution */,
    const double             /* time */,
    const bool               /* overlapped */,
    const bool               /* force_write_solution */) {
  TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::writeSolutionMVToFile");
}

}  // namespace Albany
