#include "CTM_Assembler.hpp"
#include "CTM_SolutionInfo.hpp"

#include <AAdapt_InitialCondition.hpp>
#include <Albany_APFDiscretization.hpp>
#include <Albany_Utils.hpp>

namespace CTM {

using Teuchos::rcp;
using Teuchos::rcpFromRef;
using Teuchos::rcp_dynamic_cast;

Assembler::Assembler(
    RCP<ParameterList> p,
    RCP<SolutionInfo> s_info,
    RCP<Albany::AbstractProblem> prob,
    RCP<Albany::AbstractDiscretization> d,
    RCP<Albany::StateManager> sm) {
  params = p;
  sol_info = s_info;
  problem = prob;
  disc = d;
  state_mgr = sm;
  neq = disc->getNumEq();
  initial_setup();
  set_initial_conditions();
}

void Assembler::initial_setup() {
  using RSD = PHAL::AlbanyTraits::Residual;
  fm = problem->getFieldManager();
  dfm = problem->getDirichletFieldManager();
  nfm = problem->getNeumannFieldManager();
  mesh_specs = disc->getMeshStruct()->getMeshSpecs();
  sfm.resize(mesh_specs.size());
  auto dummy = rcp(new PHX::MDALayout<Dummy>(0));
  for (int ps = 0; ps < mesh_specs.size(); ++ps) {
    auto eb_name = mesh_specs[ps]->ebName;
    auto response_ids = state_mgr->getResidResponseIDsToRequire(eb_name);
    sfm[ps] = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    auto tags = problem->buildEvaluators(
        *sfm[ps], *mesh_specs[ps], *state_mgr,
        Albany::BUILD_STATE_FM, Teuchos::null);
    for (auto it = response_ids.begin(); it != response_ids.end(); ++it) {
      auto id = *it;
      PHX::Tag<RSD::ScalarT> res_response_tag(id, dummy);
      sfm[ps]->requireField<RSD>(res_response_tag);
    }
  }
  post_reg_setup();
  state_post_reg_setup();
}

void Assembler::set_initial_conditions() {
  auto wsElNodeEqID = disc->getWsElNodeEqID();
  auto coords = disc->getCoords();
  auto wsEBNames = disc->getWsEBNames();
  int num_dims = disc->getNumDim();
  auto x = sol_info->ghost->x;
  auto icp = params->sublist("Initial Condition");
  auto restart = disc->hasRestartSolution();
  AAdapt::InitialConditionsT(
      x, wsElNodeEqID, wsEBNames, coords, neq, num_dims, icp, restart);
  sol_info->gather_x();
  auto apf_disc = rcp_dynamic_cast<Albany::APFDiscretization>(disc);
  apf_disc->writeSolutionToMeshDatabaseT(*x, 0, true);
}

void Assembler::load_ws_bucket(PHAL::Workset& workset, const int ws) {

  // get discretization data
  auto wsElNodeEqID = disc->getWsElNodeEqID();
  auto wsElNodeID = disc->getWsElNodeID();
  auto coords = disc->getCoords();
  auto wsEBNames = disc->getWsEBNames();

  // populate workset info
  workset.numCells = wsElNodeEqID[ws].size();
  workset.wsElNodeEqID = wsElNodeEqID[ws];
  workset.wsElNodeID = wsElNodeID[ws];
  workset.wsCoords = coords[ws];
  workset.EBName = wsEBNames[ws];
  workset.wsIndex = ws;

  workset.local_Vp.resize(workset.numCells);
  workset.sideSets = rcpFromRef(disc->getSideSets(ws));
  workset.stateArrayPtr =
    &(state_mgr->getStateArray(Albany::StateManager::ELEM, ws));

  // kokkos views
  Kokkos::View<int***, PHX::Device> wsElNodeEqID_kokkos(
      "wsElNodeEqID_kokkos",
      workset.numCells, wsElNodeEqID[ws][0].size(),
      wsElNodeEqID[ws][0][0].size());
  workset.wsElNodeEqID_kokkos = wsElNodeEqID_kokkos;
  for (int i = 0; i < workset.numCells; ++i)
  for (int j = 0; j < wsElNodeEqID[ws][0].size(); ++j)
  for (int k = 0; k < wsElNodeEqID[ws][0][0].size(); ++k)
    workset.wsElNodeEqID_kokkos(i, j, k) = workset.wsElNodeEqID[i][j][k];
}

void Assembler::load_ws_basic(
    PHAL::Workset& workset, const double t_new, const double t_old) {
  workset.numEqs = neq;
  workset.xT = sol_info->ghost->x;
  workset.xdotT = sol_info->ghost->x_dot;
  workset.xdotdotT = Teuchos::null;
  workset.current_time = t_new;
  workset.previous_time = t_old;
  workset.distParamLib = Teuchos::null;
  workset.disc = disc;
  workset.transientTerms = Teuchos::nonnull(workset.xdotT);
  workset.accelerationTerms = false;
}

void Assembler::load_ws_jacobian(
    PHAL::Workset& workset, const double alpha, const double beta,
    const double omega) {
  workset.m_coeff = alpha;
  workset.n_coeff = omega;
  workset.j_coeff = beta;
  workset.ignore_residual = false;
  workset.is_adjoint = false;
}

void Assembler::load_ws_nodeset(PHAL::Workset& workset) {
  workset.nodeSets = rcpFromRef(disc->getNodeSets());
  workset.nodeSetCoords = rcpFromRef(disc->getNodeSetCoords());
}

void Assembler::post_reg_setup() {
  using RSD = PHAL::AlbanyTraits::Residual;
  using JAC = PHAL::AlbanyTraits::Jacobian;
  for (int ps = 0; ps < fm.size(); ++ps) {
    std::vector<PHX::index_size_type> dd;
    int nnodes = mesh_specs[ps].get()->ctd.node_count;
    int deriv_dims = neq * nnodes;
    dd.push_back(deriv_dims);
    fm[ps]->setKokkosExtendedDataTypeDimensions<JAC>(dd);
    fm[ps]->postRegistrationSetupForType<JAC>("Jacobian");
    fm[ps]->postRegistrationSetupForType<RSD>("Residual");
    if (nfm != Teuchos::null && ps < nfm.size()) {
      // nfm[ps]->setKokkosExtendedDataTypeDimensions<JAC>(dd);
      // nfm[ps]->postRegistrationSetupForType<JAC>("Jacobian");
      ALBANY_ALWAYS_ASSERT_VERBOSE(false,
          "CTM does not support neumann boundary conditions");
    }
  }
  if (dfm != Teuchos::null) {
    std::vector<PHX::index_size_type> dd;
    int nnodes = mesh_specs[0].get()->ctd.node_count;
    int deriv_dims = neq * nnodes;
    dd.push_back(deriv_dims);
    dfm->setKokkosExtendedDataTypeDimensions<JAC>(dd);
    dfm->postRegistrationSetupForType<JAC>("Jacobian");
    dfm->postRegistrationSetupForType<RSD>("Residual");
  }
}

void Assembler::state_post_reg_setup() {
  using RSD = PHAL::AlbanyTraits::Residual;
  using JAC = PHAL::AlbanyTraits::Jacobian;
  for (int ps = 0; ps < sfm.size(); ++ps) {
    std::vector<PHX::index_size_type> dd;
    int nnodes = mesh_specs[ps].get()->ctd.node_count;
    int deriv_dims = neq * nnodes;
    dd.push_back(deriv_dims);
    sfm[ps]->setKokkosExtendedDataTypeDimensions<JAC>(dd);
    sfm[ps]->postRegistrationSetup("");
  }
}

void Assembler::assemble_system(
    const double alpha,
    const double beta,
    const double omega,
    const double t_new,
    const double t_old) {

  using JAC = PHAL::AlbanyTraits::Jacobian;

  // get owned algebra containers
  auto owned_x = sol_info->owned->x;
  auto owned_x_dot = sol_info->owned->x_dot;
  auto owned_f = sol_info->owned->f;
  auto owned_J = sol_info->owned->J;

  // get ghost algebra containers
  auto ghost_x = sol_info->ghost->x;
  auto ghost_x_dot = sol_info->ghost->x_dot;
  auto ghost_f = sol_info->ghost->f;
  auto ghost_J = sol_info->ghost->J;

  // make sure we've got the good stuff
  ALBANY_ALWAYS_ASSERT(owned_x != Teuchos::null);
  ALBANY_ALWAYS_ASSERT(owned_f != Teuchos::null);
  ALBANY_ALWAYS_ASSERT(owned_J != Teuchos::null);
  ALBANY_ALWAYS_ASSERT(ghost_x != Teuchos::null);
  ALBANY_ALWAYS_ASSERT(ghost_f != Teuchos::null);
  ALBANY_ALWAYS_ASSERT(ghost_J != Teuchos::null);

  // scatter x and x dot to the ghost distribution
  sol_info->scatter_x();
  sol_info->scatter_x_dot();

  // zero out the residual + jacobian
  owned_f->putScalar(0.0);
  ghost_f->putScalar(0.0);
  owned_J->resumeFill();
  ghost_J->resumeFill();
  owned_J->setAllToScalar(0.0);
  ghost_J->setAllToScalar(0.0);

  // index information
  int num_worksets = disc->getWsElNodeEqID().size();
  auto ws_phys_idx = disc->getWsPhysIndex();

  { // set data in the workset struct + assemble
    PHAL::Workset workset;
    load_ws_basic(workset, t_new, t_old);
    workset.fT = ghost_f;
    workset.JacT = ghost_J;
    load_ws_jacobian(workset, alpha, beta, omega);
    for (int ps = 0; ps < fm.size(); ++ps) {
      int nnodes = mesh_specs[ps].get()->ctd.node_count;
      int deriv_dims = neq * nnodes;
      (workset.Jacobian_deriv_dims).push_back(deriv_dims);
    }
    for (int ws = 0; ws < num_worksets; ++ws) {
      load_ws_bucket(workset, ws);
      fm[ws_phys_idx[ws]]->evaluateFields<JAC>(workset);
      // neumann field eval here if we supported it
    }
  }

  sol_info->gather_f();
  sol_info->gather_J();

  // evaluate the dirichlet fields if needed
  if (Teuchos::nonnull(dfm)) {
    PHAL::Workset workset;
    workset.xT = owned_x;
    workset.transientTerms = Teuchos::nonnull(owned_x_dot);
    workset.accelerationTerms = false;
    workset.fT = owned_f;
    workset.JacT = owned_J;
    workset.m_coeff = alpha;
    workset.n_coeff = omega;
    workset.j_coeff = beta;
    workset.current_time = t_new;
    workset.previous_time = t_old;
    workset.distParamLib = Teuchos::null;
    workset.disc = disc;
    workset.distParamLib = Teuchos::null;
    load_ws_nodeset(workset);
    dfm->evaluateFields<JAC>(workset);
  }

  owned_J->fillComplete();
  ghost_J->fillComplete();

}

void Assembler::assemble_state(
    const double t_new,
    const double t_old) {

  using RSD = PHAL::AlbanyTraits::Residual;

  // index information
  int num_worksets = disc->getWsElNodeEqID().size();
  auto ws_phys_idx = disc->getWsPhysIndex();

  // scatter x and x dot to ghost distribution
  sol_info->scatter_x();
  sol_info->scatter_x_dot();

  // fill in standard workset stuff
  PHAL::Workset workset;
  load_ws_basic(workset, t_new, t_old);
  workset.fT = sol_info->ghost->f;

  for (int ws = 0; ws < num_worksets; ++ws) {
    load_ws_bucket(workset, ws);
    sfm[ws_phys_idx[ws]]->evaluateFields<RSD>(workset);
  }
}

} // namespace CTM
