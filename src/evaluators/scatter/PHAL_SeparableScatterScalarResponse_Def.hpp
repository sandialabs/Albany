//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_Utilities.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Albany_Hessian.hpp"

// **********************************************************************
// Base Class Generic Implementation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
SeparableScatterScalarResponseBase<EvalT, Traits>::
SeparableScatterScalarResponseBase(const Teuchos::ParameterList& p,
                                   const Teuchos::RCP<Albany::Layouts>& dl)
{
  setup(p, dl);
}

template<typename EvalT, typename Traits>
void SeparableScatterScalarResponseBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(local_response,fm);
  if (!this->stand_alone) {
    this->utils.setFieldData(local_response_eval,fm);
  }
}

template<typename EvalT, typename Traits>
void
SeparableScatterScalarResponseBase<EvalT, Traits>::
setup(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& /* dl */)
{
  this->stand_alone = p.get<bool>("Stand-alone Evaluator");

  // Setup fields we require
  auto local_response_tag =
    p.get<PHX::Tag<ScalarT> >("Local Response Field Tag");
  local_response = decltype(local_response)(local_response_tag);
  if (this->stand_alone) {
    this->addDependentField(local_response);
  } else {
    local_response_eval = decltype(local_response_eval)(local_response_tag);
    this->addEvaluatedField(local_response_eval);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
SeparableScatterScalarResponse<AlbanyTraits::Jacobian, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::Jacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Initialize derivatives
  Teuchos::RCP<Thyra_MultiVector> dgdx = workset.dgdx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdx = workset.overlapped_dgdx;
  if (dgdx != Teuchos::null) {
    dgdx->assign(0.0);
    overlapped_dgdx->assign(0.0);
  }

  Teuchos::RCP<Thyra_MultiVector> dgdxdot = workset.dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdot = workset.overlapped_dgdxdot;
  if (dgdxdot != Teuchos::null) {
    dgdxdot->assign(0.0);
    overlapped_dgdxdot->assign(0.0);
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* response derivative
  Teuchos::RCP<Thyra_MultiVector> dgdx = workset.overlapped_dgdx;
  Teuchos::RCP<Thyra_MultiVector> dgdxdot = workset.overlapped_dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> dg;
  if (dgdx != Teuchos::null) {
    dg = dgdx;
  } else {
    dg = dgdxdot;
  }

  auto dg_data = Albany::getNonconstLocalData(dg);

  const auto& dof_mgr = workset.disc->getNewDOFManager();
  const int neq = dof_mgr->getNumFields();
  const int  ws = workset.wsIndex;
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto elem_lids     = workset.disc->getElementLIDs_host(ws);
  const auto& part_name = dof_mgr->part_name();

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);

    // Loop over equations per node
    for (int eq_dof=0; eq_dof<neq; eq_dof++) {

      // Get offsets of this dof in the lids array
      auto offsets = dof_mgr->getGIDFieldOffsets(part_name,eq_dof);

      const int num_nodes = offsets.size();

      for (int i=0; i<num_nodes; ++i) {

        const int deriv   = offsets[i];
        const int dof_lid = elem_dof_lids(elem_LID,deriv);

        // Loop over responses
        for (std::size_t res = 0; res < this->global_response.size(); res++) {
          auto val = this->local_response(cell, res);

          // Set dg/dx
          // NOTE: mv local data is in column major
          dg_data[res][dof_lid] += val.dx(deriv);
        } // column equations
      } // column nodes
    } // response
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::Jacobian, Traits>::
evaluate2DFieldsDerivativesDueToExtrudedSolution(
    typename Traits::EvalData workset,
    std::string& sidesetName,
    Teuchos::RCP<const CellTopologyData> cellTopo)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
      "Side sets not properly specified on the mesh.\n");

  // Ensure we have ONE cell per layer.
  const auto topo_base = cellTopo->base;

  const auto hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
  const auto quad  = shards::getCellTopologyData<shards::Quadrilateral<4>>();
  const auto wedge = shards::getCellTopologyData<shards::Wedge<6>>();

  TEUCHOS_TEST_FOR_EXCEPTION (
      topo_base==hexa || topo_base==quad || topo_base==wedge, std::logic_error,
      " [evaluate2DFieldsDerivativesDueToExtrudedSolution]\n"
      "   Feature only available for extruded meshes with one element per layer.\n");

  // Check for early return
  if (workset.sideSets->count(sidesetName)==0) {
    return;
  }

  // Here we scatter the *local* response derivative
  const auto dgdx    = workset.overlapped_dgdx;
  const auto dgdxdot = workset.overlapped_dgdxdot;
  const auto dg = dgdx.is_null() ? dgdxdot : dgdx;

  auto dg_data = Albany::getNonconstLocalData(dg);

  constexpr auto ALL = Kokkos::ALL();

  const auto& disc            = workset.disc;
  const auto& dof_mgr         = disc->getNewDOFManager();
  const auto& sol_name        = disc->solution_dof_name();
  const auto& part_name       = dof_mgr->part_name();
  const auto& node_dof_mgr    = disc->getNodeNewDOFManager(part_name);
  const auto& ss_node_dof_mgr = disc->getNodeNewDOFManager(sidesetName);

  const auto  elem_dof_lids   = dof_mgr->elem_dof_lids().host();
  const auto& cell_indexer    = dof_mgr->cell_indexer();

  const auto& node_layers = disc->getMeshStruct()->layered_mesh_numbering_nodes;
  const auto& cell_layers = disc->getMeshStruct()->layered_mesh_numbering_cells;
  const int   numLayers   = cell_layers->numLayers;

  const int neq = dof_mgr->getNumFields();

  const auto& sideSet = workset.sideSets->find(sidesetName)->second;

  std::vector<GO> elem_node_gids, basal_node_gids;

  for (const auto& side : sideSet) {
    const auto base_elem_GID = side.side_GID;
    const auto& side_topo = cellTopo->side[side.side_pos].topology;
    const int numSideNodes = side_topo->node_count;

    // Get the gids of the basal nodes, for later use
    ss_node_dof_mgr->getElementGIDs(side.side_LID,basal_node_gids);

    // Given a basal node gid, figure out it's relative position in the side
    auto get_basal_inode = [&](const GO gid) -> int {
      auto it = std::find(basal_node_gids.begin(),basal_node_gids.end(),gid);
      TEUCHOS_TEST_FOR_EXCEPTION(it!=basal_node_gids.end(), std::runtime_error,
          "Error! Could not locate basal node gid " << gid << " in basal side.\n");
      return std::distance(basal_node_gids.begin(),it);
    };

    // Loop over all responses
    for (std::size_t res=0; res<this->global_response.size(); ++res) {
      // Loop over layers
      for (int ilayer=0; ilayer<=numLayers; ++ilayer) {
        const auto elem_GID = cell_layers->getId(base_elem_GID,ilayer);
        const auto elem_LID = cell_indexer->getLocalElement(elem_GID);
        auto val = this->local_response(elem_LID, res);

        // Get dof LIDs and node GIDs in this element
        const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
        node_dof_mgr->getElementGIDs(elem_LID,elem_node_gids);

        // Loop over all equations
        for (int eq=0; eq<neq; ++eq) {
          // Offsets for this eq in the dof_lids view
          const auto offsets = dof_mgr->getGIDFieldOffsets(part_name,eq);

          // Loop over all nodes in this cell
          for (int inode=0; inode<this->numNodes; ++inode) {
            // Get dof LID and node GID
            const auto dof_lid = dof_lids(offsets[inode]);
            const auto node_gid = elem_node_gids[inode];

            // Retrieve basal node GID, and its position in the basal side
            const auto base_node_gid = node_layers->getColumnId(node_gid);
            const auto base_inode    = get_basal_inode(base_node_gid);

            // Compute index of derivative.
            const int deriv = neq*this->numNodes +
                              ilayer*neq*numSideNodes +
                              neq*base_inode + eq;

            dg_data[res][dof_lid] += val.dx(deriv);
          }
        }
      }
    }
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* response
  Teuchos::RCP<Thyra_Vector> g = workset.g;
  if (g != Teuchos::null) {
    Teuchos::ArrayRCP<ST> g_nonconstView = Albany::getNonconstLocalData(g);
    for (MDFieldIterator<const ScalarT> gr(this->global_response);
         ! gr.done(); ++gr)
      g_nonconstView[gr.idx()] = gr.ref().val();
  }

  // Here we scatter the *global* response derivatives
  Teuchos::RCP<Thyra_MultiVector> dgdx = workset.dgdx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdx = workset.overlapped_dgdx;
  if (dgdx != Teuchos::null) {
    workset.x_cas_manager->combine(overlapped_dgdx, dgdx, Albany::CombineMode::ADD);
  }

  Teuchos::RCP<Thyra_MultiVector> dgdxdot = workset.dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdot = workset.overlapped_dgdxdot;
  if (dgdxdot != Teuchos::null) {
    workset.x_cas_manager->combine(overlapped_dgdxdot, dgdxdot, Albany::CombineMode::ADD);
  }
}

// **********************************************************************
// Specialization: Distributed Parameter Derivative
// **********************************************************************
template<typename Traits>
SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  Teuchos::RCP<Thyra_MultiVector> dgdp = workset.dgdp;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdp = workset.overlapped_dgdp;
  if (dgdp != Teuchos::null) {
    dgdp->assign(0.0);
  }
  if (overlapped_dgdp != Teuchos::null) {
    overlapped_dgdp->assign(0.0);
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Check for early return
  auto dgdp = workset.overlapped_dgdp;
  if (dgdp.is_null()) {
    return;
  }

  constexpr auto ALL = Kokkos::ALL();
  const auto dgdp_data = Albany::getNonconstLocalData(dgdp);
  const int  ws = workset.wsIndex;
  const int num_deriv = numNodes;

  const auto elem_lids     = workset.disc->getElementLIDs_host(ws);
  const auto param = workset.distParamLib->get(workset.dist_param_deriv_name);
  const auto p_elem_dof_lids = param->elem_dof_lids().host();

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      // Loop over nodes in cell
      for (int deriv=0; deriv<num_deriv; ++deriv) {

        // If param defined at this node, update dg/dp
        const int row = dof_lids(deriv);
        if(row >=0){
          dgdp_data[res][row] += this->local_response(cell, res).dx(deriv);
        }
      } // deriv
    } // response
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  auto g = workset.g;
  if (g != Teuchos::null) {
    auto g_nonconstView = Albany::getNonconstLocalData(g);
    for (std::size_t res=0; res<this->global_response.size(); ++res) {
      g_nonconstView[res] = this->global_response[res].val();
    }
  }

  auto dgdp = workset.dgdp;
  auto overlapped_dgdp = workset.overlapped_dgdp;
  if (!dgdp.is_null() && !overlapped_dgdp.is_null()) {
    workset.p_cas_manager->combine(overlapped_dgdp, dgdp, Albany::CombineMode::ADD);
  }
}

// **********************************************************************
template<typename Traits>
void SeparableScatterScalarResponseWithExtrudedParams<AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& param_name = workset.dist_param_deriv_name;
  const auto level_it = extruded_params_levels->find(param_name);
  if (level_it == extruded_params_levels->end()) {
    //if parameter is not extruded use usual scatter.
    return Base::evaluateFields(workset);
  }

  // Check for early return
  const auto dgdp = workset.overlapped_dgdp;
  if (dgdp.is_null()) {
    return;
  }
  const auto dgdp_data = Albany::getNonconstLocalData(dgdp);

  const int ws = workset.wsIndex;
  const int fieldLevel = level_it->second;

  const auto& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const auto p_dof_mgr    = workset.disc->getNewDOFManager(param_name);
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto ov_p_indexer = p_dof_mgr->ov_indexer();
  const auto elem_lids    = workset.disc->getElementLIDs_host(ws);

  // Loop over cells in workset
  std::vector<GO> node_gids;
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);

    node_dof_mgr->getElementGIDs(elem_LID,node_gids);

    // Loop over responses
    for (size_t res=0; res<this->global_response.size(); ++res) {
      const auto lresp = this->local_response(cell,res);

      // Loop over nodes in cell
      for (int node=0; node<this->numNodes; ++node) {
        const GO base_id = layeredMeshNumbering.getColumnId(node_gids[node]);
        const GO ginode  = layeredMeshNumbering.getId(base_id, fieldLevel);
        const LO row     = ov_p_indexer->getLocalElement(ginode);

        // Set dg/dp
        if(row >=0) {
          dgdp_data[res][row] += lresp.dx(node);
        }
      } // deriv
    } // response
  } // cell
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************
template<typename Traits>
SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  const auto& hws = workset.hessianWorkset;
  const auto hess_vec_prod_g_xx = hws.hess_vec_prod_g_xx;
  const auto hess_vec_prod_g_xp = hws.hess_vec_prod_g_xp;
  const auto hess_vec_prod_g_px = hws.hess_vec_prod_g_px;
  const auto hess_vec_prod_g_pp = hws.hess_vec_prod_g_pp;
  const auto overlapped_hess_vec_prod_g_xx = hws.overlapped_hess_vec_prod_g_xx;
  const auto overlapped_hess_vec_prod_g_xp = hws.overlapped_hess_vec_prod_g_xp;
  const auto overlapped_hess_vec_prod_g_px = hws.overlapped_hess_vec_prod_g_px;
  const auto overlapped_hess_vec_prod_g_pp = hws.overlapped_hess_vec_prod_g_pp;

  if (hess_vec_prod_g_xx != Teuchos::null) {
    hess_vec_prod_g_xx->assign(0.0);
  }
  if (hess_vec_prod_g_xp != Teuchos::null) {
    hess_vec_prod_g_xp->assign(0.0);
  }
  if (hess_vec_prod_g_px != Teuchos::null) {
    hess_vec_prod_g_px->assign(0.0);
  }
  if (hess_vec_prod_g_pp != Teuchos::null) {
    hess_vec_prod_g_pp->assign(0.0);
  }
  if (overlapped_hess_vec_prod_g_xx != Teuchos::null) {
    overlapped_hess_vec_prod_g_xx->assign(0.0);
  }
  if (overlapped_hess_vec_prod_g_xp != Teuchos::null) {
    overlapped_hess_vec_prod_g_xp->assign(0.0);
  }
  if (overlapped_hess_vec_prod_g_px != Teuchos::null) {
    overlapped_hess_vec_prod_g_px->assign(0.0);
  }
  if (overlapped_hess_vec_prod_g_pp != Teuchos::null) {
    overlapped_hess_vec_prod_g_pp->assign(0.0);
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // First, the function checks whether the parameter associated to workset.dist_param_deriv_name
  // is a distributed parameter (distributed==true) or a parameter vector
  // (distributed==false).
  int param_id;
  bool distributed;
  Albany::getParameterVectorID(param_id, distributed, workset.dist_param_deriv_name);

  // Here we scatter the *local* response derivative
  const auto& hws = workset.hessianWorkset;
  const auto hess_vec_prod_g_xx = hws.overlapped_hess_vec_prod_g_xx;
  const auto hess_vec_prod_g_xp = hws.overlapped_hess_vec_prod_g_xp;
  const auto hess_vec_prod_g_px = hws.overlapped_hess_vec_prod_g_px;
  const auto hess_vec_prod_g_pp = hws.overlapped_hess_vec_prod_g_pp;

  // Check for early return
  if (hess_vec_prod_g_xx.is_null() && hess_vec_prod_g_xp.is_null() &&
      hess_vec_prod_g_px.is_null() && hess_vec_prod_g_pp.is_null()) {
    return;
  }

  // Extract multivectors raw data
  using mv_data_t = Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>;
  mv_data_t hess_vec_prod_g_xx_data, hess_vec_prod_g_xp_data,
                 hess_vec_prod_g_px_data, hess_vec_prod_g_pp_data;

  if (!hess_vec_prod_g_xx.is_null())
    hess_vec_prod_g_xx_data = Albany::getNonconstLocalData(hess_vec_prod_g_xx);
  if (!hess_vec_prod_g_xp.is_null())
    hess_vec_prod_g_xp_data = Albany::getNonconstLocalData(hess_vec_prod_g_xp);
  if (!hess_vec_prod_g_px.is_null())
    hess_vec_prod_g_px_data = Albany::getNonconstLocalData(hess_vec_prod_g_px);
  if (!hess_vec_prod_g_pp.is_null())
    hess_vec_prod_g_pp_data = Albany::getNonconstLocalData(hess_vec_prod_g_pp);

  const bool do_xx        = !hess_vec_prod_g_xx.is_null();
  const bool do_xp        = !hess_vec_prod_g_xp.is_null();
  const bool do_scalar_px = !hess_vec_prod_g_px.is_null() && !distributed;
  const bool do_scalar_pp = !hess_vec_prod_g_pp.is_null() && !distributed;
  const bool do_dist_px   = !hess_vec_prod_g_px.is_null() &&  distributed;
  const bool do_dist_pp   = !hess_vec_prod_g_pp.is_null() &&  distributed;

  // Get some data from the discretization
  const auto param_name = workset.dist_param_deriv_name;
  Albany::DualView<int**>::host_t p_elem_dof_lids;
  if (distributed) {
    auto dist_param = workset.distParamLib->get(param_name);
    p_elem_dof_lids = dist_param->elem_dof_lids().host();
  }

  const auto& dof_mgr   = workset.disc->getNewDOFManager();

  const int  ws = workset.wsIndex;
  const int neq = dof_mgr->getNumFields();

  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& elem_lids     = workset.disc->getElementLIDs_host(ws);

  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);

    // Loop over responses
    for (size_t res=0; res<this->global_response.size(); ++res) {
      auto lresp = this->local_response(cell, res);

      if (do_xx || do_xp) {
        // Loop over equations per node
        for (int eq_dof=0; eq_dof<neq; eq_dof++) {

          // Get offsets of this dof in the lids array
          const auto offsets = dof_mgr->getGIDFieldOffsets(eq_dof);

          const int num_nodes = offsets.size();

          for (int i=0; i<num_nodes; ++i) {

            const int deriv   = offsets[i];
            const int dof_lid = elem_dof_lids(elem_LID,deriv);

            if (do_xx)
              hess_vec_prod_g_xx_data[res][dof_lid] += lresp.dx(deriv).dx(0);

            if (do_xp)
              hess_vec_prod_g_xp_data[res][dof_lid] += lresp.dx(deriv).dx(0);
          }
        }
      }

      if (do_dist_px) {
        for (int deriv=0; deriv<numNodes; ++deriv) {
          const int row = p_elem_dof_lids(elem_LID,deriv);
          if (row>=0) {
            hess_vec_prod_g_px_data[res][row] += lresp.dx(deriv).dx(0);
          }
        }
      }
      if (do_dist_pp) {
        for (int deriv=0; deriv<numNodes; ++deriv) {
          const int row = p_elem_dof_lids(elem_LID,deriv);
          if (row>=0) {
            hess_vec_prod_g_pp_data[res][row] += lresp.dx(deriv).dx(0);
          }
        }
      }

      if (do_scalar_px) {
        for (int deriv=0; deriv<hess_vec_prod_g_px_data[res].size(); ++deriv) {
          hess_vec_prod_g_px_data[res][deriv] += lresp.dx(deriv).dx(0);
        }
      }
      if (do_scalar_pp) {
        for (int deriv=0; deriv<hess_vec_prod_g_pp_data[res].size(); ++deriv) {
          hess_vec_prod_g_pp_data[res][deriv] += lresp.dx(deriv).dx(0);
        }
      }
    }
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // First, the function checks whether the parameter associated to workset.dist_param_deriv_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, workset.dist_param_deriv_name);

  const auto g = workset.g;
  if (g != Teuchos::null) {
    const auto g_nonconstView = Albany::getNonconstLocalData(g);
    for (size_t res=0; res<this->global_response.size(); res++) {
      g_nonconstView[res] = this->global_response[res].val().val();
    }
  }

  const auto& hws = workset.hessianWorkset;

  const auto hess_vec_prod_g_xx = hws.hess_vec_prod_g_xx;
  const auto hess_vec_prod_g_xp = hws.hess_vec_prod_g_xp;
  const auto hess_vec_prod_g_px = hws.hess_vec_prod_g_px;
  const auto hess_vec_prod_g_pp = hws.hess_vec_prod_g_pp;

  const auto overlapped_hess_vec_prod_g_xx = hws.overlapped_hess_vec_prod_g_xx;
  const auto overlapped_hess_vec_prod_g_xp = hws.overlapped_hess_vec_prod_g_xp;
  const auto overlapped_hess_vec_prod_g_px = hws.overlapped_hess_vec_prod_g_px;
  const auto overlapped_hess_vec_prod_g_pp = hws.overlapped_hess_vec_prod_g_pp;

  const auto x_cas_mgr = workset.x_cas_manager;
  const auto p_cas_mgr = workset.p_cas_manager;
  constexpr auto ADD    = Albany::CombineMode::ADD;
  constexpr auto INSERT = Albany::CombineMode::INSERT;

  if (!hess_vec_prod_g_xx.is_null() && !overlapped_hess_vec_prod_g_xx.is_null()) {
    x_cas_mgr->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, ADD);
  }
  if (!hess_vec_prod_g_xp.is_null() && !overlapped_hess_vec_prod_g_xp.is_null()) {
    x_cas_mgr->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, ADD);
  }
  if (l1_is_distributed) {
    if (!hess_vec_prod_g_px.is_null() && !overlapped_hess_vec_prod_g_px.is_null()) {
      p_cas_mgr->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, ADD);
    }
    if (!hess_vec_prod_g_pp.is_null() && !overlapped_hess_vec_prod_g_pp.is_null()) {
      p_cas_mgr->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, ADD);
    }
  } else {
    if (!hess_vec_prod_g_px.is_null() && !overlapped_hess_vec_prod_g_px.is_null()) {
      auto tmp = Thyra::createMembers(p_cas_mgr->getOwnedVectorSpace(),overlapped_hess_vec_prod_g_px->domain()->dim());
      p_cas_mgr->combine(overlapped_hess_vec_prod_g_px, tmp, ADD);
      p_cas_mgr->scatter(tmp, hess_vec_prod_g_px, INSERT);
    }
    if (!hess_vec_prod_g_pp.is_null() && !overlapped_hess_vec_prod_g_pp.is_null()) {
      auto tmp = Thyra::createMembers(p_cas_mgr->getOwnedVectorSpace(),overlapped_hess_vec_prod_g_pp->domain()->dim());
      p_cas_mgr->combine(overlapped_hess_vec_prod_g_pp, tmp, ADD);
      p_cas_mgr->scatter(tmp, hess_vec_prod_g_pp, INSERT);
    }
  }
}

// **********************************************************************
template<typename Traits>
void SeparableScatterScalarResponseWithExtrudedParams<AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  if(level_it == extruded_params_levels->end()) {
    //if parameter is not extruded use usual scatter.
    return Base::evaluateFields(workset);
  }

  // Here we scatter the *local* response derivative
  const auto& hws = workset.hessianWorkset;
  const auto hess_vec_prod_g_px = hws.overlapped_hess_vec_prod_g_px;
  const auto hess_vec_prod_g_pp = hws.overlapped_hess_vec_prod_g_pp;

  // Check for early return
  if (hess_vec_prod_g_px.is_null() && hess_vec_prod_g_pp.is_null()) {
    return;
  }

  // Extract raw data from multivectors
  using mv_data_t = Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>;
  mv_data_t hess_vec_prod_g_px_data, hess_vec_prod_g_pp_data;
  if(!hess_vec_prod_g_px.is_null()) {
    const auto hess_vec_prod_g_px_data = Albany::getNonconstLocalData(hess_vec_prod_g_px);
  }
  if (!hess_vec_prod_g_pp.is_null()) {
    const auto hess_vec_prod_g_pp_data = Albany::getNonconstLocalData(hess_vec_prod_g_pp);
  }

  // Get some data from the discretization
  const int fieldLevel = level_it->second;
  const auto& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const auto& p_dof_mgr = workset.disc->getNewDOFManager(workset.dist_param_deriv_name);
  const auto& ov_p_indexer = p_dof_mgr->ov_indexer();
  const auto& elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto& node_dof_mgr = workset.disc->getNodeNewDOFManager();

  // Loop over cells in workset
  std::vector<GO> node_gids;
  for (size_t cell=0; cell < workset.numCells; ++cell) {
    // Get cell's node GIDs
    const auto elem_LID = elem_lids(cell);
    node_dof_mgr->getElementGIDs(elem_LID,node_gids);

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {

      auto lresp = this->local_response(cell,res);
      // Loop over nodes in cell
      for (int node=0; node<this->numNodes; ++node) {

        const GO base_id = layeredMeshNumbering.getColumnId(node_gids[node]);
        const GO ginode  = layeredMeshNumbering.getId(base_id, fieldLevel);
        const LO row     = ov_p_indexer->getLocalElement(ginode);

        if(row >=0) {
          if (!hess_vec_prod_g_px_data.is_null()) {
            hess_vec_prod_g_px_data[res][row] += lresp.dx(node).dx(0);
          }
          if (!hess_vec_prod_g_pp_data.is_null()) {
            hess_vec_prod_g_pp_data[res][row] += lresp.dx(node).dx(0);
          }
        }
      } // node
    } // response
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>::
evaluate2DFieldsDerivativesDueToExtrudedSolution(typename Traits::EvalData workset, std::string& sideset, Teuchos::RCP<const CellTopologyData> cellTopo)
{
  if (workset.sideSets == Teuchos::null) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Side sets not properly specified on the mesh" << std::endl);
  }

  // Check for early return
  if (workset.sideSets->count(sideset)==0) {
    return;
  }

  // Here we scatter the *local* response derivative
  const auto& hws = workset.hessianWorkset;
  const auto hess_vec_prod_g_xx = hws.overlapped_hess_vec_prod_g_xx;
  const auto hess_vec_prod_g_xp = hws.overlapped_hess_vec_prod_g_xp;

  // Check for early return
  if (hess_vec_prod_g_xx.is_null() && hess_vec_prod_g_xp.is_null()) {
    return;
  }

  // Extract raw data from multivectors
  using mv_data_t = Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>;
  mv_data_t hess_vec_prod_g_xp_data, hess_vec_prod_g_xx_data;
  if(!hess_vec_prod_g_xp.is_null()) {
    const auto hess_vec_prod_g_xp_data = Albany::getNonconstLocalData(hess_vec_prod_g_xp);
  }
  if (!hess_vec_prod_g_xx.is_null()) {
    const auto hess_vec_prod_g_xx_data = Albany::getNonconstLocalData(hess_vec_prod_g_xx);
  }

  const auto sol_dof_mgr = workset.disc->getNewDOFManager();
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const auto elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();

  const auto ov_node_indexer = node_dof_mgr->ov_indexer();

  const auto ALL = Kokkos::ALL();
  const int numLayers = layeredMeshNumbering.numLayers;
  const int neq = sol_dof_mgr->getNumFields();

  std::vector<GO> node_gids;
  const auto sideSet = workset.sideSets->at(sideset);
  for (size_t iside=0; iside<sideSet.size(); ++iside) {
    // Get the data that corresponds to the side
    const int elem_LID = sideSet[iside].elem_LID;
    const int elem_side = sideSet[iside].side_pos;
    const CellTopologyData_Subcell& side =  cellTopo->side[elem_side];
    const int numSideNodes = side.topology->node_count;

    // Get cell dofs lids
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    // Get cell node GIDs
    node_dof_mgr->getElementGIDs(elem_LID,node_gids);

    for (size_t res=0; res<this->global_response.size(); ++res) {
      auto val = this->local_response(elem_LID, res);

      for (int i = 0; i < numSideNodes; ++i) {
        const auto node = side.node[i];
        const GO base_id = layeredMeshNumbering.getColumnId(node_gids[node]);
        for (int il_col=0; il_col<numLayers+1; il_col++) {
          const GO ginode = layeredMeshNumbering.getId(base_id, il_col);
          const LO  inode = ov_node_indexer->getLocalElement(ginode);
          for (int eq_col=0; eq_col<neq; eq_col++) {
            const auto& offsets = sol_dof_mgr->getGIDFieldOffsets(eq_col);
            const LO dof = dof_lids(offsets[inode]);
            int deriv = neq *this->numNodes+il_col*neq*numSideNodes + neq*i + eq_col;
            if (!hess_vec_prod_g_xx_data.is_null()) {
              hess_vec_prod_g_xx_data[res][dof] += val.dx(deriv).dx(0);
            }
            if (!hess_vec_prod_g_xp_data.is_null()) {
              hess_vec_prod_g_xp_data[res][dof] += val.dx(deriv).dx(0);
            }
          }
        }
      }
    }
  }
}

} // namespace PHAL
