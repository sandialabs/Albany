//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_Utilities.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Albany_Hessian.hpp"
#include "Albany_DualView.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_KokkosUtils.hpp"

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

  auto dg_data = Albany::getNonconstDeviceData(dg);

  const auto& dof_mgr = workset.disc->getDOFManager();
  const int neq = dof_mgr->getNumFields();
  const int  ws = workset.wsIndex;

  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().dev();
  const auto elem_lids     = workset.disc->getWsElementLIDs();
  const auto elem_lids_dev = Kokkos::subview(elem_lids.dev(),ws,Kokkos::ALL);

  for (int eq_dof=0; eq_dof<neq; eq_dof++) {
    auto offsets = dof_mgr->getGIDFieldOffsetsKokkos(eq_dof);
    const int num_nodes = offsets.size();
    Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                         KOKKOS_CLASS_LAMBDA(const int& cell) {
      const auto elem_LID = elem_lids_dev(cell);

      for (int i=0; i<num_nodes; ++i) {
        const int deriv   = offsets(i);
        const int dof_lid = elem_dof_lids(elem_LID,deriv);

        // Loop over responses
        for (std::size_t res = 0; res < this->global_response.size(); res++) {
          auto val = this->local_response(cell, res);

          // Set dg/dx
          KU::atomic_add<ExecutionSpace>(&dg_data(dof_lid,res), val.dx(deriv));
        } // response
      } // column nodes
    }); // cell
  } // column equations
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::Jacobian, Traits>::
evaluate2DFieldsDerivativesDueToColumnContraction(
    typename Traits::EvalData workset,
    std::string& sidesetName)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
      "Side sets not properly specified on the mesh.\n");

  // Check for early return
  if (workset.sideSets->count(sidesetName)==0) {
    return;
  }

  // Here we scatter the *local* response derivative
  const auto dgdx    = workset.overlapped_dgdx;
  const auto dgdxdot = workset.overlapped_dgdxdot;
  const auto dg = dgdx.is_null() ? dgdxdot : dgdx;

  auto dg_data = Albany::getNonconstLocalData(dg);

  // Layers data
  const auto layers_data = workset.disc->getLayeredMeshNumberingLO();
  const int top = layers_data->top_side_pos;
  const int bot = layers_data->bot_side_pos;
  const int numLayers = layers_data->numLayers;

  // Dof mgr data
  const auto dof_mgr = workset.disc->getDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const int neq = dof_mgr->getNumFields();

#ifdef ALBANY_DEBUG
  // Ensure we have ONE cell per layer.
  const auto topo_base = dof_mgr->get_topology().getCellTopologyData()->base;

  const auto hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
  const auto wedge = shards::getCellTopologyData<shards::Wedge<6>>();

  TEUCHOS_TEST_FOR_EXCEPTION (
      topo_base==hexa || topo_base==wedge, std::logic_error,
      " [evaluate2DFieldsDerivativesDueToExtrudedSolution]\n"
      "   Feature only available for extruded meshes with one element per layer.\n");
#endif

  const int columnsOffset = neq*this->numNodes;
  const auto& sideSet = workset.sideSets->find(sidesetName)->second;
  for (const auto& side : sideSet) {
    const int side_elem_LID = side.ws_elem_idx;
    const int basal_elem_LID = layers_data->getColumnId(side_elem_LID);
    const int side_pos = side.side_pos;

    for (std::size_t res=0; res<this->global_response.size(); ++res) {
      auto val = this->local_response(side_elem_LID, res);

      const auto f = [&] (const int ilayer, const int pos)
      {
        const int elem_LID = layers_data->getId(basal_elem_LID,ilayer);
        const int ilevel = pos==bot ? ilayer : ilayer+1;
        for (int eq=0; eq<neq; ++eq) {
          // Note: grab offsets on top/bot ordered in the same way as on sideset side,
          //       to guarantee corresponding nodes are vertically aligned.
          const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq,pos,side_pos);
          const int numSideNodes = offsets.size();
          for (int i=0; i<numSideNodes; ++i) {
            int deriv = columnsOffset+ilevel*neq*numSideNodes+neq*i+eq;
            const LO x_lid = elem_dof_lids(elem_LID,offsets[i]);
            dg_data[res][x_lid] += val.dx(deriv);
          }
        }
      };

      // On all layer, execute f on bot nodes
      for (int ilayer=0; ilayer<numLayers; ++ilayer) {
        f(ilayer,bot);
      }
      // On last layer, also execute f on top nodes
      f(numLayers-1,top);
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
    Albany::ThyraVDeviceView<ST> g_nonconstView = Albany::getNonconstDeviceData(g);
    MDFieldVectorRight<const ScalarT> gr(this->global_response);
    global_response_reader = gr;
    Kokkos::parallel_for(this->getName(),
                        Kokkos::RangePolicy<ExecutionSpace>(0,this->global_response.size()),
                        KOKKOS_CLASS_LAMBDA(const int i) {
      g_nonconstView(i) = global_response_reader[i].val();
    });
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
  const auto dgdp_data = Albany::getNonconstDeviceData(dgdp);
  const int  ws = workset.wsIndex;
  const int num_deriv = numNodes;

  const auto elem_lids_all = workset.disc->getWsElementLIDs();
  const auto elem_lids = Kokkos::subview(elem_lids_all.dev(),ws,ALL);

  const auto param = workset.distParamLib->get(workset.dist_param_deriv_name);
  const auto p_elem_dof_lids = param->get_dof_mgr()->elem_dof_lids().dev();

  // Loop over cells in workset
  Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      // Loop over nodes in cell
      for (int deriv=0; deriv<num_deriv; ++deriv) {

        // If param defined at this node, update dg/dp
        const int row = dof_lids(deriv);
        if(row >=0){
          KU::atomic_add<ExecutionSpace>(&dgdp_data(row,res), this->local_response(cell, res).dx(deriv));
        }
      } // deriv
    } // response
  }); // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  auto g = workset.g;
  if (g != Teuchos::null) {
    auto g_nonconstView = Albany::getNonconstDeviceData(g);
    MDFieldVectorRight<const ScalarT> gr(this->global_response);
    global_response_reader = gr;
    Kokkos::parallel_for(this->getName(),
                        Kokkos::RangePolicy<ExecutionSpace>(0,this->global_response.size()),
                        KOKKOS_CLASS_LAMBDA(const int i) {
      g_nonconstView(i) = global_response_reader[i].val();
    });
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
  const auto  elem_lids    = workset.disc->getElementLIDs_host(ws);

  const auto& layers_data     = workset.disc->getLayeredMeshNumberingLO();
  const int top = layers_data->top_side_pos;
  const int bot = layers_data->bot_side_pos;
  const auto& p_dof_mgr       = workset.disc->getDOFManager(param_name);
  const auto& p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();

  const int fieldLevel = level_it->second;
  const int fieldLayer = fieldLevel==layers_data->numLayers ? fieldLevel-1 : fieldLevel;
  const int field_pos = fieldLayer==fieldLevel ? bot : top;

  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto& top_offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,top,field_pos);
  const auto& bot_offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_pos);

  const auto& field_offsets = fieldLayer==fieldLevel ? bot_offsets : top_offsets;
  const int numSideNodes = field_offsets.size();

  // Loop over cells in workset
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto basal_elem_LID = layers_data->getColumnId(elem_LID);
    const auto field_elem_LID = layers_data->getId(basal_elem_LID,fieldLayer);

    // Loop over responses
    for (size_t res=0; res<this->global_response.size(); ++res) {
      const auto lresp = this->local_response(cell,res);

      // We process bot and bot nodes separately
      for (int node=0; node<numSideNodes; ++node) {
        const LO row = p_elem_dof_lids(field_elem_LID,field_offsets[node]);
        if (row>=0) {
          dgdp_data[res][row] += lresp.dx(bot_offsets[node]);
          dgdp_data[res][row] += lresp.dx(top_offsets[node]);
        }
      }
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
  using mv_data_t = Albany::ThyraMVDeviceView<ST>;
  mv_data_t hess_vec_prod_g_xx_data, hess_vec_prod_g_xp_data,
                 hess_vec_prod_g_px_data, hess_vec_prod_g_pp_data;

  if (!hess_vec_prod_g_xx.is_null())
    hess_vec_prod_g_xx_data = Albany::getNonconstDeviceData(hess_vec_prod_g_xx);
  if (!hess_vec_prod_g_xp.is_null())
    hess_vec_prod_g_xp_data = Albany::getNonconstDeviceData(hess_vec_prod_g_xp);
  if (!hess_vec_prod_g_px.is_null())
    hess_vec_prod_g_px_data = Albany::getNonconstDeviceData(hess_vec_prod_g_px);
  if (!hess_vec_prod_g_pp.is_null())
    hess_vec_prod_g_pp_data = Albany::getNonconstDeviceData(hess_vec_prod_g_pp);

  const bool do_xx        = !hess_vec_prod_g_xx.is_null();
  const bool do_xp        = !hess_vec_prod_g_xp.is_null();
  const bool do_scalar_px = !hess_vec_prod_g_px.is_null() && !distributed;
  const bool do_scalar_pp = !hess_vec_prod_g_pp.is_null() && !distributed;
  const bool do_dist_px   = !hess_vec_prod_g_px.is_null() &&  distributed;
  const bool do_dist_pp   = !hess_vec_prod_g_pp.is_null() &&  distributed;

  // Get some data from the discretization
  const auto param_name = workset.dist_param_deriv_name;
  Albany::DualView<const int**>::dev_t p_elem_dof_lids;
  if (do_dist_pp || do_dist_px) {
    auto dist_param = workset.distParamLib->get(param_name);
    p_elem_dof_lids = dist_param->get_dof_mgr()->elem_dof_lids().dev();
  }

  const auto& dof_mgr   = workset.disc->getDOFManager();

  const int  ws = workset.wsIndex;
  const int neq = dof_mgr->getNumFields();

  const int g_px_size = hess_vec_prod_g_px_data.extent(1);
  const int g_pp_size = hess_vec_prod_g_pp_data.extent(1);

  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().dev();
  const auto elem_lids_all  = workset.disc->getWsElementLIDs();
  const auto elem_lids = Kokkos::subview(elem_lids_all.dev(),ws,Kokkos::ALL);

  for (int eq_dof=0; eq_dof<neq; eq_dof++) {
    // Get offsets of this dof in the lids array
    auto offsets = dof_mgr->getGIDFieldOffsetsKokkos(eq_dof);
    const int num_nodes = offsets.size();
    Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                         KOKKOS_CLASS_LAMBDA(const int& cell) {
      const auto elem_LID = elem_lids(cell);
      // Loop over responses
      for (size_t res=0; res<this->global_response.size(); ++res) {
        auto lresp = this->local_response(cell, res);

        if (do_xx || do_xp) {
          for (int i=0; i<num_nodes; ++i) {

            const int deriv   = offsets(i);
            const int dof_lid = elem_dof_lids(elem_LID,deriv);

            if (do_xx)
              KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_g_xx_data(dof_lid, res)), lresp.dx(deriv).dx(0));

            if (do_xp)
              KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_g_xp_data(dof_lid, res)), lresp.dx(deriv).dx(0));
          } // column nodes
        }
        if (do_dist_px) {
          for (int deriv=0; deriv<numNodes; ++deriv) {
            const int row = p_elem_dof_lids(elem_LID,deriv);
            if (row>=0) {
              KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_g_px_data(row,res)), lresp.dx(deriv).dx(0));
            }
          }
        }
        if (do_dist_pp) {
          for (int deriv=0; deriv<numNodes; ++deriv) {
            const int row = p_elem_dof_lids(elem_LID,deriv);
            if (row>=0) {
              KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_g_pp_data(row,res)), lresp.dx(deriv).dx(0));
            }
          }
        }

        if (do_scalar_px) {
          for (int deriv=0; deriv<g_px_size; ++deriv) {
            KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_g_px_data(deriv,res)), lresp.dx(deriv).dx(0));
          }
        }
        if (do_scalar_pp) {
          for (int deriv=0; deriv<g_pp_size; ++deriv) {
            KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_g_pp_data(deriv,res)), lresp.dx(deriv).dx(0));
          }
        }
      } // response
    }); // cell
  } // column equations
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
    Albany::ThyraVDeviceView<ST> g_nonconstView = Albany::getNonconstDeviceData(g);
    MDFieldVectorRight<const ScalarT> gr(this->global_response);
    global_response_reader = gr;
    Kokkos::parallel_for(this->getName(),
                        Kokkos::RangePolicy<ExecutionSpace>(0,this->global_response.size()),
                        KOKKOS_CLASS_LAMBDA(const int i) {
      g_nonconstView(i) = this->global_response[i].val().val();
    });
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
    hess_vec_prod_g_px_data = Albany::getNonconstLocalData(hess_vec_prod_g_px);
  }
  if (!hess_vec_prod_g_pp.is_null()) {
    hess_vec_prod_g_pp_data = Albany::getNonconstLocalData(hess_vec_prod_g_pp);
  }

  // Mesh data
  const auto layers_data = workset.disc->getLayeredMeshNumberingLO();
  const int top = layers_data->top_side_pos;
  const int bot = layers_data->bot_side_pos;
  const int numLayers = layers_data->numLayers;
  const auto& elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);

  // Solution dof mgr data
  const auto dof_mgr = workset.disc->getDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  // Parameter data
  const int fieldLevel = level_it->second;
  const int fieldLayer = fieldLevel==numLayers ? fieldLevel-1 : fieldLevel;
  const int field_pos = fieldLayer==fieldLevel ? bot : top;
  const auto& p_dof_mgr = workset.disc->getDOFManager(workset.dist_param_deriv_name);
  const auto& p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto& top_offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,top,field_pos);
  const auto& bot_offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_pos);
  const auto& field_offsets = fieldLevel==fieldLayer ? bot_offsets : top_offsets;

  const int numSideNodes = field_offsets.size();

  // Loop over cells in workset
  for (size_t cell=0; cell < workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto basal_elem_LID = layers_data->getColumnId(elem_LID);
    const auto field_elem_LID = layers_data->getId(basal_elem_LID,fieldLayer);

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {

      auto lresp = this->local_response(cell,res);
      const auto f = [&] (const int node, const int row) {
        if (!hess_vec_prod_g_px_data.is_null()) {
          hess_vec_prod_g_px_data[res][row] += lresp.dx(node).dx(0);
        }
        if (!hess_vec_prod_g_pp_data.is_null()) {
          hess_vec_prod_g_pp_data[res][row] += lresp.dx(node).dx(0);
        }
      };

      for (int inode=0; inode<numSideNodes; ++inode) {
        const LO row = p_elem_dof_lids(field_elem_LID,field_offsets[inode]);
        if (row>=0) {
          // Run lambda for both top and bottom nodes
          f(top_offsets[inode],row);
          f(bot_offsets[inode],row);
        }
      }

    } // response
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<AlbanyTraits::HessianVec, Traits>::
evaluate2DFieldsDerivativesDueToColumnContraction(typename Traits::EvalData workset, std::string& sideset)
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
    hess_vec_prod_g_xp_data = Albany::getNonconstLocalData(hess_vec_prod_g_xp);
  }
  if (!hess_vec_prod_g_xx.is_null()) {
    hess_vec_prod_g_xx_data = Albany::getNonconstLocalData(hess_vec_prod_g_xx);
  }

  // Layers data
  const auto layers_data = workset.disc->getLayeredMeshNumberingLO();
  const int top = layers_data->top_side_pos;
  const int bot = layers_data->bot_side_pos;
  const int numLayers = layers_data->numLayers;

  // Dof mgr data
  const auto dof_mgr = workset.disc->getDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const int neq = dof_mgr->getNumFields();
  const int columnsOffset = neq*this->numNodes;

  const auto& sideSet = workset.sideSets->at(sideset);
  for (const auto& side : sideSet) {
    const int side_elem_LID = side.ws_elem_idx;
    const int basal_elem_LID = layers_data->getColumnId(side_elem_LID);
    const int side_pos = side.side_pos;

    for (size_t res=0; res<this->global_response.size(); ++res) {
      auto val = this->local_response(side_elem_LID, res);
      const auto f = [&] (const int ilayer, const int pos)
      {
        const int elem_LID = layers_data->getId(basal_elem_LID,ilayer);
        const int ilevel = pos==bot ? ilayer : ilayer+1;
        for (int eq=0; eq<neq; ++eq) {
          // Note: grab offsets on top/bot ordered in the same way as on sideset side,
          //       to guarantee corresponding nodes are vertically aligned.
          const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq,pos,side_pos);
          const int numSideNodes = offsets.size();
          for (int i=0; i<numSideNodes; ++i) {
            int deriv = columnsOffset + ilevel*neq*numSideNodes + neq*i + eq;
            const LO x_lid = elem_dof_lids(elem_LID,offsets[i]);
            if (!hess_vec_prod_g_xx_data.is_null()) {
              hess_vec_prod_g_xx_data[res][x_lid] += val.dx(deriv).dx(0);
            }
            if (!hess_vec_prod_g_xp_data.is_null()) {
              hess_vec_prod_g_xp_data[res][x_lid] += val.dx(deriv).dx(0);
            }
          }
        }
      };

      // On all layer, execute f on bot nodes
      for (int ilayer=0; ilayer<numLayers; ++ilayer) {
        f(ilayer,bot);
      }
      // On last layer, also execute f on top nodes
      f(numLayers-1,top);
    }
  }
}

} // namespace PHAL
