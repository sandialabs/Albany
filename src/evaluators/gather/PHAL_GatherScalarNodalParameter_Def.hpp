//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_GatherScalarNodalParameter.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherScalarNodalParameterBase<EvalT,Traits>::
GatherScalarNodalParameterBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl) :
    numNodes(dl->node_scalar->extent(1)),
    param_name(p.get<std::string>("Parameter Name"))
{
  std::string field_name = p.isParameter("Field Name") ? p.get<std::string>("Field Name") : param_name;
  val = PHX::MDField<ParamScalarT,Cell,Node>(field_name,dl->node_scalar);

  this->addEvaluatedField(val);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameterBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),d.memoizer_for_params_active());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// **********************************************************************
template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,dl)
{
  this->setName("GatherNodalParameter("+this->param_name+")"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  this->setName("GatherNodalParameter("+this->param_name+")"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  constexpr auto ALL = Kokkos::ALL();

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getDeviceData(p->overlapped_vector().getConst());

  // Parameter dof numbering info
  const auto p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().dev();

  // Mesh elements
  const auto ws = workset.wsIndex;
  const auto elem_lids  = workset.disc->getWsElementLIDs();
  const auto elem_lids_dev = Kokkos::subview(elem_lids.dev(),ws,ALL);

  Kokkos::parallel_for(RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids_dev(cell);
    const auto dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<this->numNodes; ++node) {
      const LO lid = dof_lids(node);
      this->val(cell,node) = lid>=0 ? p_data(lid) : 0;
    }
  });
}

// **********************************************************************
template<typename EvalT, typename Traits>
GatherScalarExtruded2DNodalParameter<EvalT, Traits>::
GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                                     const Teuchos::RCP<Albany::Layouts>& dl) :
    GatherScalarNodalParameterBase<EvalT, Traits>(p, dl),
    fieldLevel(p.get<int>("Field Level"))
{
  this->setName("GatherScalarExtruded2DNodalParameter("+this->param_name+")"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarExtruded2DNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  const auto layers_data  = workset.disc->getMeshStruct()->local_cell_layers_data;
  const auto bot = layers_data->bot_side_pos;
  const auto top = layers_data->top_side_pos;
  const auto ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  // Pick element layer that contains the field level
  const auto fieldLayer = fieldLevel==layers_data->numLayers
                        ? fieldLevel-1 : fieldLevel;
  const int field_pos = fieldLayer==fieldLevel ? bot : top;

  // Distributed parameter vector
  const auto& p      = workset.distParamLib->get(this->param_name);
  const auto  p_data = Albany::getLocalData(p->overlapped_vector().getConst());

  // Parameter dof numbering info
  const auto& p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().host();

  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto& offsets_top = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,top,field_pos);
  const auto& offsets_bot = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,bot,field_pos);
  const auto& offsets_p   = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,field_pos);
  const int num_nodes_2d = offsets_p.size();

  // Idea: loop over cells. Grab p data from a cell at the right layer,
  //       using offsets that correspond to the elem-side where the param is defined.
  //       Inside, loop over 2d nodes, and process top/bot sides separately
  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto basal_elem_LID = layers_data->getColumnId(elem_LID);
    const auto param_elem_LID = layers_data->getId(basal_elem_LID,fieldLayer);

    for (int node2d=0; node2d<num_nodes_2d; ++node2d) {
      const LO p_lid = p_elem_dof_lids(param_elem_LID,offsets_p[node2d]);
      const auto p_val = p_lid>=0 ? p_data[p_lid] : 0;
      for (auto node : {offsets_bot[node2d], offsets_top[node2d]}) {
        this->val(cell,node) = p_val;
      }
    }
  }
}

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************A


// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
  this->setName("GatherNodalParameter("+this->param_name+")"+PHX::print<PHAL::AlbanyTraits::DistParamDeriv>());
}

// **********************************************************************
template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  this->setName("GatherNodalParameter("+this->param_name+")"+PHX::print<PHAL::AlbanyTraits::DistParamDeriv>());
}

// **********************************************************************
template<typename Traits>
void GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  constexpr auto ALL = Kokkos::ALL();

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

  const auto Vp = workset.Vp;
  const auto Vp_data = !Vp.is_null() ? Albany::getLocalData(Vp) : Teuchos::null;

  // Parameter/solution/nodes dof numbering info
  const auto dof_mgr      = workset.disc->getDOFManager();
  const auto p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().host();

  const auto ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  // Are we differentiating w.r.t. this parameter?
  const bool is_active = (workset.dist_param_deriv_name == this->param_name);

  // If active, initialize data needed for differentiation
  if (is_active) {
    const int neq = dof_mgr->getNumFields();
    const int num_deriv = this->numNodes;
    bool trans = workset.transpose_dist_param_deriv;
    const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
    for (std::size_t cell=0; cell<workset.numCells; ++cell) {
      const auto elem_LID = elem_lids(cell);
      const auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);
      for (int node=0; node<num_deriv; ++node) {
        const LO lid = p_dof_lids(node);

        // Initialize Fad type for parameter value
        const auto p_val = lid>=0 ? p_data[lid] : 0;
        ParamScalarT v(num_deriv, node, p_val);
        this->val(cell,node) = v;
      }

      if (Vp != Teuchos::null) {
        const int num_cols = Vp->domain()->dim();

        auto& local_Vp = workset.local_Vp[cell];

        if (trans) {
          auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
          // const auto& offsets = this->m_sol_fields_offsets;
          local_Vp.resize(dof_lids.size());
          for (int eq=0; eq<neq; ++eq) {
            const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
            for (const auto o : offsets) {
              local_Vp[o].resize(num_cols);
              const LO lid = dof_lids(o);
              for (int col=0; col<num_cols; ++col)
                local_Vp[o][col] = Vp_data[col][lid];
            }
          }
        } else {
          local_Vp.resize(num_deriv);
          for (int node=0; node<num_deriv; ++node) {
            const LO lid = p_dof_lids(node);
            local_Vp[node].resize(num_cols);
            for (int col=0; col<num_cols; ++col)
              local_Vp[node][col] = lid>=0 ? Vp_data[col][lid] : 0;
          }
        }
      }
    }
  } else {
    // If not active, just set the parameter value in the phalanx field
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const auto elem_LID = elem_lids(cell);
      const auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);
      for (int node=0; node<this->numNodes; ++node) {
        const LO lid = p_dof_lids(node);
        this->val(cell,node) = lid>=0 ? p_data[lid] : 0;
      }
    }
  }
}

// **********************************************************************
template<typename Traits>
GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                                     const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p, dl),
  fieldLevel(p.get<int>("Field Level"))
{
  this->setName("GatherExtruded2DNodalParameter("+this->param_name+")"+
      PHX::print<PHAL::AlbanyTraits::DistParamDeriv>());
}

// **********************************************************************
template<typename Traits>
void GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  constexpr auto ALL = Kokkos::ALL();

  const auto layers_data  = workset.disc->getMeshStruct()->local_cell_layers_data;
  const auto bot = layers_data->bot_side_pos;
  const auto top = layers_data->top_side_pos;
  const auto ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  // Pick element layer that contains the field level
  const auto fieldLayer = fieldLevel==layers_data->numLayers
                        ? fieldLevel-1 : fieldLevel;
  const int field_pos = fieldLayer==fieldLevel ? bot : top;

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

  const auto Vp = workset.Vp;
  const auto Vp_data = !Vp.is_null() ? Albany::getLocalData(Vp) : Teuchos::null;

  // Parameter/solution dof numbering info
  const auto& sol_dof_mgr = workset.disc->getDOFManager();
  const auto& p_dof_mgr       = p->get_dof_mgr();
  const auto& elem_dof_lids   = sol_dof_mgr->elem_dof_lids().host();
  const auto& p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().host();

  // Idea: loop over cells. Grab p data from a cell at the right layer,
  //       using offsets that correspond to the elem-side where the param is defined.
  //       Inside, loop over 2d nodes, and process top/bot sides separately

  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto& offsets_top = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,top,field_pos);
  const auto& offsets_bot = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,bot,field_pos);
  const auto& offsets_p   = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,field_pos);
  const int num_nodes_2d = offsets_p.size();

  // Are we differentiating w.r.t. this parameter?
  const bool is_active = (workset.dist_param_deriv_name == this->param_name);

  // If active, initialize data needed for differentiation
  if (is_active) {
    const int neq = sol_dof_mgr->getNumFields();
    const int num_deriv = this->numNodes;
    const bool trans = workset.transpose_dist_param_deriv;
    for (std::size_t cell=0; cell<workset.numCells; ++cell) {
      const auto elem_LID = elem_lids(cell);
      const auto basal_elem_LID = layers_data->getColumnId(elem_LID);
      const auto param_elem_LID = layers_data->getId(basal_elem_LID,fieldLayer);
      for (int node2d=0; node2d<num_nodes_2d; ++node2d) {
        const LO p_lid = p_elem_dof_lids(param_elem_LID,offsets_p[node2d]);
        const auto p_val = p_lid>=0 ? p_data[p_lid] : 0;
        for (auto node : {offsets_bot[node2d], offsets_top[node2d]}) {
          ParamScalarT v(num_deriv, node, p_val);
          if(p_lid < 0) {
            v.fastAccessDx(node) = 0;
          }
          this->val(cell,node) = v;
        }
      }

      if (Vp != Teuchos::null) {
        const int num_cols = workset.Vp->domain()->dim();

        auto& local_Vp = workset.local_Vp[cell];
        if (trans) {
          auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
          local_Vp.resize(dof_lids.size());
          for (int eq=0; eq<neq; ++eq) {
            const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(eq);
            for (const auto o : sol_offsets) {
              local_Vp[o].resize(num_cols);
              const LO lid = dof_lids(o);
              for (int col=0; col<num_cols; ++col)
                local_Vp[o][col] = Vp_data[col][lid];
            }
          }
        } else {
          local_Vp.resize(num_deriv);
          for (int node2d=0; node2d<num_nodes_2d; ++node2d) {
            const LO p_lid = p_elem_dof_lids(param_elem_LID,offsets_p[node2d]);
            for (auto node : {offsets_bot[node2d], offsets_top[node2d]}) {
              local_Vp[node].resize(num_cols);
              for (int col=0; col<num_cols; ++col) {
                local_Vp[node][col] = p_lid>=0 ? Vp_data[col][p_lid] : 0;
              }
            }
          }
        }
      }
    }
  } else {
    // If not active, just set the parameter value in the phalanx field
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const auto elem_LID = elem_lids(cell);
      const auto basal_elem_LID = layers_data->getColumnId(elem_LID);
      const auto param_elem_LID = layers_data->getId(basal_elem_LID,fieldLayer);
      for (int node2d=0; node2d<num_nodes_2d; ++node2d) {
        const LO p_lid = p_elem_dof_lids(param_elem_LID,offsets_p[node2d]);
        const auto p_val = p_lid>=0 ? p_data[p_lid] : 0;
        for (auto node : {offsets_bot[node2d], offsets_top[node2d]}) {
          this->val(cell,node) = p_val;
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************
template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl)
{
  this->setName("GatherNodalParameter("+this->param_name+")"+PHX::print<PHAL::AlbanyTraits::HessianVec>());
}

// **********************************************************************
template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  this->setName("GatherNodalParameter("+this->param_name+")"+PHX::print<PHAL::AlbanyTraits::HessianVec>());
}

// **********************************************************************
template<typename Traits>
void GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  constexpr auto ALL = Kokkos::ALL();

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  Albany::DeviceView1d<const ST> p_data = Albany::getDeviceData(p->overlapped_vector().getConst());

  // Direction vector for the Hessian-vector product
  const auto vvec = workset.hessianWorkset.direction_p;

  const auto& hessian_ws = workset.hessianWorkset;

  const bool g_xp_is_active = !hessian_ws.hess_vec_prod_g_xp.is_null();
  const bool g_px_is_active = !hessian_ws.hess_vec_prod_g_px.is_null();
  const bool g_pp_is_active = !hessian_ws.hess_vec_prod_g_pp.is_null();
  const bool f_xp_is_active = !hessian_ws.hess_vec_prod_f_xp.is_null();
  const bool f_px_is_active = !hessian_ws.hess_vec_prod_f_px.is_null();
  const bool f_pp_is_active = !hessian_ws.hess_vec_prod_f_pp.is_null();

  // is_p_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_px, Hv_g_pp, Hv_f_px, or Hv_f_pp, i.e. if the first derivative is w.r.t. this parameter.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  const bool is_p_active = (workset.dist_param_deriv_name == this->param_name)
    && (g_px_is_active || g_pp_is_active || f_px_is_active || f_pp_is_active);

  // is_p_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xp, Hv_g_pp, Hv_f_xp, or Hv_f_pp, i.e. if the second derivative is w.r.t. this parameter direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_p_direction_active = (hessian_ws.dist_param_deriv_direction_name == this->param_name)
    && (g_xp_is_active || g_pp_is_active || f_xp_is_active || f_pp_is_active);

  TEUCHOS_TEST_FOR_EXCEPTION(
      is_p_direction_active && vvec.is_null(),
      Teuchos::Exceptions::InvalidParameter,
      "\nError in GatherScalarNodalParameter<HessianVec, Traits>: "
      "direction_p is not set and the direction is active.\n");
  
  Albany::DeviceView1d<const ST> vvec_data;
  if (is_p_direction_active) vvec_data = Albany::getDeviceData(vvec->col(0).getConst());

  const int ws = workset.wsIndex;

  // Parameter/nodes dof numbering info
  const auto p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().dev();
  const auto elem_lids  = workset.disc->getWsElementLIDs();
  const auto elem_lids_dev = Kokkos::subview(elem_lids.dev(),ws,ALL);

  using ref_t = typename PHAL::Ref<ParamScalarT>::type;
  Kokkos::parallel_for(RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const int num_deriv = this->val(0,0).size();
    const auto elem_LID = elem_lids_dev(cell);
    const auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<this->numNodes; ++node) {
      const LO lid = p_dof_lids(node);

      // Initialize Fad type for parameter value
      const auto p_val = lid>=0 ? p_data(lid) : 0;

      ref_t val_ref = this->val(cell,node);
      val_ref = HessianVecFad(num_deriv, p_val);
      // If we differentiate w.r.t. this parameter, we have to set the first
      // derivative to 1
      if (is_p_active)
        val_ref.fastAccessDx(node).val() = 1;
      // If we differentiate w.r.t. this parameter direction, we have to set
      // the second derivative to the related direction value
      if (is_p_direction_active)
        val_ref.val().fastAccessDx(0) = lid>=0 ? vvec_data(lid) : 0;
    }
  });
}

// **********************************************************************
template<typename Traits>
GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                                     const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>(p, dl),
  fieldLevel(p.get<int>("Field Level"))
{
  this->setName("GatherExtruded2DNodalParameter("+this->param_name+")"+
    PHX::print<PHAL::AlbanyTraits::HessianVec>());
}

// **********************************************************************
template<typename Traits>
void GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (this->memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  const auto layers_data  = workset.disc->getLayeredMeshNumberingLO();
  const auto bot = layers_data->bot_side_pos;
  const auto top = layers_data->top_side_pos;
  const auto ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  // Direction vector for the Hessian-vector product
  const auto vvec = workset.hessianWorkset.direction_p;

  const auto& hessian_ws = workset.hessianWorkset;

  const bool g_xp_is_active = !hessian_ws.hess_vec_prod_g_xp.is_null();
  const bool g_px_is_active = !hessian_ws.hess_vec_prod_g_px.is_null();
  const bool g_pp_is_active = !hessian_ws.hess_vec_prod_g_pp.is_null();
  const bool f_xp_is_active = !hessian_ws.hess_vec_prod_f_xp.is_null();
  const bool f_px_is_active = !hessian_ws.hess_vec_prod_f_px.is_null();
  const bool f_pp_is_active = !hessian_ws.hess_vec_prod_f_pp.is_null();

  // is_p_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_px, Hv_g_pp, Hv_f_px, or Hv_f_pp, i.e. if the first derivative is w.r.t. this parameter.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .dx().fastAccessDx().
  const bool is_p_active = workset.dist_param_deriv_name==this->param_name
    && (g_px_is_active || g_pp_is_active || f_px_is_active || f_pp_is_active);

  // is_p_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xp, Hv_g_pp, Hv_f_xp, or Hv_f_pp, i.e. if the second derivative is w.r.t. this parameter direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .fastAccessDx().dx().
  const bool is_p_direction_active = hessian_ws.dist_param_deriv_direction_name==this->param_name
    && (g_xp_is_active || g_pp_is_active || f_xp_is_active || f_pp_is_active);

  TEUCHOS_TEST_FOR_EXCEPTION(
      is_p_direction_active && vvec.is_null(),
      Teuchos::Exceptions::InvalidParameter,
      "\nError in GatherScalarExtruded2DNodalParameter<HessianVec, Traits>: "
      "direction_p is not set and the direction is acrive.\n");
  const auto vvec_data = is_p_direction_active ? Albany::getLocalData(vvec->col(0).getConst()) : Teuchos::null;

  // Pick element layer that contains the field level
  const auto fieldLayer = fieldLevel==layers_data->numLayers
                        ? fieldLevel-1 : fieldLevel;
  const int field_pos = fieldLayer==fieldLevel ? bot : top;

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

  // Parameter dof numbering info
  const auto p_dof_mgr        = p->get_dof_mgr();
  const auto& p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().host();

  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto& offsets_top = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,top,field_pos);
  const auto& offsets_bot = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,bot,field_pos);
  const auto& offsets_p   = p->get_dof_mgr()->getGIDFieldOffsetsSide(0,field_pos);
  const int num_nodes_2d = offsets_p.size();

  using ref_t = typename PHAL::Ref<ParamScalarT>::type;
  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto basal_elem_LID = layers_data->getColumnId(elem_LID);
    const auto param_elem_LID = layers_data->getId(basal_elem_LID,fieldLayer);
    for (int node2d=0; node2d<num_nodes_2d; ++node2d) {
      const LO p_lid = p_elem_dof_lids(param_elem_LID,offsets_p[node2d]);
      const auto p_val = p_lid>=0 ? p_data[p_lid] : 0;
      for (auto node : {offsets_bot[node2d], offsets_top[node2d]}) {
        ref_t val = this->val(cell,node);
        val = HessianVecFad(val.size(), p_val);
        // If we differentiate w.r.t. this parameter, we have to set the first
        // derivative to 1
        if (is_p_active)
          val.fastAccessDx(node).val() = 1;
        // If we differentiate w.r.t. this parameter direction, we have to set
        // the second derivative to the related direction value
        if (is_p_direction_active)
          val.val().fastAccessDx(0) = p_lid>=0 ? vvec_data[p_lid] : 0;
      }
    }
  }
}

} // namespace PHAL
