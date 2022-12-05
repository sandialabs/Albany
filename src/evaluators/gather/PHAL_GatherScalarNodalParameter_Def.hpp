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
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

  // Parameter dof numbering info
  const auto p_elem_dof_lids = p->elem_dof_lids().host();

  // Mesh elements
  const auto ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  for (std::size_t cell = 0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);
    for (std::size_t node=0; node<this->numNodes; ++node) {
      const LO lid = dof_lids(node);
      this->val(cell,node) = lid>=0 ? p_data[lid] : 0;
    }
  }
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

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

  // Parameter/nodes dof numbering info
  const auto p_dof_mgr    = workset.disc->getNewDOFManager(this->param_name);
  const auto p_indexer    = p_dof_mgr->ov_indexer();
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto layers_data  = workset.disc->getLayeredMeshNumbering();

  const auto ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);
  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
    for (std::size_t node=0; node<this->numNodes; ++node) {
      const GO base_id = layers_data->getColumnId(node_gids[node]);
      const GO ginode  = layers_data->getId(base_id, fieldLevel);
      const LO p_lid   = p_indexer->getLocalElement(ginode);
      this->val(cell,node) = p_lid>=0 ? p_data[p_lid] : 0;
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
  const auto dof_mgr      = workset.disc->getNewDOFManager();
  const auto p_elem_dof_lids = p->elem_dof_lids().host();

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
      for (std::size_t node=0; node<this->numNodes; ++node) {
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

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

  const auto Vp = workset.Vp;
  const auto Vp_data = !Vp.is_null() ? Albany::getLocalData(Vp) : Teuchos::null;

  // Parameter/solution/nodes dof numbering info
  const auto dof_mgr      = workset.disc->getNewDOFManager();
  const auto p_dof_mgr    = workset.disc->getNewDOFManager(this->param_name);
  const auto p_indexer    = p_dof_mgr->ov_indexer();
  const auto layers_data  = workset.disc->getLayeredMeshNumbering();
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();

  // Are we differentiating w.r.t. this parameter?
  const bool is_active = (workset.dist_param_deriv_name == this->param_name);

  const auto ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  // If active, initialize data needed for differentiation
  if (is_active) {
    const int neq = dof_mgr->getNumFields();
    const int num_deriv = this->numNodes;
    const bool trans = workset.transpose_dist_param_deriv;
    const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
    for (std::size_t cell=0; cell<workset.numCells; ++cell) {
      const auto elem_LID = elem_lids(cell);
      const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
      // const auto dof_lids  = Kokkos::subview(el_dof_lids,elem_LID,ALL);
      for (int node=0; node<num_deriv; ++node) {
        const GO base_id = layers_data->getColumnId(node_gids[node]);
        const GO ginode  = layers_data->getId(base_id, fieldLevel);

        const LO p_lid   = p_indexer->getLocalElement(ginode);
        const auto p_val = p_lid>=0 ? p_data[p_lid] : 0;

        ParamScalarT v(num_deriv, node, p_val);
        if(p_lid < 0) {
          v.fastAccessDx(node) = 0;
        }
        this->val(cell,node) = v;
      }

      if (Vp != Teuchos::null) {
        const int num_cols = workset.Vp->domain()->dim();

        auto& local_Vp = workset.local_Vp[cell];
        if (trans) {
          auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
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
            const GO base_id = layers_data->getColumnId(node_gids[node]);
            const GO ginode  = layers_data->getId(base_id, fieldLevel);
            const LO p_lid   = p_indexer->getLocalElement(ginode);
            local_Vp[node].resize(num_cols);
            for (int col=0; col<num_cols; ++col) {
              local_Vp[node][col] = p_lid>=0 ? Vp_data[col][p_lid] : 0;
            }
          }
        }
      }
    }
  } else {
    // If not active, just set the parameter value in the phalanx field
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const auto elem_LID = elem_lids(cell);
      const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
      for (std::size_t node=0; node<this->numNodes; ++node) {
        const GO base_id = layers_data->getColumnId(node_gids[node]);
        const GO ginode  = layers_data->getId(base_id, fieldLevel);
        const LO lid     = p_indexer->getLocalElement(ginode);
        this->val(cell,node) = lid>=0 ? p_data[lid] : 0;
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
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

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
  
  const auto vvec_data = is_p_direction_active ? Albany::getLocalData(vvec->col(0).getConst()) : Teuchos::null;

  const int num_nodes = this->numNodes;
  const int num_deriv = this->val(0,0).size();
  const int ws = workset.wsIndex;

  // Parameter/nodes dof numbering info
  const auto p_elem_dof_lids = p->elem_dof_lids().host();
  const auto elem_lids    = workset.disc->getElementLIDs_host(ws);
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto p_dof_mgr    = workset.disc->getNewDOFManager(this->param_name);

  std::vector<GO> node_gids;

  using ref_t = typename PHAL::Ref<ParamScalarT>::type;
  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<num_nodes; ++node) {
      const LO lid = p_dof_lids(node);

      // Initialize Fad type for parameter value
      const auto p_val = lid>=0 ? p_data[lid] : 0;

      ref_t val = this->val(cell,node);
      val = HessianVecFad(num_deriv, p_val);
      // If we differentiate w.r.t. this parameter, we have to set the first
      // derivative to 1
      if (is_p_active)
        val.fastAccessDx(node).val() = 1;
      // If we differentiate w.r.t. this parameter direction, we have to set
      // the second derivative to the related direction value
      if (is_p_direction_active)
        val.val().fastAccessDx(0) = lid>=0 ? vvec_data[lid] : 0;
    }
  }
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

  // Distributed parameter vector
  const auto p      = workset.distParamLib->get(this->param_name);
  const auto p_data = Albany::getLocalData(p->overlapped_vector().getConst());

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

  const int num_deriv = this->numNodes;
  const int ws        = workset.wsIndex;

  // Parameter/nodes dof numbering info
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto elem_lids    = workset.disc->getElementLIDs_host(ws);
  const auto layers_data  = workset.disc->getLayeredMeshNumbering();
  const auto p_dof_mgr    = workset.disc->getNewDOFManager(this->param_name);
  const auto p_indexer    = p_dof_mgr->ov_indexer();

  using ref_t = typename PHAL::Ref<ParamScalarT>::type;
  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
    for (int node=0; node<num_deriv; ++node) {
      const GO base_id = layers_data->getColumnId(node_gids[node]);
      const GO ginode  = layers_data->getId(base_id, fieldLevel);
      const LO p_lid   = p_indexer->getLocalElement(ginode);
      const auto p_val = p_lid>=0 ? p_data[p_lid] : 0;

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

} // namespace PHAL
