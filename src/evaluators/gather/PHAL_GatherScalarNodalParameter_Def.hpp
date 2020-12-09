//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_GatherScalarNodalParameter.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

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

  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lid = wsElDofs((int)cell,(int)node,0);
      (this->val)(cell,node) = (lid >= 0 ) ? pvec_constView[lid] : 0;
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

  // TODO: find a way to abstract away from the map concept. Perhaps using Panzer::ConnManager?
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  // auto overlapNodeVS = workset.disc->getOverlapNodeVectorSpace();
  // auto ov_node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);
  auto pspace_indexer = Albany::createGlobalLocalIndexer(pvec->space());
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
      const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
      const LO p_lid= pspace_indexer->getLocalElement(ginode);
      (this->val)(cell,node) = ( p_lid >= 0) ? pvec_constView[p_lid] : 0;
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

  // Distributed parameter vector
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  Teuchos::RCP<const Thyra_MultiVector> Vp = workset.Vp;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vp_data;
  if (!Vp.is_null()) {
    Vp_data = Albany::getLocalData(workset.Vp);
  }

  auto nodeID = workset.wsElNodeEqID;
  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  // Are we differentiating w.r.t. this parameter?
  bool is_active = (workset.dist_param_deriv_name == this->param_name);

  // If active, intialize data needed for differentiation
  if (is_active) {
    const int num_deriv = this->numNodes;
    const int num_nodes_res = this->numNodes;
    bool trans = workset.transpose_dist_param_deriv;
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (int node = 0; node < num_deriv; ++node) {

        // Initialize Fad type for parameter value
        const LO id = wsElDofs((int)cell,(int)node,0);
        double pvec_id = (id >= 0) ? pvec_constView[id] : 0;
        ParamScalarT v(num_deriv, node, pvec_id);
        (this->val)(cell,node) = v;
      }

      if (workset.Vp != Teuchos::null) {
        const int num_cols = Vp->domain()->dim();

        Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp = workset.local_Vp[cell];

        if (trans) {
          local_Vp.resize(num_nodes_res*workset.numEqs);
          for (int node = 0; node < num_nodes_res; ++node) {
            // Store Vp entries
            for (std::size_t eq = 0; eq < workset.numEqs; eq++) {
              local_Vp[node*workset.numEqs+eq].resize(num_cols);
              const LO id = nodeID(cell,node,eq);
              for (int col=0; col<num_cols; ++col)
                local_Vp[node*workset.numEqs+eq][col] = Vp_data[col][id];
            }
          }
        } else {
          local_Vp.resize(num_deriv);
          for (int node=0; node<num_deriv; ++node) {
            const LO id = wsElDofs((int)cell,node,0);
            local_Vp[node].resize(num_cols);
            for (int col=0; col<num_cols; ++col)
              local_Vp[node][col] = (id >= 0) ? Vp_data[col][id] : 0;
          }
        }
      }
    }
  } else {
    // If not active, just set the parameter value in the phalanx field
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        const LO lid = wsElDofs((int)cell,(int)node,0);
        (this->val)(cell,node) = (lid >= 0) ? pvec_constView[lid] : 0;
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

  // TODO: find a way to abstract away from the map concept. Perhaps using Panzer::ConnManager?
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  // Are we differentiating w.r.t. this parameter?
  bool is_active = (workset.dist_param_deriv_name == this->param_name);

  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  auto nodeID = workset.wsElNodeEqID;

  // If active, intialize data needed for differentiation
  auto p_indexer = Albany::createGlobalLocalIndexer(pvec->space());
  if (is_active) {
    const int num_deriv = this->numNodes;
    const int num_nodes_res = this->numNodes;
    bool trans = workset.transpose_dist_param_deriv;
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
      for (int node = 0; node < num_deriv; ++node) {
        const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
        const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
        const LO p_lid= p_indexer->getLocalElement(ginode);
        double pvec_id = ( p_lid >= 0) ? pvec_constView[p_lid] : 0;

        ParamScalarT v(num_deriv, node, pvec_id);
        if(p_lid < 0) {
          v.fastAccessDx(node) = 0;
        }
        (this->val)(cell,node) = v;
      }

      if (workset.Vp != Teuchos::null) {
        const std::size_t num_cols = workset.Vp->domain()->dim();
        auto Vp_data = Albany::getLocalData(workset.Vp);

        Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp = workset.local_Vp[cell];

        if (trans) {
          local_Vp.resize(num_nodes_res*workset.numEqs);
          for (int node = 0; node < num_nodes_res; ++node) {
            // Store Vp entries
            for (std::size_t eq = 0; eq < workset.numEqs; eq++) {
              local_Vp[node*workset.numEqs+eq].resize(num_cols);
              const LO id = nodeID(cell,node,eq);
              for (std::size_t col=0; col<num_cols; ++col)
                local_Vp[node*workset.numEqs+eq][col] = Vp_data[col][id];
            }
          }
        } else {
          local_Vp.resize(num_deriv);
          for (int node = 0; node < num_deriv; ++node) {
            const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
            const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
            const LO id = p_indexer->getLocalElement(ginode);
            local_Vp[node].resize(num_cols);
            for (std::size_t col=0; col<num_cols; ++col) {
              local_Vp[node][col] = (id >= 0) ? Vp_data[col][id] : 0;
            }
          }
        }
      }
    }
  } else {
    // If not active, just set the parameter value in the phalanx field
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
        const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
        const LO p_lid= p_indexer->getLocalElement(ginode);
        (this->val)(cell,node) = ( p_lid >= 0) ? pvec_constView[p_lid] : 0;
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

  // Distributed parameter vector
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  // Direction vector for the Hessian-vector product
  Teuchos::RCP<const Thyra_MultiVector> vvec = workset.hessianWorkset.direction_p;

  auto nodeID = workset.wsElNodeEqID;
  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool g_pp_is_active = !workset.hessianWorkset.hess_vec_prod_g_pp.is_null();
  bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();
  bool f_pp_is_active = !workset.hessianWorkset.hess_vec_prod_f_pp.is_null();

  // is_p_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_px, Hv_g_pp, Hv_f_px, or Hv_f_pp, i.e. if the first derivative is w.r.t. this parameter.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  const bool is_p_active = (workset.dist_param_deriv_name == this->param_name)
    && (g_px_is_active||g_pp_is_active||f_px_is_active||f_pp_is_active);

  // is_p_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xp, Hv_g_pp, Hv_f_xp, or Hv_f_pp, i.e. if the second derivative is w.r.t. this parameter direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_p_direction_active = (workset.hessianWorkset.dist_param_deriv_direction_name == this->param_name)
    && (g_xp_is_active || g_pp_is_active || f_xp_is_active || f_pp_is_active);

  Teuchos::ArrayRCP<const ST> vvec_constView;
  if(is_p_direction_active) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        vvec.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in GatherScalarNodalParameter<HessianVec, Traits>: "
        "direction_p is not set and the direction is active.\n");
    vvec_constView = Albany::getLocalData(vvec->col(0));
  }

  const int num_nodes = this->numNodes;

  const int num_deriv = (this->val)(0,0).size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (int node = 0; node < num_nodes; ++node) {

      // Initialize Fad type for parameter value
      const LO id = wsElDofs((int)cell,(int)node,0);
      RealType pvec_val = (id >= 0) ? pvec_constView[id] : 0;

      auto val = (this->val)(cell,node);
      val = FadType(num_deriv, pvec_val);
      // If we differentiate w.r.t. this parameter, we have to set the first
      // derivative to 1
      if (is_p_active)
        val.fastAccessDx(node).val() = 1;
      // If we differentiate w.r.t. this parameter direction, we have to set
      // the second derivative to the related direction value
      if (is_p_direction_active)
        val.val().fastAccessDx(0) = (id >= 0) ? vvec_constView[id] : 0;
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

  // TODO: find a way to abstract away from the map concept. Perhaps using Panzer::ConnManager?
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  // Direction vector for the Hessian-vector product
  Teuchos::RCP<const Thyra_MultiVector> vvec = workset.hessianWorkset.direction_p;

  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  auto nodeID = workset.wsElNodeEqID;

  // If active, intialize data needed for differentiation
  auto overlapNodeVS = workset.disc->getOverlapNodeVectorSpace();
  auto ov_node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);
  auto p_indexer = Albany::createGlobalLocalIndexer(pvec->space());

  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool g_pp_is_active = !workset.hessianWorkset.hess_vec_prod_g_pp.is_null();
  bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();
  bool f_pp_is_active = !workset.hessianWorkset.hess_vec_prod_f_pp.is_null();

  // is_p_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_px, Hv_g_pp, Hv_f_px, or Hv_f_pp, i.e. if the first derivative is w.r.t. this parameter.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .dx().fastAccessDx().
  const bool is_p_active = (workset.dist_param_deriv_name == this->param_name)
    && (g_px_is_active||g_pp_is_active||f_px_is_active||f_pp_is_active);

  // is_p_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xp, Hv_g_pp, Hv_f_xp, or Hv_f_pp, i.e. if the second derivative is w.r.t. this parameter direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .fastAccessDx().dx().
  const bool is_p_direction_active = (workset.hessianWorkset.dist_param_deriv_direction_name == this->param_name)
    && (g_xp_is_active || g_pp_is_active || f_xp_is_active || f_pp_is_active);

  Teuchos::ArrayRCP<const ST> vvec_constView;
  if(is_p_direction_active) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        vvec.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in GatherScalarExtruded2DNodalParameter<HessianVec, Traits>: "
        "direction_p is not set and the direction is acrive.\n");
    vvec_constView = Albany::getLocalData(vvec->col(0));
  }

  const int num_deriv = this->numNodes;
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    for (int node = 0; node < num_deriv; ++node) {
      const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
      const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
      const LO p_lid= p_indexer->getLocalElement(ginode);
      RealType pvec_val = (p_lid >= 0) ? pvec_constView[p_lid] : 0;

      auto val = (this->val)(cell,node);
      val = FadType(val.size(), pvec_val);
      // If we differentiate w.r.t. this parameter, we have to set the first
      // derivative to 1
      if (is_p_active)
        val.fastAccessDx(node).val() = 1;
      // If we differentiate w.r.t. this parameter direction, we have to set
      // the second derivative to the related direction value
      if (is_p_direction_active)
        val.val().fastAccessDx(0) = (p_lid >= 0) ? vvec_constView[p_lid] : 0;
    }
  }
}

} // namespace PHAL
