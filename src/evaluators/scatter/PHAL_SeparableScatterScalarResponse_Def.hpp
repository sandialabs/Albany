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
// Base Class Generic Implemtation
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
SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
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
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* response derivative
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_MultiVector> dgdx = workset.overlapped_dgdx;
  Teuchos::RCP<Thyra_MultiVector> dgdxdot = workset.overlapped_dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> dg;
  if (dgdx != Teuchos::null) {
    dg = dgdx;
  } else {
    dg = dgdxdot;
  }

  auto dg_data = Albany::getNonconstLocalData(dg);

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    // Loop over responses

    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      auto val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (int node_dof=0; node_dof<numNodes; node_dof++) {
        int neq = nodeID.extent(2);

        // Loop over equations per node
        for (int eq_dof=0; eq_dof<neq; eq_dof++) {

          // local derivative component
          int deriv = neq * node_dof + eq_dof;

          // local DOF
          int dof = nodeID(cell,node_dof,eq_dof);

          // Set dg/dx
          // NOTE: mv local data is in column major
          dg_data[res][dof] += val.dx(deriv);

        } // column equations
      } // column nodes
    } // response

  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluate2DFieldsDerivativesDueToExtrudedSolution(typename Traits::EvalData workset, std::string& sideset, Teuchos::RCP<const CellTopologyData> cellTopo)
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

  const int neq = workset.wsElNodeEqID.extent(2);
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  if (workset.sideSets == Teuchos::null)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets not properly specified on the mesh" << std::endl);

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(sideset);

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    auto ov_node_indexer = workset.disc->getOverlapNodeGlobalLocalIndexer();

    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  cellTopo->side[elem_side];
      int numSideNodes = side.topology->node_count;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      for (std::size_t res = 0; res < this->global_response.size(); res++) {
        auto val = this->local_response(elem_LID, res);
        GO base_id;
        for (int i = 0; i < numSideNodes; ++i) {
          std::size_t node = side.node[i];
          base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
          for (GO il_col=0; il_col<numLayers+1; il_col++) {
            const GO ginode = layeredMeshNumbering.getId(base_id, il_col);
            const LO  inode = ov_node_indexer->getLocalElement(ginode);
            for (int eq_col=0; eq_col<neq; eq_col++) {
              const LO dof = solDOFManager.getLocalDOF(inode, eq_col);
              int deriv = neq *this->numNodes+il_col*neq*numSideNodes + neq*i + eq_col;
              dg_data[res][dof] += val.dx(deriv);
            }
          }
        }
      }
    }
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* response
  Teuchos::RCP<Thyra_Vector> g = workset.g;
  if (g != Teuchos::null) {
    Teuchos::ArrayRCP<ST> g_nonconstView = Albany::getNonconstLocalData(g);
    for (PHAL::MDFieldIterator<const ScalarT> gr(this->global_response);
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
SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
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
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* response derivative
  Teuchos::RCP<Thyra_MultiVector> dgdp = workset.overlapped_dgdp;

  if (dgdp.is_null()) {
    return;
  }

  auto dgdp_data = Albany::getNonconstLocalData(dgdp);

  int num_deriv = numNodes;

  // Loop over cells in workset

  const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
     // ScalarT& val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (int deriv=0; deriv<num_deriv; ++deriv) {
        const int row = wsElDofs((int)cell,deriv,0);

        // Set dg/dp
        if(row >=0){
          dgdp_data[res][row] += this->local_response(cell, res).dx(deriv);
        }
      } // deriv
    } // response
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  Teuchos::RCP<Thyra_Vector> g = workset.g;
  if (g != Teuchos::null) {
    Teuchos::ArrayRCP<double> g_nonconstView = Albany::getNonconstLocalData(g);
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      g_nonconstView[res] = this->global_response[res].val();
    }
  }

  Teuchos::RCP<Thyra_MultiVector> dgdp = workset.dgdp;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdp = workset.overlapped_dgdp;
  if (!dgdp.is_null() && !overlapped_dgdp.is_null()) {
    workset.p_cas_manager->combine(overlapped_dgdp, dgdp, Albany::CombineMode::ADD);
  }
}

// **********************************************************************
template<typename Traits>
void SeparableScatterScalarResponseWithExtrudedParams<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  if(level_it == extruded_params_levels->end()) //if parameter is not extruded use usual scatter.
    return SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(workset);

  // Here we scatter the *local* response derivative
  Teuchos::RCP<Thyra_MultiVector> dgdp = workset.overlapped_dgdp;

  if (dgdp.is_null()) {
    return;
  }

  auto dgdp_data = Albany::getNonconstLocalData(dgdp);

  int num_deriv = this->numNodes;
  auto nodeID = workset.wsElNodeEqID;
  int fieldLevel = level_it->second;

  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  auto overlap_p_vs = workset.distParamLib->get(workset.dist_param_deriv_name)->overlap_vector_space();
  auto ov_node_indexer = workset.disc->getOverlapNodeGlobalLocalIndexer();
  auto ov_p_indexer = workset.disc->getOverlapGlobalLocalIndexer(workset.dist_param_deriv_name);

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {

      // Loop over nodes in cell
      for (int deriv=0; deriv<num_deriv; ++deriv) {
        const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[deriv]);
        const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
        const LO row = ov_p_indexer->getLocalElement(ginode);

        // Set dg/dp
        if(row >=0){
          dgdp_data[res][row] += this->local_response(cell, res).dx(deriv);
        }
      } // deriv
    } // response
  } // cell
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************
template<typename Traits>
SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xx = workset.hessianWorkset.hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xp = workset.hessianWorkset.hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_px = workset.hessianWorkset.hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_pp = workset.hessianWorkset.hess_vec_prod_g_pp;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_xx = workset.hessianWorkset.overlapped_hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_xp = workset.hessianWorkset.overlapped_hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_px = workset.hessianWorkset.overlapped_hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_pp = workset.hessianWorkset.overlapped_hess_vec_prod_g_pp;
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
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // First, the function checks whether the parameter associated to workset.dist_param_deriv_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, workset.dist_param_deriv_name);

  // Here we scatter the *local* response derivative
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xx = workset.hessianWorkset.overlapped_hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xp = workset.hessianWorkset.overlapped_hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_px = workset.hessianWorkset.overlapped_hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_pp = workset.hessianWorkset.overlapped_hess_vec_prod_g_pp;

  if (hess_vec_prod_g_xx.is_null() && hess_vec_prod_g_xp.is_null() &&
      hess_vec_prod_g_px.is_null() && hess_vec_prod_g_pp.is_null()) {
    return;
  }
  if(!hess_vec_prod_g_xx.is_null())
  {
    auto hess_vec_prod_g_xx_data = Albany::getNonconstLocalData(hess_vec_prod_g_xx);

    // Loop over cells in workset
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      // Loop over responses

      for (std::size_t res = 0; res < this->global_response.size(); res++) {
        auto val = this->local_response(cell, res);

        // Loop over nodes in cell
        for (int node_dof=0; node_dof<numNodes; node_dof++) {
          int neq = nodeID.extent(2);

          // Loop over equations per node
          for (int eq_dof=0; eq_dof<neq; eq_dof++) {

            // local derivative component
            int deriv = neq * node_dof + eq_dof;

            // local DOF
            int dof = nodeID(cell,node_dof,eq_dof);

            hess_vec_prod_g_xx_data[res][dof] += val.dx(deriv).dx(0);

          } // column equations
        } // column nodes
      } // response
    } // cell
  }
  if(!hess_vec_prod_g_xp.is_null())
  {
    auto hess_vec_prod_g_xp_data = Albany::getNonconstLocalData(hess_vec_prod_g_xp);

    // Loop over cells in workset
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      // Loop over responses

      for (std::size_t res = 0; res < this->global_response.size(); res++) {
        auto val = this->local_response(cell, res);

        // Loop over nodes in cell
        for (int node_dof=0; node_dof<numNodes; node_dof++) {
          int neq = nodeID.extent(2);

          // Loop over equations per node
          for (int eq_dof=0; eq_dof<neq; eq_dof++) {

            // local derivative component
            int deriv = neq * node_dof + eq_dof;

            // local DOF
            int dof = nodeID(cell,node_dof,eq_dof);

            hess_vec_prod_g_xp_data[res][dof] += val.dx(deriv).dx(0);

          } // column equations
        } // column nodes
      } // response
    } // cell
  }
  if (!hess_vec_prod_g_px.is_null())
  {
    auto hess_vec_prod_g_px_data = Albany::getNonconstLocalData(hess_vec_prod_g_px);

    int num_deriv = numNodes;

    if (l1_is_distributed) {
      const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];

      // Loop over cells in workset
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {

        // Loop over responses
        for (std::size_t res = 0; res < this->global_response.size(); res++) {
        // ScalarT& val = this->local_response(cell, res);

          // Loop over nodes in cell
          for (int deriv=0; deriv<num_deriv; ++deriv) {
            const int row = wsElDofs((int)cell,deriv,0);

            // Set hess_vec_prod_g_px
            if(row >=0){
              hess_vec_prod_g_px_data[res][row] += this->local_response(cell, res).dx(deriv).dx(0);
            }
          } // deriv
        } // response
      } // cell
    }
    else {
      // Loop over cells in workset
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {

        // Loop over responses
        for (std::size_t res = 0; res < this->global_response.size(); res++) {

          // Loop over nodes in cell
          for (int deriv=0; deriv<hess_vec_prod_g_px_data[res].size(); ++deriv) {
            hess_vec_prod_g_px_data[res][deriv] += this->local_response(cell, res).dx(deriv).dx(0);
          } // deriv
        } // response
      } // cell
    }
  }
  if (!hess_vec_prod_g_pp.is_null())
  {
    auto hess_vec_prod_g_pp_data = Albany::getNonconstLocalData(hess_vec_prod_g_pp);

    int num_deriv = numNodes;

    if (l1_is_distributed) {
      const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];

      // Loop over cells in workset
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {

        // Loop over responses
        for (std::size_t res = 0; res < this->global_response.size(); res++) {
        // ScalarT& val = this->local_response(cell, res);

          // Loop over nodes in cell
          for (int deriv=0; deriv<num_deriv; ++deriv) {
            const int row = wsElDofs((int)cell,deriv,0);

            // Set hess_vec_prod_g_pp
            if(row >=0){
              hess_vec_prod_g_pp_data[res][row] += this->local_response(cell, res).dx(deriv).dx(0);
            }
          } // deriv
        } // response
      } // cell
    }
    else {
      // Loop over cells in workset
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {

        // Loop over responses
        for (std::size_t res = 0; res < this->global_response.size(); res++) {

          // Loop over nodes in cell
          for (int deriv=0; deriv<hess_vec_prod_g_pp_data[res].size(); ++deriv) {
            hess_vec_prod_g_pp_data[res][deriv] += this->local_response(cell, res).dx(deriv).dx(0);
          } // deriv
        } // response
      } // cell
    }
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // First, the function checks whether the parameter associated to workset.dist_param_deriv_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, workset.dist_param_deriv_name);

  Teuchos::RCP<Thyra_Vector> g = workset.g;
  if (g != Teuchos::null) {
    Teuchos::ArrayRCP<double> g_nonconstView = Albany::getNonconstLocalData(g);
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      g_nonconstView[res] = this->global_response[res].val().val();
    }
  }

  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xx = workset.hessianWorkset.hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xp = workset.hessianWorkset.hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_px = workset.hessianWorkset.hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_pp = workset.hessianWorkset.hess_vec_prod_g_pp;

  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_xx = workset.hessianWorkset.overlapped_hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_xp = workset.hessianWorkset.overlapped_hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_px = workset.hessianWorkset.overlapped_hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_pp = workset.hessianWorkset.overlapped_hess_vec_prod_g_pp;

  if (!hess_vec_prod_g_xx.is_null() && !overlapped_hess_vec_prod_g_xx.is_null()) {
    workset.x_cas_manager->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, Albany::CombineMode::ADD);
  }
  if (!hess_vec_prod_g_xp.is_null() && !overlapped_hess_vec_prod_g_xp.is_null()) {
    workset.x_cas_manager->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, Albany::CombineMode::ADD);
  }
  if (l1_is_distributed) {
    if (!hess_vec_prod_g_px.is_null() && !overlapped_hess_vec_prod_g_px.is_null()) {
      workset.p_cas_manager->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, Albany::CombineMode::ADD);
    }
    if (!hess_vec_prod_g_pp.is_null() && !overlapped_hess_vec_prod_g_pp.is_null()) {
      workset.p_cas_manager->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, Albany::CombineMode::ADD);
    }
  }
  else {
    if (!hess_vec_prod_g_px.is_null() && !overlapped_hess_vec_prod_g_px.is_null()) {
      auto tmp = Thyra::createMembers(workset.p_cas_manager->getOwnedVectorSpace(),overlapped_hess_vec_prod_g_px->domain()->dim());
      workset.p_cas_manager->combine(overlapped_hess_vec_prod_g_px, tmp, Albany::CombineMode::ADD);
      workset.p_cas_manager->scatter(tmp, hess_vec_prod_g_px, Albany::CombineMode::INSERT);
    }
    if (!hess_vec_prod_g_pp.is_null() && !overlapped_hess_vec_prod_g_pp.is_null()) {
      auto tmp = Thyra::createMembers(workset.p_cas_manager->getOwnedVectorSpace(),overlapped_hess_vec_prod_g_pp->domain()->dim());
      workset.p_cas_manager->combine(overlapped_hess_vec_prod_g_pp, tmp, Albany::CombineMode::ADD);
      workset.p_cas_manager->scatter(tmp, hess_vec_prod_g_pp, Albany::CombineMode::INSERT);
    }
  }
}

// **********************************************************************
template<typename Traits>
void SeparableScatterScalarResponseWithExtrudedParams<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  if(level_it == extruded_params_levels->end()) //if parameter is not extruded use usual scatter.
    return SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>::evaluateFields(workset);

  // Here we scatter the *local* response derivative
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xx = workset.hessianWorkset.overlapped_hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xp = workset.hessianWorkset.overlapped_hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_px = workset.hessianWorkset.overlapped_hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_pp = workset.hessianWorkset.overlapped_hess_vec_prod_g_pp;

  if (hess_vec_prod_g_px.is_null() && hess_vec_prod_g_pp.is_null()) {
    return;
  }
  if(!hess_vec_prod_g_px.is_null())
  {
    auto hess_vec_prod_g_px_data = Albany::getNonconstLocalData(hess_vec_prod_g_px);

    int num_deriv = this->numNodes;
    int fieldLevel = level_it->second;

    const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    auto overlap_p_vs = workset.distParamLib->get(workset.dist_param_deriv_name)->overlap_vector_space();
    auto ov_p_indexer = Albany::createGlobalLocalIndexer(overlap_p_vs);

    // Loop over cells in workset
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];

      // Loop over responses
      for (std::size_t res = 0; res < this->global_response.size(); res++) {

        // Loop over nodes in cell
        for (int deriv=0; deriv<num_deriv; ++deriv) {
          const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[deriv]);
          const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
          const LO row = ov_p_indexer->getLocalElement(ginode);

          // Set dg/dp
          if(row >=0){
            hess_vec_prod_g_px_data[res][row] += this->local_response(cell, res).dx(deriv).dx(0);
          }
        } // deriv
      } // response
    } // cell
  }
  if(!hess_vec_prod_g_pp.is_null())
  {
    auto hess_vec_prod_g_pp_data = Albany::getNonconstLocalData(hess_vec_prod_g_pp);

    int num_deriv = this->numNodes;
    int fieldLevel = level_it->second;

    const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    auto overlap_p_vs = workset.distParamLib->get(workset.dist_param_deriv_name)->overlap_vector_space();
    auto ov_p_indexer = Albany::createGlobalLocalIndexer(overlap_p_vs);

    // Loop over cells in workset
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];

      // Loop over responses
      for (std::size_t res = 0; res < this->global_response.size(); res++) {

        // Loop over nodes in cell
        for (int deriv=0; deriv<num_deriv; ++deriv) {
          const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[deriv]);
          const GO ginode = layeredMeshNumbering.getId(base_id, fieldLevel);
          const LO row = ov_p_indexer->getLocalElement(ginode);

          // Set dg/dp
          if(row >=0){
            hess_vec_prod_g_pp_data[res][row] += this->local_response(cell, res).dx(deriv).dx(0);
          }
        } // deriv
      } // response
    } // cell
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluate2DFieldsDerivativesDueToExtrudedSolution(typename Traits::EvalData workset, std::string& sideset, Teuchos::RCP<const CellTopologyData> cellTopo)
{
  // Here we scatter the *local* response derivative
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xx = workset.hessianWorkset.overlapped_hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xp = workset.hessianWorkset.overlapped_hess_vec_prod_g_xp;

  if (hess_vec_prod_g_xx.is_null() && hess_vec_prod_g_xp.is_null())
    return;

  const int neq = workset.wsElNodeEqID.extent(2);
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  if (workset.sideSets == Teuchos::null)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets not properly specified on the mesh" << std::endl);

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(sideset);

  if(!hess_vec_prod_g_xx.is_null())
  {
    auto hess_vec_prod_g_xx_data = Albany::getNonconstLocalData(hess_vec_prod_g_xx);

    if (it != ssList.end()) {
      const std::vector<Albany::SideStruct>& sideSet = it->second;

      auto overlapNodeVS = workset.disc->getOverlapNodeVectorSpace();
      auto ov_node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);

      for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name
        // Get the data that corresponds to the side
        const int elem_LID = sideSet[iSide].elem_LID;
        const int elem_side = sideSet[iSide].side_local_id;
        const CellTopologyData_Subcell& side =  cellTopo->side[elem_side];
        int numSideNodes = side.topology->node_count;

        const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
        for (std::size_t res = 0; res < this->global_response.size(); res++) {
          auto val = this->local_response(elem_LID, res);
          GO base_id;
          for (int i = 0; i < numSideNodes; ++i) {
            std::size_t node = side.node[i];
            base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
            for (int il_col=0; il_col<numLayers+1; il_col++) {
              const GO ginode = layeredMeshNumbering.getId(base_id, il_col);
              const LO  inode = ov_node_indexer->getLocalElement(ginode);
              for (int eq_col=0; eq_col<neq; eq_col++) {
                const LO dof = solDOFManager.getLocalDOF(inode, eq_col);
                int deriv = neq *this->numNodes+il_col*neq*numSideNodes + neq*i + eq_col;
                hess_vec_prod_g_xx_data[res][dof] += val.dx(deriv).dx(0);
              }
            }
          }
        }
      }
    }
  }
  if(!hess_vec_prod_g_xp.is_null())
  {
    auto hess_vec_prod_g_xp_data = Albany::getNonconstLocalData(hess_vec_prod_g_xp);

    if (it != ssList.end()) {
      const std::vector<Albany::SideStruct>& sideSet = it->second;

      auto overlapNodeVS = workset.disc->getOverlapNodeVectorSpace();
      auto ov_node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);

      for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name
        // Get the data that corresponds to the side
        const int elem_LID = sideSet[iSide].elem_LID;
        const int elem_side = sideSet[iSide].side_local_id;
        const CellTopologyData_Subcell& side =  cellTopo->side[elem_side];
        int numSideNodes = side.topology->node_count;

        const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
        for (std::size_t res = 0; res < this->global_response.size(); res++) {
          auto val = this->local_response(elem_LID, res);
          GO base_id;
          for (int i = 0; i < numSideNodes; ++i) {
            std::size_t node = side.node[i];
            base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
            for (int il_col=0; il_col<numLayers+1; il_col++) {
              const GO ginode = layeredMeshNumbering.getId(base_id, il_col);
              const LO  inode = ov_node_indexer->getLocalElement(ginode);
              for (int eq_col=0; eq_col<neq; eq_col++) {
                const LO dof = solDOFManager.getLocalDOF(inode, eq_col);
                int deriv = neq *this->numNodes+il_col*neq*numSideNodes + neq*i + eq_col;
                hess_vec_prod_g_xp_data[res][dof] += val.dx(deriv).dx(0);
              }
            }
          }
        }
      }
    }
  }
}

} // namespace PHAL
